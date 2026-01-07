import cv2
import torch
import abc
import time
import torch.nn.functional as F
from model import MattingNetwork


class RealtimeInference:
    def auto_downsample_ratio(self, h, w):
        """
        Automatically find a downsample ratio so that the largest side of the resolution be 512px.
        """
        return min(512 / max(h, w), 1)

    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device
        self.rec = [None] * 4
        self.downsample_ratio = None

    def infere(self, frame: torch.Tensor):
        if self.downsample_ratio is None:
            self.downsample_ratio = self.auto_downsample_ratio(
                *frame.shape[2:])
        frame = frame.to(self.device, torch.float32, non_blocking=True).unsqueeze(
            0)  # [B, T, C, H, W]
        fgr, pha, *self.rec = self.model(frame,
                                         *self.rec, self.downsample_ratio)
        return fgr, pha


class VideoSource:
    @abc.abstractmethod
    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        raise NotImplementedError


class LocalCamera(VideoSource):
    def __init__(self, camera_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # B C H W, normalized to [0,1]
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

    def __del__(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class VideoFile(VideoSource):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {file_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        # Preload all frames into memory and release the capture
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(
                torch.from_numpy(frame).permute(
                    2, 0, 1).float().div(255.0)  # C,H,W in [0,1]
            )
        self.frame_count = len(self.frames)
        if self.frame_count == 0:
            raise RuntimeError("No frames decoded from video.")
        self.cap.release()
        self.cap = None
        # Accumulate time to compute index directly from interval
        self._time_accum = 0.0
        self._current_idx = 0

    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        # Compute frame index directly from interval and fps; fall back to sequential when interval is None
        if time_interval is not None and self.fps > 0:
            self._time_accum += time_interval
            idx = int(round(self._time_accum * self.fps)) % self.frame_count
        else:
            self._current_idx = (self._current_idx + 1) % self.frame_count
            idx = self._current_idx
        frame = self.frames[idx]
        return frame.unsqueeze(0)  # B,C,H,W

    def __del__(self):
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            self.cap.release()


class IPCamera(VideoSource):
    def __init__(self, url: str, retry_delay: float = 1.5):
        super().__init__()
        self.url = url
        self.retry_delay = retry_delay
        self.cap = self._open_capture()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")
        # keep last good frame; set flush controls
        self.last_frame = None
        self.flush_budget = 0.02  # seconds to spend flushing buffered frames
        self.max_flush = 8        # max grabs per call

    def _stream_candidates(self, url: str):
        """Generate common IP camera endpoints and http fallback (inlined)."""
        import urllib.parse as urlparse  # local import to avoid global changes
        u = urlparse.urlsplit(url)
        schemes = [u.scheme] if u.scheme else []
        if u.scheme == "https":
            schemes.append("http")  # many builds can't handle HTTPS; try HTTP
        if not schemes:
            schemes = ["rtsp", "http"]

        base_path = u.path or ""
        paths = [base_path] if base_path else [
            "", "/video", "/mjpeg", "/mjpegfeed", "/stream.mjpg", "/stream", "/live"
        ]
        candidates = []
        for s in schemes:
            for p in paths:
                candidates.append(urlparse.urlunsplit(
                    (s, u.netloc, p, u.query, u.fragment)))
        if u.netloc and "rtsp" in schemes:
            candidates.insert(0, f"rtsp://{u.netloc}{u.path or ''}")
        # de-duplicate while preserving order
        return list(dict.fromkeys(candidates))

    def _open_capture(self):
        for candidate in self._stream_candidates(self.url):
            cap = cv2.VideoCapture(candidate)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lower latency
            except Exception:
                pass
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        # Read latest frame; no reconnects, just flush buffered frames
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("IP camera stream is not opened.")

        # Flush buffered frames to reduce latency
        flushed = 0
        t0 = time.perf_counter()
        while flushed < self.max_flush and (time.perf_counter() - t0) < self.flush_budget:
            if not self.cap.grab():
                break
            flushed += 1

        # Retrieve the most recent frame
        ret, frame = self.cap.retrieve()
        if not ret:
            # Fallback to a direct read once
            ret, frame = self.cap.read()
        if not ret:
            if self.last_frame is None:
                raise RuntimeError("Failed to read frame.")
            frame = self.last_frame
        else:
            self.last_frame = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

    def __del__(self):
        cap = getattr(self, "cap", None)
        if cap is not None and cap.isOpened():
            cap.release()


class App:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f"Using device: {device}")
        self.device = device
        self.foreground_source: VideoSource = None
        self.background_source: VideoSource = None

    def set_foreground_source(self, source: VideoSource):
        if source is not None and not isinstance(source, VideoSource):
            raise TypeError("foreground_source must be a VideoSource")
        self.foreground_source = source

    def set_background_source(self, source: VideoSource):
        if source is not None and not isinstance(source, VideoSource):
            raise TypeError("background_source must be a VideoSource")
        self.background_source = source

    def start(self):
        infer = RealtimeInference(
            'mobilenetv3', './model_data/rvm_mobilenetv3.pth', self.device)
        prev_time = time.time()
        while True:
            now = time.time()
            interval = now - prev_time
            prev_time = now

            loop_start = now
            frame_tensor = self.foreground_source.read_next_frame(
                interval if interval > 0 else None)
            if frame_tensor is None:
                continue

            fgr, pha = infer.infere(frame_tensor)
            # Ensure outputs are NCHW (drop temporal dim if present)
            if fgr.dim() == 5:
                fgr = fgr[:, -1]
            if pha.dim() == 5:
                pha = pha[:, -1]

            background_tensor = self.background_source.read_next_frame(
                interval if interval > 0 else None)
            if background_tensor is None:
                continue

            bgr = background_tensor.to(self.device)
            if bgr.shape[2:] != fgr.shape[2:]:
                bgr = F.interpolate(
                    bgr, size=fgr.shape[2:], mode='bilinear', align_corners=False)

            com = fgr * pha + bgr * (1 - pha)
            com_display = (com[0].permute(
                1, 2, 0).cpu().numpy() * 255).astype('uint8')
            cv2.imshow('Composite', cv2.cvtColor(
                com_display, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.set_foreground_source(LocalCamera())
    # app.set_background_source(VideoFile('./video_data/galway.MP4'))
    app.set_background_source(IPCamera('http://192.168.0.32:8080'))
    app.start()
