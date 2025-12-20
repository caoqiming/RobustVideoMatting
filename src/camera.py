import cv2
import torch
import abc
import time
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
        return torch.from_numpy(frame).permute(
            2, 0, 1).unsqueeze(0).float()  # B C H W

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

    def _restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        if time_interval:
            time.sleep(time_interval)
        ret, frame = self.cap.read()
        if not ret:
            self._restart()
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame after restart.")
        # B C H W
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()

    def __del__(self):
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            self.cap.release()


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

    def start(self):
        infer = RealtimeInference(
            'mobilenetv3', './model_data/rvm_mobilenetv3.pth', self.device)
        while True:
            frame_tensor = self.foreground_source.read_next_frame()
            fgr, pha = infer.infere(frame_tensor)
            print(pha)


if __name__ == "__main__":
    app = App()
    app.start()
