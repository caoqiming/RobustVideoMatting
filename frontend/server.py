from src.app import *
import http.server
import socketserver
import ssl
import os
import asyncio
import base64
import json
import threading
import cv2
import numpy as np
import websockets

PORT = 443  # HTTPS 默认端口
HTTP_PORT = 80  # HTTP 重定向端口
WS_PORT = 8765
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
frame_data = {}  # 储存最近的一帧图片
ws_clients = set()  # 存储所有连接的 WebSocket 客户端
ws_loop = None  # 存储 WebSocket 事件循环的引用


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # 添加CORS头和安全头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()


def generate_self_signed_cert():
    """生成自签名证书"""
    cert_file = os.path.join(DIRECTORY, 'cert.pem')
    key_file = os.path.join(DIRECTORY, 'key.pem')

    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("使用现有证书文件")
        return cert_file, key_file

    print("生成自签名证书...")
    print("需要安装 openssl，请在终端运行以下命令：")
    print(
        f"openssl req -x509 -newkey rsa:4096 -nodes -out {cert_file} -keyout {key_file} -days 365 -subj '/CN=localhost'")
    print("\n或者使用 Python 生成（需要安装 cryptography 库）：")
    print("pip install cryptography")

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # 生成证书
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.IPAddress(u"127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # 保存证书
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # 保存私钥
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        print("✓ 证书生成成功！")
        return cert_file, key_file

    except ImportError:
        print("\n错误: 未安装 cryptography 库")
        print("请运行: pip install cryptography")
        return None, None


async def handle_ws(websocket):
    print("✓ WebSocket 客户端已连接")
    ws_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
                img_b64 = payload.get("data")
                camera = payload.get("camera", "camera")

                if not img_b64 or "," not in img_b64:
                    print(f"警告: 无效的图像数据格式")
                    continue

                _, b64data = img_b64.split(",", 1)

                try:
                    img_bytes = base64.b64decode(b64data)
                except Exception as e:
                    print(f"Base64 解码失败: {e}")
                    continue

                if not img_bytes:
                    print("警告: 接收到空的图像数据")
                    continue

                try:
                    frame = cv2.imdecode(np.frombuffer(
                        img_bytes, np.uint8), cv2.IMREAD_COLOR)

                    if frame is None:
                        print(f"警告: 无法解码图像，摄像头 {camera}，数据可能损坏")
                        continue

                    # 存储帧到 frame_data 字典
                    frame_data[camera] = frame
                    # print(f"✓ 接收到来自 {camera} 的帧，尺寸: {frame.shape}")

                except cv2.error as e:
                    print(f"OpenCV 错误 ({camera}): {e}")
                    continue

            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}")
            except Exception as e:
                print(f"处理帧失败 ({type(e).__name__}): {e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket 连接已关闭")
    finally:
        ws_clients.discard(websocket)
        print("WebSocket 处理器退出")


async def run_ws_server(cert_file, key_file):
    global ws_loop
    ws_loop = asyncio.get_event_loop()
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(cert_file, key_file)
    print(f"✓ WebSocket 服务器运行在 wss://localhost:{WS_PORT}")
    async with websockets.serve(handle_ws, "", WS_PORT, ssl=ssl_ctx, max_size=8 * 1024 * 1024):
        await asyncio.Future()  # run forever


def start_https_thread(cert_file, key_file):
    def _serve():
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert_file, key_file)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            print(f"✓ HTTPS 服务器运行在 https://localhost:{PORT}")
            print(f"✓ 服务目录: {DIRECTORY}")
            print("\n按 Ctrl+C 停止服务器")
            httpd.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def start_ws_thread(cert_file, key_file):
    def _serve():
        asyncio.run(run_ws_server(cert_file, key_file))

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def start_https_server():
    cert_file, key_file = generate_self_signed_cert()
    if not cert_file or not key_file:
        print("\n无法启动 HTTPS 服务器，使用 HTTP 模式（摄像头可能无法访问）")
        print(f"HTTP 服务器运行在 http://localhost:{HTTP_PORT}")
        with socketserver.TCPServer(("", HTTP_PORT), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
        return

    start_https_thread(cert_file, key_file)
    start_ws_thread(cert_file, key_file)


class StreamFromWebsocket(VideoSource):
    def __init__(self, camera_name):
        super().__init__()
        self.camera_name = camera_name

    def read_next_frame(self, time_interval: float = None) -> torch.Tensor:
        if self.camera_name not in frame_data:
            return None

        frame = cv2.cvtColor(frame_data[self.camera_name], cv2.COLOR_BGR2RGB)
        # B C H W, normalized to [0,1]
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div(255.0)


def send_composite_to_clients(com_display):
    """将合成图像发送给所有连接的客户端"""
    if not ws_clients or ws_loop is None:
        return

    # 将图像编码为 JPEG
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(
        com_display, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_base64}"

    # 异步发送给所有客户端
    message = json.dumps({"type": "composite", "data": img_data_url})

    # 在 WebSocket 事件循环中执行广播
    async def broadcast():
        disconnected = set()
        for client in ws_clients.copy():
            try:
                await client.send(message)
            except Exception as e:
                print(f"发送失败: {e}")
                disconnected.add(client)
        for client in disconnected:
            ws_clients.discard(client)

    # 将协程提交到 WebSocket 线程的事件循环
    asyncio.run_coroutine_threadsafe(broadcast(), ws_loop)


if __name__ == "__main__":
    # 注意：在Unix/Linux/Mac上，使用443端口需要root权限
    # 可以使用 sudo python server.py 运行
    # 或者修改PORT为8443等不需要特殊权限的端口
    start_https_server()

    app = App()
    app.set_foreground_source(StreamFromWebsocket('front'))
    # app.set_background_source(VideoFile('./video_data/galway.MP4'))
    app.set_background_source(StreamFromWebsocket('rear'))
    app.set_composite_callback(send_composite_to_clients)  # 设置回调
    app.start()
