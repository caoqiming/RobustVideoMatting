import http.server
import socketserver
import ssl
import os

PORT = 443  # HTTPS 默认端口
HTTP_PORT = 80  # HTTP 重定向端口
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


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


def run_https_server():
    cert_file, key_file = generate_self_signed_cert()

    if not cert_file or not key_file:
        print("\n无法启动 HTTPS 服务器，使用 HTTP 模式（摄像头可能无法访问）")
        print(f"HTTP 服务器运行在 http://localhost:{HTTP_PORT}")
        with socketserver.TCPServer(("", HTTP_PORT), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
        return

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        # 配置 SSL
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        print(f"✓ HTTPS 服务器运行在 https://localhost:{PORT}")
        print(f"✓ 服务目录: {DIRECTORY}")
        print("\n重要提示：")
        print("1. 首次访问会提示证书不安全，这是正常的（自签名证书）")
        print("2. 在浏览器中点击\"高级\" -> \"继续访问\"即可")
        print("3. 手机访问请使用: https://你的电脑IP地址")
        print("4. 手机也需要接受不安全的证书警告")
        print("\n按 Ctrl+C 停止服务器")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")


if __name__ == "__main__":
    # 注意：在Unix/Linux/Mac上，使用443端口需要root权限
    # 可以使用 sudo python server.py 运行
    # 或者修改PORT为8443等不需要特殊权限的端口
    run_https_server()
