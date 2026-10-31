"""Local development server: serves public/ and the /api/predict endpoint.

Run with:  python run_local.py   then open http://localhost:8000
"""
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

from api import predict as P

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")


class Local(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC, **kwargs)

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/predict":
            return self._json(200, P.config())
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] == "/api/predict":
            length = int(self.headers.get("Content-Length", 0) or 0)
            code, payload = P.handle_post(self.rfile.read(length))
            return self._json(code, payload)
        self.send_error(404)


if __name__ == "__main__":
    print("Open http://localhost:8000  (press Ctrl+C to stop)")
    HTTPServer(("", 8000), Local).serve_forever()