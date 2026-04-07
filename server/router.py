#!/usr/bin/env python3
"""
Model router for dual Nemotron serving.
Routes OpenAI-compatible API requests to the correct vLLM backend
based on the "model" field in the request body.

Usage: python3 router.py [port]   (default: 8080)
"""
import http.server
import json
import socketserver
import sys
import urllib.error
import urllib.request

BACKENDS = {
    "nemotron-3-super": "http://localhost:8000",
    "nemotron-3-nano": "http://localhost:8002",
}
DEFAULT_BACKEND = "http://localhost:8000"
LISTEN_PORT = 8080


class Router(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[router] {self.address_string()} {fmt % args}\n")

    def do_GET(self):
        if self.path == "/v1/models":
            return self._merge_models()
        if self.path == "/health":
            return self._health()
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    # ---- routing logic ----

    def _resolve_backend(self, body):
        if body:
            try:
                model = json.loads(body).get("model", "")
                for name, url in BACKENDS.items():
                    if name in model:
                        return url
            except (json.JSONDecodeError, AttributeError):
                pass
        return DEFAULT_BACKEND

    # ---- reverse proxy (handles streaming) ----

    def _proxy(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        backend = self._resolve_backend(body)
        url = f"{backend}{self.path}"

        req = urllib.request.Request(url, data=body, method=self.command)
        for key, val in self.headers.items():
            if key.lower() not in ("host", "transfer-encoding", "content-length"):
                req.add_header(key, val)
        if body:
            req.add_header("Content-Length", str(len(body)))

        try:
            resp = urllib.request.urlopen(req)
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding",):
                    self.send_header(key, val)
            self.end_headers()
            while chunk := resp.read(4096):
                self.wfile.write(chunk)
                self.wfile.flush()
            resp.close()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except urllib.error.URLError:
            self._send_json(503, {
                "error": {"message": f"Backend {backend} not ready", "type": "service_unavailable"}
            })

    # ---- aggregated endpoints ----

    def _merge_models(self):
        models = []
        for _name, url in BACKENDS.items():
            try:
                with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as resp:
                    models.extend(json.loads(resp.read()).get("data", []))
            except Exception:
                pass
        self._send_json(200, {"object": "list", "data": models})

    def _health(self):
        statuses = {}
        for name, url in BACKENDS.items():
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                    statuses[name] = "healthy" if resp.status == 200 else "unhealthy"
            except Exception:
                statuses[name] = "unavailable"
        all_ok = all(s == "healthy" for s in statuses.values())
        self._send_json(200 if all_ok else 503, {"healthy": all_ok, "backends": statuses})

    # ---- helpers ----

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    server = ThreadedServer(("0.0.0.0", port), Router)
    print(f"[router] Listening on http://0.0.0.0:{port}")
    print(f"[router] Backends: {json.dumps(BACKENDS, indent=2)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[router] Shutting down")
        server.shutdown()
