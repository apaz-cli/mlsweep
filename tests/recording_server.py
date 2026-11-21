"""A local HTTP server that stands in for the manager in client tests."""

import http.server
import json
import threading


class RecordingServer:
    """A local HTTP server that answers every request with one status and JSON body,
    and records ``(method, path)`` of each."""

    def __init__(self, status: int, body: dict):
        calls: list[tuple[str, str]] = []
        payload = json.dumps(body).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                calls.append((self.command, self.path))
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _answer

            def log_message(self, *args):
                pass

        self.calls = calls
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
