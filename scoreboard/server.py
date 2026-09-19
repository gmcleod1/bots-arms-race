"""A tiny local server for the scoreboard: the dashboard plus the public feed, nothing else.

Serves ONLY dashboard.html (at /), state.json and events.jsonl from the round directory, on
127.0.0.1. Everything else in the round directory is refused on purpose: the agent's log stays
sealed until the round ends, and outcomes.json lists every flagged account. CORS is open so a
separate overlay page (e.g. the stream-overlay app) can fetch the feed.

OBS: add a Browser Source at http://127.0.0.1:<port>/ (1920x1080). Options: ?compact=1 for just
the score and tiles, ?bg=transparent, ?theme=dark|light.
"""
from __future__ import annotations

import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASHBOARD = Path(__file__).with_name("dashboard.html")
PUBLIC = {"/state.json": ("state.json", "application/json"), "/events.jsonl": ("events.jsonl", "application/x-ndjson")}


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, round_dir: Path, *args, **kwargs) -> None:
        self.round_dir = round_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/dashboard.html"):
            self._send(DASHBOARD, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)  # nothing to serve; keeps the browser console clean
            self.end_headers()
        elif path in PUBLIC:
            name, ctype = PUBLIC[path]
            self._send(self.round_dir / name, ctype)
        else:
            self.send_error(404)

    def _send(self, file: Path, ctype: str) -> None:
        try:
            body = file.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep the console for the defender's REPL
        pass


def serve(round_dir: Path, port: int = 8430) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start serving in a background thread. Use port 0 to pick a free one (server.server_port)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler, round_dir))
    thread = threading.Thread(target=server.serve_forever, name="scoreboard-http", daemon=True)
    thread.start()
    return server, thread
