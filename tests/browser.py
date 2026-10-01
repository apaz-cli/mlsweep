"""A minimal headless-Chromium driver for dashboard tests.

Speaks the Chrome DevTools Protocol over the stdlib WebSocket client in
``mlsweep.run_sweep``, so the tests need no extra packages.  ``find_chromium``
returns None when no browser is installed; callers skip in that case.
"""

import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

from conftest import _find_free_port
from mlsweep.run_sweep import _WS_OP_TEXT, _WebSocket

_CANDIDATES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome")


def find_chromium():
    """Path to a Chromium-family browser ($MLSWEEP_TEST_CHROME first), or None."""
    env = os.environ.get("MLSWEEP_TEST_CHROME")
    if env:
        return env
    for name in _CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


class Browser:
    """One headless browser with one page, driven synchronously."""

    def __init__(self, binary):
        self._profile = tempfile.mkdtemp(prefix="mlsweep-chrome-")
        port = _find_free_port()
        self._proc = subprocess.Popen(
            [binary, "--headless=new", "--no-sandbox", "--disable-gpu",
             "--no-first-run", "--no-default-browser-check",
             f"--remote-debugging-port={port}", f"--user-data-dir={self._profile}",
             "--window-size=1300,800", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 20
        while True:
            try:
                req = urllib.request.Request(f"{base}/json/new?about:blank", method="PUT")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    target = json.loads(resp.read())
                break
            except Exception:
                if time.time() > deadline:
                    self.close()
                    raise RuntimeError("Chromium did not start")
                time.sleep(0.2)
        self._ws = _WebSocket(target["webSocketDebuggerUrl"], token="", timeout=30.0)
        self._ws.connect()
        self._ids = itertools.count(1)
        self.call("Page.enable")
        self.call("Runtime.enable")

    def call(self, method, **params):
        """Send one CDP command and return its result (events are skipped)."""
        msg_id = next(self._ids)
        self._ws.send_frame(_WS_OP_TEXT, json.dumps(
            {"id": msg_id, "method": method, "params": params}).encode())
        while True:
            frame = self._ws.recv_frame()
            if frame is None:
                raise RuntimeError("DevTools connection closed")
            opcode, payload = frame
            if opcode != _WS_OP_TEXT:
                continue
            msg = json.loads(payload)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr):
        """Evaluate *expr* in the page (awaiting promises) and return its value."""
        res = self.call("Runtime.evaluate", expression=expr,
                        awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in res:
            raise RuntimeError(f"JS error in {expr!r}: {res['exceptionDetails']}")
        return res["result"].get("value")

    def wait_for(self, expr, timeout=15.0, interval=0.1):
        """Poll *expr* until it is truthy; return its value or raise TimeoutError."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = self.eval(expr)
            except RuntimeError:
                last = None
            if last:
                return last
            time.sleep(interval)
        raise TimeoutError(f"timed out waiting for {expr!r} (last value {last!r})")

    def goto(self, url):
        """Load *url* and wait for the document to finish loading."""
        self.call("Page.navigate", url=url)
        time.sleep(0.1)
        self.wait_for(f"location.href.startsWith({json.dumps(url.split('?')[0])}) "
                      "&& document.readyState === 'complete'")

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        shutil.rmtree(self._profile, ignore_errors=True)
