from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import webbrowser

os.environ.setdefault("PYTHONUTF8", "1")
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "readtome_mpl"))


def _open_browser_when_ready(url: str, delay: float = 1.2) -> None:
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="readtome", description="Start the local ReadToMe web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--open", action="store_true", help="Open in default browser after starting")
    args = parser.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"\nRunning at {url}\n")
    if args.open:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    uvicorn.run(
        "readtome.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
