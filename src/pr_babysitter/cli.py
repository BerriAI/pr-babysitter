from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .config import CONFIG_DIR
from .tui import PRBabysitterApp


class _SecureRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that creates each log file (both the active file
    and rollover successors) with mode 0o600 from the moment of creation.

    The stock handler honours the process umask, leaving a TOCTOU window
    between file creation and any subsequent `os.chmod` where another user
    on a shared system could read the PATs that the babysitter echoes into
    transcript tails (and from there into the log)."""

    def _open(self):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self.baseFilename, flags, 0o600)
        return os.fdopen(
            fd,
            self.mode,
            encoding=self.encoding,
            errors=getattr(self, "errors", None),
        )


def _configure_logging() -> None:
    # TUI owns stdout/stderr, so log to a file. Rotates to keep ~10MB of recent
    # tick history; enough to diagnose flapping verdicts across many ticks.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CONFIG_DIR / "babysitter.log"
    # Use a handler subclass that opens each log file with 0o600 from
    # creation. The log echoes transcript tails (which may contain the PATs
    # baked into the agent system prompt) and other sensitive operational
    # detail, so we can't rely on the default umask + post-creation chmod.
    handler = _SecureRotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=5,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("pr_babysitter").setLevel(logging.DEBUG)


def main() -> None:
    _configure_logging()
    app = PRBabysitterApp()
    try:
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
