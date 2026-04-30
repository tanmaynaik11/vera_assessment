import logging
import sys
from datetime import datetime
from pathlib import Path

_configured = False


def setup(log_dir: str = "logs") -> None:
    """Call once at startup. Safe to call multiple times — configures only once."""
    global _configured
    if _configured:
        return
    _configured = True

    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"vera_{timestamp}.log"

    fmt_console = "%(asctime)s.%(msecs)03d [%(threadName)-15s] %(levelname)-5s %(message)s"
    fmt_file    = "%(asctime)s.%(msecs)03d [%(threadName)-15s] [%(name)-22s] %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console — INFO and above, clean format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt_console, datefmt=datefmt))
    root.addHandler(ch)

    # File — DEBUG and above, full detail
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt_file, datefmt=datefmt))
    root.addHandler(fh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.info(f"Logging started → {log_file}")
