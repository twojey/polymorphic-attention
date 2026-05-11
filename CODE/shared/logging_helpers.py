"""
logging_helpers.py — logging horodaté + exception capture pour drivers Python.

Refactor 2026-05-11 suite Run 3 silent crash. Cf. carnet + memory feedback
"Robustesse scripts obligatoire".

Usage typique dans un driver phase :

    from shared.logging_helpers import setup_logging, log_exceptions

    logger = setup_logging(phase="1.5", prefix="run3_skl")

    @log_exceptions(logger)
    @hydra.main(...)
    def main(cfg):
        logger.info("démarrage avec cfg=%s", cfg)
        ...

Garanties :
- Format horodaté ISO-8601 UTC.
- Handler console (stderr) ET fichier (OPS/logs/<prefix>_<TS>.log) si demandé.
- sys.excepthook installé : toute exception non-attrapée → stack trace dans logger.
- Le log file de bash (ASP_LOG_FILE) peut être réutilisé pour cohérence.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_DEFAULT_FORMAT = "%(asctime)sZ [%(levelname)s] %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class _UTCFormatter(logging.Formatter):
    """Formatter qui rapporte les timestamps en UTC (pas locale)."""

    converter = staticmethod(  # type: ignore[assignment]
        lambda *args: datetime.now(timezone.utc).timetuple()
    )


def _resolve_log_dir() -> Path:
    """OPS/logs/ relatif au repo root (détecté via cwd ou env)."""
    # On accepte deux signaux : ASP_REPO_ROOT (set par common.sh) ou cwd
    repo = os.environ.get("ASP_REPO_ROOT")
    if repo and Path(repo).is_dir():
        return Path(repo) / "OPS" / "logs"
    # Fallback : monte depuis cwd jusqu'à trouver OPS/
    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "OPS").is_dir() and (parent / "CODE").is_dir():
            return parent / "OPS" / "logs"
    # Fallback ultime
    return Path("OPS/logs")


def setup_logging(
    *,
    phase: str,
    prefix: str | None = None,
    level: int = logging.INFO,
    to_file: bool = True,
    reuse_bash_log: bool = True,
) -> logging.Logger:
    """Configure le root logger + retourne un logger nommé "phase{phase}".

    Args:
        phase: identifiant phase ("1", "1.5", "2", ...). Sert au nom du logger.
        prefix: préfixe du fichier log. Défaut : "phase{phase}".
        level: niveau (DEBUG, INFO, WARNING, ERROR).
        to_file: si True, ajoute un FileHandler vers OPS/logs/<prefix>_<TS>.log.
        reuse_bash_log: si True et $ASP_LOG_FILE est set par common.sh, append à
            ce fichier au lieu d'en créer un nouveau. Garantit log cohérent
            shell+Python dans un seul fichier.

    Returns:
        Logger nommé "phase{phase}".
    """
    prefix = prefix or f"phase{phase}"
    root = logging.getLogger()
    root.setLevel(level)

    # Reset handlers existants (évite duplication en cas de double-init)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = _UTCFormatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    # Console handler (stderr — n'interfère pas avec stdout métier)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    root.addHandler(ch)

    if to_file:
        log_path = _bash_log_path() if reuse_bash_log else None
        if log_path is None:
            log_dir = _resolve_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = log_dir / f"{prefix}_{ts}.log"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        root.addHandler(fh)
        # Log path visible dans tous les logs ultérieurs
        root.info("logging vers %s (level=%s)", log_path, logging.getLevelName(level))

    # Hook global : toute exception non-attrapée → stack trace via logger
    _install_excepthook(root)

    logger = logging.getLogger(f"phase{phase}")
    return logger


def _bash_log_path() -> Path | None:
    """Retourne le path du log bash courant ($ASP_LOG_FILE) ou None."""
    path = os.environ.get("ASP_LOG_FILE")
    if not path:
        return None
    p = Path(path)
    if not p.parent.is_dir():
        return None
    return p


def _install_excepthook(logger: logging.Logger) -> None:
    """Capture toute exception non-attrapée dans le logger."""

    def hook(exc_type, exc_value, exc_tb):
        # KeyboardInterrupt : laisser le comportement par défaut (silencieux)
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Uncaught exception : %s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = hook


def log_exceptions(logger: logging.Logger) -> Callable:
    """Decorator : log toute exception levée dans la fonction wrappée.

    Usage :
        logger = setup_logging(phase="1.5")

        @log_exceptions(logger)
        def main(...):
            ...
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Exception dans %s", fn.__qualname__)
                raise

        return wrapper

    return deco


def log_checkpoint(logger: logging.Logger, label: str, **kv: Any) -> None:
    """Log un checkpoint d'avancement avec key-value pairs.

    Utile pour suivre la progression : si crash, le dernier checkpoint vu
    permet de localiser le point de défaillance.

    Example:
        log_checkpoint(logger, "calib_batch_done", batch=3, seq_len=4371, mem_mb=12000)
    """
    if not kv:
        logger.info("[ckpt] %s", label)
    else:
        kv_str = " ".join(f"{k}={v}" for k, v in kv.items())
        logger.info("[ckpt] %s %s", label, kv_str)
