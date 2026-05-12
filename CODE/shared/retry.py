"""
retry.py — décorateur @retry avec backoff exponentiel + jitter.

Spec : robustesse scripts (DOC feedback "y en a marre des scripts qui plantent
sans retry").

Usage typique :

    from shared.retry import retry

    @retry(max_attempts=3, base_delay=1.0, jitter=0.5,
           catch=(ConnectionError, OSError))
    def download_model(url: str) -> bytes:
        ...

Garanties :
- Loggue chaque retry avec logger fourni (ou root logger)
- Backoff exponentiel : delay = base_delay * (2 ** attempt) + jitter
- jitter : random uniform [0, jitter * delay] pour éviter le thundering herd
- Si toutes les tentatives échouent, lève la dernière exception
- KeyboardInterrupt + SystemExit ne sont JAMAIS attrapés

Pattern alternatif sans décorateur :

    from shared.retry import retry_call

    result = retry_call(my_func, args=(x,), max_attempts=5)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Exceptions qu'on ne retry JAMAIS (signal de l'utilisateur ou bug fatal)
_NEVER_RETRY: tuple[type[BaseException], ...] = (
    KeyboardInterrupt, SystemExit, MemoryError,
)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.5,
    catch: tuple[type[BaseException], ...] | type[BaseException] = (Exception,),
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    logger: logging.Logger | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Décorateur @retry.

    Args
    ----
    max_attempts : tentatives max (≥ 1). max_attempts=1 = pas de retry.
    base_delay : délai initial (s) avant 1ère retry
    jitter : facteur jitter [0, 1] (uniform random)
    catch : exception(s) qui déclenchent retry. Tuple ou classe unique.
    backoff_factor : multiplicateur du délai à chaque attempt (défaut 2× exp)
    max_delay : plafond du délai (s)
    logger : logger pour les warnings retry. Si None, utilise root logger.
    on_retry : callback (attempt, exception, delay_next) appelé avant chaque sleep.

    Returns
    -------
    Décorateur qui wrappe la fonction cible.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts ≥ 1, reçu {max_attempts}")
    if isinstance(catch, type):
        catch = (catch,)
    log = logger or logging.getLogger(__name__)

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except _NEVER_RETRY:
                    raise
                except catch as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        log.error(
                            "retry exhausted (%d/%d) for %s : %s",
                            attempt, max_attempts, fn.__qualname__, exc,
                        )
                        raise
                    delay = min(
                        base_delay * (backoff_factor ** (attempt - 1)),
                        max_delay,
                    )
                    if jitter > 0:
                        delay += random.uniform(0.0, jitter * delay)
                    log.warning(
                        "retry %d/%d for %s after %s : %s — sleep %.2fs",
                        attempt, max_attempts, fn.__qualname__,
                        type(exc).__name__, exc, delay,
                    )
                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc, delay)
                        except Exception:  # noqa: BLE001
                            log.exception("on_retry callback échec")
                    time.sleep(delay)
            # Inaccessible (raise dans la boucle) mais mypy/linters appreciate
            if last_exc is not None:
                raise last_exc  # noqa: RSE102
            raise RuntimeError(f"retry: unreachable for {fn.__qualname__}")

        return wrapper

    return deco


def retry_call(
    fn: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    **retry_kwargs: Any,
) -> T:
    """Variante sans décorateur : appelle fn avec retry.

    Equivalent à @retry(**retry_kwargs)(fn)(*args, **kwargs).
    """
    kwargs = kwargs or {}
    decorated = retry(**retry_kwargs)(fn)
    return decorated(*args, **kwargs)
