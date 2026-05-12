"""Tests retry décorateur."""

from __future__ import annotations

import logging
import time

import pytest

from shared.retry import retry, retry_call


def test_retry_success_first_attempt() -> None:
    calls = []
    @retry(max_attempts=3, base_delay=0.001)
    def fn():
        calls.append(1)
        return "ok"
    assert fn() == "ok"
    assert len(calls) == 1


def test_retry_succeeds_after_2_failures() -> None:
    state = {"attempts": 0}
    @retry(max_attempts=5, base_delay=0.001, jitter=0)
    def fn():
        state["attempts"] += 1
        if state["attempts"] < 3:
            raise ConnectionError("boom")
        return "ok"
    assert fn() == "ok"
    assert state["attempts"] == 3


def test_retry_exhausted_raises_last_exception() -> None:
    @retry(max_attempts=3, base_delay=0.001, jitter=0)
    def fn():
        raise ValueError("nope")
    with pytest.raises(ValueError, match="nope"):
        fn()


def test_retry_catch_specific_exception() -> None:
    """Une exception hors `catch` est levée immédiatement (pas de retry)."""
    state = {"attempts": 0}
    @retry(max_attempts=5, base_delay=0.001, catch=(ConnectionError,))
    def fn():
        state["attempts"] += 1
        raise ValueError("autre")
    with pytest.raises(ValueError):
        fn()
    assert state["attempts"] == 1


def test_retry_keyboard_interrupt_not_caught() -> None:
    """KeyboardInterrupt n'est JAMAIS retried même avec catch=(BaseException,)."""
    @retry(max_attempts=5, base_delay=0.001, catch=(BaseException,))
    def fn():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        fn()


def test_retry_max_attempts_validation() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        @retry(max_attempts=0)
        def fn():
            pass


def test_retry_on_callback_invoked() -> None:
    seen: list[tuple[int, str, float]] = []
    def cb(attempt, exc, delay):
        seen.append((attempt, type(exc).__name__, delay))
    @retry(max_attempts=3, base_delay=0.001, jitter=0, on_retry=cb)
    def fn():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        fn()
    # 2 retries avant échec final (attempts 1, 2 raise → retry, attempt 3 raise → exhausted)
    assert len(seen) == 2
    assert seen[0][0] == 1
    assert seen[1][0] == 2


def test_retry_exponential_backoff_timing() -> None:
    """base_delay=0.05 backoff=2 jitter=0 → delays ~ [0.05, 0.10]."""
    @retry(max_attempts=3, base_delay=0.05, jitter=0, backoff_factor=2.0)
    def fn():
        raise RuntimeError("x")
    t0 = time.time()
    with pytest.raises(RuntimeError):
        fn()
    dt = time.time() - t0
    # 2 sleeps : 0.05 + 0.10 = 0.15s mini, on accepte ±50ms
    assert 0.10 < dt < 0.30


def test_retry_call_form() -> None:
    """retry_call : variante sans décorateur."""
    state = {"n": 0}
    def fn(x):
        state["n"] += 1
        if state["n"] < 2:
            raise IOError("boom")
        return x * 2
    out = retry_call(fn, args=(5,), max_attempts=3, base_delay=0.001, jitter=0)
    assert out == 10


def test_retry_logger_used(caplog) -> None:
    logger = logging.getLogger("test_retry_logger")
    @retry(max_attempts=3, base_delay=0.001, jitter=0, logger=logger)
    def fn():
        raise ConnectionError("x")
    with caplog.at_level(logging.WARNING, logger="test_retry_logger"):
        with pytest.raises(ConnectionError):
            fn()
    # 2 warnings (attempt 1 + 2) + 1 error (exhausted)
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(warning_records) >= 2
    assert len(error_records) >= 1
