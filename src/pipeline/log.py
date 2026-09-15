"""Logging estruturado em stdout, com tempo por estagio."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


def _fmt(value: object) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text


def log(event: str, **fields: object) -> None:
    ts = time.strftime("%H:%M:%S")
    parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    print(f"{ts} {event} {parts}".rstrip(), file=sys.stdout, flush=True)


@contextmanager
def stage(name: str, **fields: object):
    log(f"{name}.start", **fields)
    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        log(f"{name}.failed", elapsed=f"{time.monotonic() - started:.1f}s", error=repr(exc))
        raise
    log(f"{name}.done", elapsed=f"{time.monotonic() - started:.1f}s")
