# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Profiling utilities for WWB tests."""

import time
from contextlib import contextmanager
from datetime import datetime, timezone


def _ts() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    """Log message with timestamp."""
    print(f"[{_ts()}] [wwb] {message}", flush=True)


@contextmanager
def _stage(name: str):
    """Context manager to time and log a stage of execution."""
    _log(f"START {name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        _log(f"END   {name} dt={dt:.3f}s")
