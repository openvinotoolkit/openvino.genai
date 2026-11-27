# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
from unittest.mock import patch

import pytest

import conftest
from conftest import run_wwb

RATE_LIMIT_OUTPUT = (
    "huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-6aea2325)\n"
    "429 Too Many Requests: you have reached your 'api' rate limit.\n"
    "Retry after 45 seconds (0/3000 requests remaining in current 300s window).\n"
    "OSError: Unable to load vocabulary from file.\n"
)
DNS_OUTPUT = "httpx.ConnectError: [Errno -3] Temporary failure in name resolution\n"
REAL_FAILURE_OUTPUT = "AssertionError: similarity 0.42 is below the threshold 0.9\n"


def _failure(output: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(1, ["wwb"], output=output)


@pytest.fixture(autouse=True)
def reset_session_retries(monkeypatch):
    monkeypatch.setattr(conftest, "_wwb_session_retries", 0)


@pytest.mark.parametrize(
    ("output", "expected_delay"),
    [(RATE_LIMIT_OUTPUT, 50), (DNS_OUTPUT, conftest.WWB_DEFAULT_RETRY_DELAY_SECONDS)],
    ids=["rate_limit_waits_retry_after", "network_error_waits_default"],
)
def test_run_wwb_retries_transient_error(output, expected_delay):
    with (
        patch("conftest.subprocess.check_output", side_effect=[_failure(output), "ok"]) as check_output,
        patch("conftest.time.sleep") as sleep,
    ):
        assert run_wwb(["--help"]) == "ok"
    assert check_output.call_count == 2
    sleep.assert_called_once_with(expected_delay)


def test_run_wwb_does_not_retry_real_failure():
    with (
        patch("conftest.subprocess.check_output", side_effect=_failure(REAL_FAILURE_OUTPUT)) as check_output,
        patch("conftest.time.sleep") as sleep,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_wwb(["--help"])
    assert check_output.call_count == 1
    sleep.assert_not_called()


def test_run_wwb_stops_after_max_attempts():
    with (
        patch("conftest.subprocess.check_output", side_effect=_failure(RATE_LIMIT_OUTPUT)) as check_output,
        patch("conftest.time.sleep"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_wwb(["--help"])
    assert check_output.call_count == conftest.WWB_MAX_ATTEMPTS


def test_run_wwb_stops_when_session_retries_are_used_up(monkeypatch):
    monkeypatch.setattr(conftest, "_wwb_session_retries", conftest.WWB_MAX_SESSION_RETRIES)
    with (
        patch("conftest.subprocess.check_output", side_effect=_failure(RATE_LIMIT_OUTPUT)) as check_output,
        patch("conftest.time.sleep") as sleep,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_wwb(["--help"])
    assert check_output.call_count == 1
    sleep.assert_not_called()
