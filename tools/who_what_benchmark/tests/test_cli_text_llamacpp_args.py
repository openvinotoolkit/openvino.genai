# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404

import pytest

from conftest import run_wwb


def _assert_wwb_cli_error(args: list[str], expected_message: str) -> str:
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_wwb(args)

    output = error.value.output
    assert expected_message in output
    return output


def test_text_llamacpp_chat_requires_llamacpp(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(gt_data),
            "--model-type",
            "text",
            "--llamacpp-chat",
        ],
        "--llamacpp-chat requires --llamacpp",
    )


def test_text_llamacpp_n_ctx_requires_llamacpp(tmp_path):
    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(tmp_path / "gt.csv"),
            "--model-type",
            "text",
            "--llamacpp-n-ctx",
            "4096",
        ],
        "--llamacpp-n-ctx requires --llamacpp",
    )


def test_text_non_llamacpp_run_not_blocked_by_n_ctx_default(tmp_path):
    output = _assert_wwb_cli_error(
        [
            "--gt-data",
            str(tmp_path / "missing.csv"),
            "--model-type",
            "text",
        ],
        "Ground-truth data file",
    )

    assert "--llamacpp-n-ctx requires --llamacpp" not in output
