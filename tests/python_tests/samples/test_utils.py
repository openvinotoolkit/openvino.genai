# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import os
import subprocess # nosec B404

PA_FALLBACK_WARNING = (
    "[WARNING] Paged Attention backend initialization failed. Falling back to SDPA backend. "
    'Set ATTENTION_BACKEND="SDPA" to skip Paged Attention initialization.'
)


def normalize_sample_output(output: str) -> str:
    if PA_FALLBACK_WARNING not in output:
        return output
    return output.replace(PA_FALLBACK_WARNING, "").strip()

def run_sample(
    command: list[str],
    input_data: str | None = None,
    env: dict[str, str] = os.environ,
    cwd: str | None = None,
):
    logger.info(f"Running sample command: {' '.join(map(str, command))}")
    if input_data:
        logger.info(f"Input data: {input_data}")
    try:
        result = subprocess.run(
            command,
            text=True,
            check=True,
            encoding="utf-8",
            env=env,
            input=input_data,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as error:
        logger.error(f"Sample returned {error.returncode}. Output:\n{error.output}")
        raise
    result.stdout = normalize_sample_output(result.stdout)
    logger.info(f"Sample output: {result.stdout}")
    return result
