# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import os
import subprocess # nosec B404
import time

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


def run_js_chat(
    command: list[str],
    input_data: str,
    env: dict[str, str] | None = None,
    timeout: int = 600,
):
    logger.info(f"Running JS sample command: {' '.join(map(str, command))}")
    inputs = [s for s in input_data.splitlines() if s]
    logger.info(f"Input data: {input_data}")
    proc = subprocess.Popen(
        command,
        text=True,
        encoding="utf-8",
        env=env or os.environ,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stdin is not None

    stdout_chunks: list[str] = []
    input_index = 0
    start_time = time.monotonic()
    try:
        for line in proc.stdout:
            stdout_chunks.append(line)
            if "question:" in line:
                if input_index < len(inputs):
                    proc.stdin.write(inputs[input_index])
                    proc.stdin.write("\n")
                    proc.stdin.flush()
                    input_index += 1
                else:
                    break

        proc.stdin.close()

        remaining_timeout = timeout - int(time.monotonic() - start_time)
        if remaining_timeout <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        remaining_output, _ = proc.communicate(timeout=remaining_timeout)
        if remaining_output:
            stdout_chunks.append(normalize_sample_output(remaining_output))
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        stdout = "".join(stdout_chunks)
        logger.error(f"JS sample timed out. Partial output:\n{stdout}")
        raise
    except Exception:
        proc.kill()
        proc.wait()
        raise

    stdout = "".join(stdout_chunks)
    if return_code != 0:
        logger.error(f"JS sample returned {return_code}. Output:\n{stdout}")
        raise subprocess.CalledProcessError(return_code, command, output=stdout)

    logger.info(f"Sample output: {stdout}")
    return stdout
