# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import os
import subprocess # nosec B404

def run_sample(
    command: list[str],
    input_data: str | None = None,
    env: dict[str, str] = os.environ,
):
    logger.info(f"Running sample command: {' '.join(map(str, command))}")
    if input_data:
        logger.info(f"Input data: {input_data}")
    try:
        result = subprocess.run(command, text=True, check=True, encoding='utf-8', env=env, input=input_data, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        logger.error(f"Sample returned {error.returncode}. Output:\n{error.output}")
        raise
    logger.info(f"Sample output: {result.stdout}")
    return result


def run_js_chat(
    command: list[str],
    input_data: str,
    env: dict[str, str] = os.environ,
):
    print(f"Running JS sample command: {' '.join(map(str, command))}")
    inputs = input_data.split("\n")
    print(f"Input data: {input_data}")
    try:
        proc = subprocess.Popen(
            command,
            text=True,
            encoding="utf-8",
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        output_lines: list[str] = []
        i = 0
        for line in proc.stdout:
            output_lines.append(line)
            if "question:" in line:
                if i < len(inputs):
                    proc.stdin.write(inputs[i])
                    proc.stdin.write("\n")
                    proc.stdin.flush()
                    i += 1
                else:
                    break

        proc.stdin.close()
    except subprocess.CalledProcessError as error:
        print(f"Sample returned {error.returncode}. Output:\n{error.output}")
        raise
    stdout = "".join(output_lines)
    print(f"Sample output: {stdout}")
    return stdout
