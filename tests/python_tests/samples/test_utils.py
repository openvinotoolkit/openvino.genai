# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import os
import subprocess # nosec B404

def run_sample(command, input_data=None, env=os.environ):
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
