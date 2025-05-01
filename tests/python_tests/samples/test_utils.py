# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import os
import subprocess # nosec B404

def run_sample(command, input_data=None):
    logger.info(f"Running sample command: {' '.join(command)}")
    if input_data:
        logger.info(f"Input data: {input_data}")
    try:
        result = subprocess.run(command, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, input=input_data)
    except  subprocess.CalledProcessError as error:
        print(f"'{' '.join(map(str, command))}' returned {error.returncode}. Output:")
        print(error.output)
        raise
    logger.info(f"Sample output: {result.stdout}")
    return result
