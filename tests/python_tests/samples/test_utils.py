# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from conftest import logger
import subprocess # nosec B404

def run_sample(command, input_data=None):
    logger.info(f"Running sample command: {' '.join(command)}")
    if input_data:
        logger.info(f"Input data: {input_data}")
    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', input=input_data)
    logger.info(f"Sample output: {result.stdout}")
    return result
