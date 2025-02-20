# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from conftest import logger
import subprocess # nosec B404

def run_sample(command):
    logger.info(f"Running sample command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    logger.info(f"Sample output: {result.stdout}")
    return result