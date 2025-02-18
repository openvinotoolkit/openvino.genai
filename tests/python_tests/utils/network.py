# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests
import time

def retry_request(func, retries=3):
    """
    Retries a function that makes a request up to a specified number of times.

    Parameters:
    func (callable): The function to be retried. It should be a callable that makes a request.
    retries (int): The number of retry attempts. Default is 3.

    Returns:
    Any: The return value of the function `func` if it succeeds.
    """
    for attempt in range(retries):
        try:
            return func()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e