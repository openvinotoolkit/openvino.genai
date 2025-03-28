# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import datetime
import logging
from huggingface_hub.utils import HfHubHTTPError
from subprocess import CalledProcessError # nosec B404
from requests.exceptions import RequestException

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_request(func, retries=7):
    """
    Retries a function that makes a request up to a specified number of times.

    Parameters:
    func (callable): The function to be retried. It should be a callable that makes a request.
    retries (int): The number of retry attempts. Default is 5.

    Returns:
    Any: The return value of the function `func` if it succeeds.
    """
    network_error_patterns = [
        "ConnectionError",
        "Timeout",
        "Time-out",
        "ServiceUnavailable",
        "InternalServerError"
    ]
    
    for attempt in range(retries):
        try:
            return func()
        except (CalledProcessError, RequestException, HfHubHTTPError) as e:
            print(f"[{datetime.datetime.now()}] [retry_request] exception occured: {e}")
            if isinstance(e, CalledProcessError):
                print(f"[{datetime.datetime.now()}] [retry_request] CalledProcessError exception occured: {e.stderr}")
                # if any(pattern in e.stderr for pattern in network_error_patterns):
                #     logger.warning(f"CalledProcessError occurred: {e.stderr}")
                #     print(f"[{datetime.datetime.now()}] [retry_request] exception catched: {e}")
                # else:
                #     print(f"[{datetime.datetime.now()}] [retry_request] exception re-raised: {e}")
                #     raise e
            if attempt < retries - 1:
                timeout = 2 ** attempt
                print(f"[{datetime.datetime.now()}] [retry_request] timeout: {timeout}")
                logger.info(f"Attempt {attempt + 1} failed. Retrying in {timeout} seconds.")
                time.sleep(timeout)
            else:
                raise e
