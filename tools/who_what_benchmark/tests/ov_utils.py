# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import uuid
import shutil
import logging

from pathlib import Path
from typing import Callable
from datetime import datetime
from importlib import metadata
from subprocess import CalledProcessError  # nosec B404
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import RequestException
from optimum.intel.openvino.utils import TemporaryDirectory

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_request(func, retries=7):
    """
    Retries a function that makes a request up to a specified number of times.

    Parameters:
    func (callable): The function to be retried. It should be a callable that makes a request.
    retries (int): The number of retry attempts. Default is 7.

    Returns:
    Any: The return value of the function `func` if it succeeds.
    """
    network_error_patterns = [
        "ConnectionError",
        "Timeout",
        "Time-out",
        "ServiceUnavailable",
        "InternalServerError",
        "OSError",
        "HTTPError",
    ]

    for attempt in range(retries):
        try:
            return func()
        except (CalledProcessError, RequestException, HfHubHTTPError) as e:
            if isinstance(e, CalledProcessError):
                error_output = (e.stdout or "") + (e.stderr or "")
                if error_output and any(pattern in error_output for pattern in network_error_patterns):
                    logger.warning(f"CalledProcessError occurred: {error_output}")
                else:
                    raise
            if attempt < retries - 1:
                timeout = 2**attempt
                logger.info(f"Attempt {attempt + 1} failed. Retrying in {timeout} seconds.")
                time.sleep(timeout)
            else:
                raise


def get_ov_cache_dir(temp_dir=TemporaryDirectory()):
    if "OV_CACHE" in os.environ:
        date_subfolder = datetime.now().strftime("%Y%m%d")
        ov_cache = os.path.join(os.environ["OV_CACHE"], date_subfolder)
        try:
            optimum_intel_version = metadata.version("optimum-intel")
            transformers_version = metadata.version("transformers")
            ov_cache = os.path.join(
                ov_cache, f"optimum-intel-{optimum_intel_version}_transformers-{transformers_version}"
            )
        except metadata.PackageNotFoundError:
            pass
        ov_cache_path = Path(ov_cache)
        ov_cache_path.mkdir(parents=True, exist_ok=True)
        return ov_cache_path
    else:
        ov_cache = temp_dir.name
        return Path(ov_cache)


class AtomicDownloadManager:
    def __init__(self, final_path: Path):
        self.final_path = Path(final_path)
        random_suffix = uuid.uuid4().hex[:8]
        self.temp_path = self.final_path.parent / f".tmp_{self.final_path.name}_{random_suffix}"

    def is_complete(self) -> bool:
        return self.final_path.exists()

    def execute(self, download_fn: Callable[[Path], None]) -> None:
        if self.is_complete():
            logger.info(f"Already downloaded: {self.final_path}")
            return

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        try:
            download_fn(self.temp_path)
            self._move_to_final_location()
        except Exception:
            logger.exception("Error during operation")
            self._cleanup_temp()
            raise

    def _move_to_final_location(self) -> None:
        if self.final_path.exists():
            logger.info(f"Destination already exists (created by another process): {self.final_path}")
            self._cleanup_temp()
            return

        logger.info(f"Moving temp to final location: {self.temp_path} -> {self.final_path}")
        try:
            self.temp_path.rename(self.final_path)
        except Exception:
            logger.warning("Rename failed, falling back to shutil.move")
            if self.final_path.exists():
                logger.info(f"Destination created by another process during rename attempt: {self.final_path}")
                self._cleanup_temp()
                return
            try:
                shutil.move(str(self.temp_path), str(self.final_path))
            except Exception:
                logger.exception("Error during move - assuming it was created successfully by another process")
                self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        if self.temp_path.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_path}")
            try:
                shutil.rmtree(self.temp_path)
            except Exception:
                logger.exception("Could not clean up temp directory")
