# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov

import os
import shutil
import pytest
from importlib import metadata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer
from optimum.intel.openvino.modeling import OVModel

ModelDownloaderCallable = Callable[[str], tuple[OVModel | None, AutoTokenizer | None, Path]]


CACHE_BASE_DIR = "ov_models"
CACHE_CURRENT_DIR_KEY = "ov_cache/current_dir"
CACHE_TIMESTAMP_KEY = "ov_cache/timestamp"
CACHE_VERSIONS_KEY = "ov_cache/versions"
MODELS_SUBDIR = "test_models"
DEFAULT_CACHE_EXPIRY_HOURS = 24
OPTIMUM_INTEL_PACKAGE = "optimum-intel"
TRANSFORMERS_PACKAGE = "transformers"

OV_MODEL_FILENAME = "openvino_model.xml"
OV_TOKENIZER_FILENAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_FILENAME = "openvino_detokenizer.xml"


def dt_now() -> datetime:
    return datetime.now(timezone.utc)


def get_default_llm_properties():
    return {
        hints.inference_precision: ov.Type.f32,
        hints.kv_cache_precision: ov.Type.f16,
    }


def extra_generate_kwargs():
    from optimum.intel.utils.import_utils import is_transformers_version
    additional_args = {}
    if is_transformers_version(">=", "4.51"):
        additional_args["use_model_defaults"] = False
    return additional_args


def get_disabled_mmap_ov_config():
    return {props.enable_mmap: False}


class OvTestCacheManager:
    def __init__(self, pytestconfig: pytest.Config):
        self.cache = pytestconfig.cache
        self.cache_expiry_hours = int(os.environ.get("OV_CACHE_EXPIRY_HOURS", DEFAULT_CACHE_EXPIRY_HOURS))
        self.base_cache_dir = self.cache.mkdir(CACHE_BASE_DIR)

    def get_cache_dir(self) -> Path:
        cached_dir = self.cache.get(CACHE_CURRENT_DIR_KEY, None)
        cached_timestamp = self.cache.get(CACHE_TIMESTAMP_KEY, None)
        cached_versions = self.cache.get(CACHE_VERSIONS_KEY, {})

        current_versions = self._get_version_info()

        if self._is_cache_valid(cached_timestamp, cached_versions, current_versions, cached_dir):
            return Path(cached_dir)

        return self._create_new_cache(current_versions)

    def get_models_dir(self) -> Path:
        cache_dir = self.get_cache_dir()
        models_dir = cache_dir / MODELS_SUBDIR
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _is_cache_valid(
        self,
        cached_timestamp: str | None,
        cached_versions: dict[str, str],
        current_versions: dict[str, str],
        cached_dir: str | None,
    ) -> bool:
        if not cached_timestamp or not cached_dir or not Path(cached_dir).exists():
            return False

        cache_time = datetime.fromisoformat(cached_timestamp)
        if cache_time.tzinfo is None:
            cache_time = cache_time.replace(tzinfo=timezone.utc)
        expiry_time = cache_time + timedelta(hours=self.cache_expiry_hours)

        if dt_now() > expiry_time:
            return False

        return cached_versions == current_versions

    def _create_new_cache(self, versions: dict[str, str]) -> Path:
        self._cleanup_expired_caches()

        date_str = dt_now().strftime("%Y%m%d%H")
        version_str = (
            f"{OPTIMUM_INTEL_PACKAGE}-"
            f"{versions.get(OPTIMUM_INTEL_PACKAGE, 'unknown')}_"
            f"{TRANSFORMERS_PACKAGE}-"
            f"{versions.get(TRANSFORMERS_PACKAGE, 'unknown')}"
        )
        cache_name = f"{date_str}_{version_str}"

        cache_dir = self.base_cache_dir / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        timestamp = dt_now().isoformat()

        self.cache.set(CACHE_CURRENT_DIR_KEY, str(cache_dir))
        self.cache.set(CACHE_TIMESTAMP_KEY, timestamp)
        self.cache.set(CACHE_VERSIONS_KEY, versions)

        return cache_dir

    def _get_version_info(self) -> dict[str, str]:
        versions = {}
        try:
            versions[OPTIMUM_INTEL_PACKAGE] = metadata.version(OPTIMUM_INTEL_PACKAGE)
        except metadata.PackageNotFoundError:
            versions[OPTIMUM_INTEL_PACKAGE] = "unknown"

        try:
            versions[TRANSFORMERS_PACKAGE] = metadata.version(TRANSFORMERS_PACKAGE)
        except metadata.PackageNotFoundError:
            versions[TRANSFORMERS_PACKAGE] = "unknown"

        return versions

    def _cleanup_expired_caches(self) -> None:
        if not self.base_cache_dir.exists():
            return

        threshold_dt = dt_now() - timedelta(hours=DEFAULT_CACHE_EXPIRY_HOURS)

        for cache_dir in self.base_cache_dir.iterdir():
            if not cache_dir.is_dir():
                continue

            try:
                date_token = cache_dir.name.split("_")[0]
                cache_dt = datetime.strptime(date_token, "%Y%m%d%H").replace(tzinfo=timezone.utc)

                if cache_dt <= threshold_dt:
                    shutil.rmtree(cache_dir, ignore_errors=True)
            except (ValueError, IndexError):
                continue
