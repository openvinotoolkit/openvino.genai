# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

LLM_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_BENCH_ROOT))
