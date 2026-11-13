# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# llm_bench is run as a script from its own directory rather than installed as
# a package, so the tests put that directory on sys.path the same way
# benchmark.py does implicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
