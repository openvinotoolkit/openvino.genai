#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

git lfs install
export GIT_LFS_SKIP_SMUDGE=0
git clone https://huggingface.co/bigscience/bloomz-560m
python ./llm_bench/python/convert.py --model_id bloomz-560m/pytorch  --output_dir ./ --precision FP16
python ./llm_bench/python/benchmark.py -m ./bloomz-560m/pytorch/dldt/FP16/ 