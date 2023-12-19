#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

convert_model="python ./llm_bench/python/convert.py --model_id bloomz-560m/ --output_dir ./ov_models/bloomz-560m --precision FP16"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ./ov_models/bloomz-560m/pytorch/dldt/FP16/ -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}