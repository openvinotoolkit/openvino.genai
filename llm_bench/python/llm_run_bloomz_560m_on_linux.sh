#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cat /proc/cpuinfo | grep 'model name' | uniq
grep MemTotal /proc/meminfo

git lfs install
export GIT_LFS_SKIP_SMUDGE=0
echo "Download bloomz-560m"
git_clone_bloomz_560m="git clone https://huggingface.co/bigscience/bloomz-560m"
echo ${git_clone_bloomz_560m}
eval ${git_clone_bloomz_560m}
wait

original_dir="bloomz-560m"
dst_dir="./ov_models/bloomz-560m"

convert_model="python ./llm_bench/python/convert.py --model_id ${original_dir} --output_dir ${dst_dir} --precision FP16 --stateful"
echo ${convert_model}
eval ${convert_model}
ret=$?
wait

if [ ${ret} -ne 0]; then
    echo "convert model ret=${ret}"
    exit ${ret}
end

benchmarking="python ./llm_bench/python/benchmark.py -m ${dst_dir}/pytorch/dldt/FP16/ -d cpu -n 1"
echo ${benchmarking}
eval ${benchmarking}
ret=$?

rm -rf ${original_dir}
rm -rf ${dst_dir}

if [ ${ret} -ne 0]; then
    echo "benchmarking ret=${ret}"
    exit ${ret}
fi