#!/bin/bash
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

function abs_path() {
    script_path=$(eval echo "${BASH_SOURCE[0]}")
    directory=$(dirname "$script_path")
    builtin cd "$directory" || exit
    pwd -P
}
cd "`abs_path`"

mkdir ./ov/
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2024.0.0-14004-a240ae8fadd/l_openvino_toolkit_ubuntu20_2024.0.0.dev20240116_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
sudo ./ov/install_dependencies/install_openvino_dependencies.sh

source ./ov/setupvars.sh
python -m pip install --upgrade-strategy eager "optimum>=1.14" -r ../../../llm_bench/python/requirements.txt ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu && python ../../../llm_bench/python/convert.py --model_id openlm-research/open_llama_3b_v2 --output_dir ./open_llama_3b_v2/ --precision FP16 --stateful &
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j
wait

convert_tokenizer ./open_llama_3b_v2/pytorch/dldt/FP16/ --output ./open_llama_3b_v2/pytorch/dldt/FP16/ --with-detokenizer
./build/greedy_causal_lm ./open_llama_3b_v2/pytorch/dldt/FP16/ "return 0"
