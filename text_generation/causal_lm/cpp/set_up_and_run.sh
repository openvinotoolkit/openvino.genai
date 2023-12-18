#!/bin/bash
# Copyright (C) 2023 Intel Corporation
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
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/linux/l_openvino_toolkit_ubuntu20_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
sudo ./ov/install_dependencies/install_openvino_dependencies.sh

source ./ov/setupvars.sh
python -m pip install --upgrade-strategy eager "optimum[openvino]>=1.14" -r ../../../llm_bench/python/requirements.txt ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu && python ../../../llm_bench/python/convert.py --model_id openlm-research/open_llama_3b_v2 --output_dir ./open_llama_3b_v2/ --precision FP16 --stateful &
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j
wait

python ./convert_tokenizers.py ./open_llama_3b_v2/pytorch/dldt/FP16/ --streaming-detokenizer
./build/causal_lm ./open_llama_3b_v2/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "return 0"
