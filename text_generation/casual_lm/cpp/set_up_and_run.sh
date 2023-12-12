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
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/linux/l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
sudo ./ov/install_dependencies/install_openvino_dependencies.sh

source ./ov/setupvars.sh
python -m pip install --upgrade-strategy eager transformers==4.35.2 "optimum[openvino]>=1.14" ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers] --extra-index-url https://download.pytorch.org/whl/cpu && optimum-cli export openvino -m openlm-research/open_llama_3b_v2 ./open_llama_3b_v2/ &
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j
wait

python ./convert_tokenizers.py ./open_llama_3b_v2/
./build/casual_lm ./open_llama_3b_v2/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "return 0"
