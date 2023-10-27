#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status.

function cd_to_script_location() {
    script_path=$(eval echo "${BASH_SOURCE[0]}")
    directory=$(dirname "$script_path")
    builtin cd "$directory" || exit
}

cd_to_script_location

python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu llm/cpp/thirdparty/openvino_contrib/modules/custom_operations/user_ie_extensions/tokenizer/python/[transformers] onnx git+https://github.com/huggingface/optimum-intel.git &
mkdir ov/
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/linux/l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed_x86_64.tgz | tar --directory ov/ --strip-components 1 -xz
sudo ov/install_dependencies/install_openvino_dependencies.sh
wait

source ov/setupvars.sh
optimum-cli export openvino -m openlm-research/open_llama_3b_v2 open_llama_3b_v2/ &
mkdir build/
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Werror ../llm/cpp
cmake --build . --config Release -j
wait

source ov/setupvars.sh
python llm/cpp/convert_tokenizers.py build/thirdparty/openvino_contrib/modules/custom_operations/user_ie_extensions/libuser_ov_extensions.so open_llama_3b_v2/
./build/llm open_llama_3b_v2/openvino_model.xml tokenizer.xml detokenizer.xml "return 0"
