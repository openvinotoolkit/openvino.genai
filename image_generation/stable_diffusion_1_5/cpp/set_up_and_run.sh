#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

abs_path() {
    script_path=$(eval echo "${BASH_SOURCE[0]}")
    directory=$(dirname "$script_path")
    builtin cd "$directory" || exit
    pwd -P
}
cd "`abs_path`"

# initialize OpenVINO
mkdir ./openvino
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/linux/l_openvino_toolkit_ubuntu20_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz | tar --directory ./openvino/ --strip-components 1 -xz
sudo -E ./openvino/install_dependencies/install_openvino_dependencies.sh
source ./openvino/setupvars.sh

# download extra dependencies
sudo -E apt install libeigen3-dev -y

# download / convert model and tokenizer
cd scripts
python -m pip install -U pip
python -m pip install -r ./requirements.txt
# python -m pip install ../../../thirdparty/openvino_contrib/modules/custom_operations/[transformers]
python convert_model.py -sd runwayml/stable-diffusion-v1-5 -b 1 -t FP16 -dyn True
cd ..

# build app
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release --parallel

# run app
cd build
./SD-generate -m scripts/runwayml/stable-diffusion-v1-5 -t FP16_dyn
