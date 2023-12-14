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

mkdir ./ov/
curl https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/linux/l_openvino_toolkit_ubuntu20_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz | tar --directory ./ov/ --strip-components 1 -xz
sudo ./ov/install_dependencies/install_openvino_dependencies.sh
source ./ov/setupvars.sh
python -m pip install -r ./scripts/requirements.txt && python -m convert_model.py -sd runwayml/stable-diffusion-v1-5 -b 1 -t FP16 -dyn True &
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j
wait

./build/SD-generate -m runwayml/stable-diffusion-v1-5 -t FP16_dyn