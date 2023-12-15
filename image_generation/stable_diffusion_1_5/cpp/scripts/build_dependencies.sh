#!/usr/bin/env bash

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# exit when any command fails
set -e

usage() {
    echo "SD C++ dependencies "

    exit 1
}

echo
echo "---prepare the dependencies: CMake, Eigen3 and OpenVINO---"
# Check if CMake is installed
if ! command -v cmake &> /dev/null
then
    echo "CMake is not installed. Installing CMake..."
    sudo apt update
    sudo apt install cmake -y
else
    echo "CMake is already installed."
fi

# Check if Eigen3 libraries are installed using dpkg
if dpkg -l | grep libeigen3-dev &> /dev/null; then
    echo "Eigen3 are already installed."
else
    echo "Eigen3 C++ libraries are not installed. Installing Eigen3 via apt..."
    sudo apt update
    sudo apt install libeigen3-dev -y
fi

# Prompt the user if they want to install OpenVINO
read -p "use conda-forge to install OpenVINO Toolkit 2023.1.0 (C++), or download from archives? (yes/no): " choice

if [ "$choice" = "yes" ]; then
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed. Please install Conda before proceeding."
        exit 1
    else
        echo "Conda is already installed."
    fi

    # Find the directory containing conda executable
    conda_bin_dir=$(dirname $(which conda))
    # Define the name of the conda environment
    environment_name="SD-CPP"


    # Check if the 'SD-CPP' conda environment exists
    if ! conda env list | grep -q "SD-CPP"; then
        echo "The 'SD-CPP' conda environment does not exist."
        echo "Creating 'SD-CPP' environment "
        conda create -n SD-CPP python==3.10 -y
        # Activate the conda environment using source
        source "${conda_bin_dir}/../etc/profile.d/conda.sh"
        conda activate "${environment_name}"
        conda install -c conda-forge openvino=2023.1.0 -y
        echo "Environment 'SD-CPP' created and openvino 2023.1.0 installed."
    else
        echo "The 'SD-CPP' conda environment already exists."
    fi

    
else
    echo "### downnload OpenVINO 2023.1.0 from archives, unzip and setup vars."
    
    # Check if '../download' folder exists
    if [ ! -d "../download" ]; then
        echo "Creating '../download' folder..."
        mkdir ../download
    fi

    # Check if the file already exists in the 'download' folder
    if [ -f "../download/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64.tgz" ]; then
        echo "OpenVINO Toolkit package already exists in '../download' folder."
    else
        # Download OpenVINO Toolkit
        echo "### If download too slow, stop and download manually and rerun this script to unzip"
        echo
        echo

        echo "cd ../download && wget https://storage.openvinotoolkit.org/repositories/openvino/packages/master/2023.1.0.dev20230811/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64.tgz"
        echo
        wget "https://storage.openvinotoolkit.org/repositories/openvino/packages/master/2023.1.0.dev20230811/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64.tgz" -P "../download/"
        if [ $? -eq 0 ]; then
            echo "OpenVINO Toolkit downloaded successfully."
        else
            echo "Failed to download OpenVINO Toolkit, please download manually and rerun this script to unzip."
            echo "cd ../download && wget https://storage.openvinotoolkit.org/repositories/openvino/packages/master/2023.1.0.dev20230811/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64.tgz"
            exit 1
        fi
    fi

    # Extract the downloaded tgz file
    echo "Extracting OpenVINO Toolkit..."
    echo "### If get this: 'gzip: stdin: unexpected end of file', delete the incomplete tgz file and rerun this script"
    tar zxf "../download/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64.tgz" -C "../download/"

    # Check if setupvars.sh exists in the extracted folder
    if [ -f "../download/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64/setupvars.sh" ]; then
        echo "OpenVINO Toolkit setupvars.sh found."
    else
        echo "setupvars.sh not found in the extracted folder. Please source it manually to activate OpenVINO environment."
    fi
fi

echo "### Finished all the preparation"  
echo "### Please activate the conda env manually with command 'conda activate SD-CPP', or source ../download/l_openvino_toolkit_ubuntu22_2023.1.0.dev20230811_x86_64/setupvars.sh"
echo "### Then build the pipeline with CMake following the README's guide"