# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = [line.strip() for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith("#")]


is_installing_editable = "develop" in sys.argv
is_building_release = not is_installing_editable and "--release" in sys.argv


def set_version(base_version: str):
    version_value = base_version
    if not is_building_release:
        if is_installing_editable:
            return version_value + ".dev0+editable"
        import subprocess  # nosec

        dev_version_id = "unknown_version"
        try:
            repo_root = os.path.dirname(os.path.realpath(__file__))
            dev_version_id = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)  # nosec
                .strip()
                .decode()
            )
        except subprocess.CalledProcessError:
            pass
        return version_value + f".dev0+{dev_version_id}"

    return version_value


setup(
    name="toolcallbench",
    version=set_version("0.1.0"),
    url="https://github.com/openvinotoolkit/openvino.genai.git",
    author="Intel",
    description="Tool-call capability benchmark for coding-agent style LLMs",
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["tcb=toolcallbench.tcb:main"]},
    package_data={"toolcallbench": ["data/*.jsonl"]},
)
