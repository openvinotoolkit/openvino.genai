import os
import sys
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()


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
    name="whowhatbench",
    version=set_version("1.0.0"),
    url="https://github.com/openvinotoolkit/openvino.genai.git",
    author="Intel",
    author_email="andrey.anufriev@intel.com",
    description="Short test for LLMs",
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["wwb=whowhatbench.wwb:main"]},
    package_data={"whowhatbench": ["prompts/*.yaml"]}
)
