from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="whowhatbench",
    version="1.0.0",
    url="https://github.com/openvinotoolkit/openvino.genai.git",
    author="Intel",
    author_email="andrey.anufriev@intel.com",
    description="Short test for LLMs",
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["wwb=whowhatbench.wwb:main"]},
)
