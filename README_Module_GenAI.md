# OpenVINO™ Modular GenAI

OpenVINO™ Modular GenAI is POC. It redefines GenAI application development by decomposing complex workflows into independent, reusable modules. By orchestrating these modules through a Directed Acyclic Graph (DAG), developers can build sophisticated AI pipelines with unprecedented flexibility and efficiency.

#### Key Architectural Pillars

`Extensibility & Reusability:` Designed for developers unfamiliar with underlying GenAI complexities, the framework allows for the rapid integration of new modules while seamlessly reusing existing ones. This "plug-and-play" approach drastically flattens the learning curve.

`Zero-Copy Efficiency:` New framework ensures no new data copies are introduced during inter-module communication via ov::Tensor. This maintains high throughput and minimizes memory pressure.

`Asynchronous Processing:` The DAG structure natively supports asynchronous execution between modules. This allows independent tasks to run in parallel, significantly reducing end-to-end latency for multi-stage inference tasks.

`YAML-Driven Orchestration:` High-level execution graphs are defined via human-readable YAML files. This provides a clear, declarative overview of the logic flow, making it exceptionally beginner-friendly and easy to version control.

## Getting Started

<details>
<summary>ENV</summary>

```
python -m venv python-env
source python-env/bin/activate
pip install numpy
```
</details>


<details>
<summary>Prepare OpenVINO</summary>

```
<!-- Version 25.4 or compiled from source code. The following branches include some advanced features. -->
https://github.com/xipingyan/openvino.git --branch master_modular_genai

git clone https://github.com/openvinotoolkit/openvino.git --branch 2025.4.0
cd openvino && mkdir build && cd build
git submodule update --init
cmake -DCMAKE_INSTALL_PREFIX=install ..
make -j20 && make install
```
</details>

<details>
<summary>Build GenAI</summary>

```
sudo apt-get install libyaml-cpp-dev

source ./python-env/bin/activate
source ./openvino/build/install/setupvars.sh

git clone https://github.com/xipingyan/openvino.genai.git --branch master_modular_genai
cd openvino.genai
git submodule update --init

cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j 20
cmake --install ./build/ --config Release --prefix ./install
```
</details>

## Samples

[CPP Samples](samples/cpp/module_genai/README.md)   <br>
[CPP ComfyUI Sample](samples/cpp/module_genai/comfyui/README.md) <br>
[Python Sample](samples/python/module_genai/README.md) <br>

<details>
<summary>Qwen2.5-VL</summary>

```
<!-- Convert model -->
python -m venv pyenv_cvt_model
source pyenv_cvt_model/bin/activate
pip install openvino-tokenizers openvino nncf optimum[intel]
pip install -U huggingface_hub

model_id='Qwen/Qwen2.5-VL-3B-Instruct'
optimum-cli export openvino --model $model_id --task image-text-to-text $model_id/INT4 --weight-format int4 --trust-remote-code
```

```
<!-- Python example -->
import openvino_genai

config_file="./config.yaml"

pipe = openvino_genai.ModulePipeline(config_file)

pipe.generate(img1=ov::Tensor, prompt_data="Describle the image")
output = pipe.get_output("generated_text")

print("output = ", output)
```
``Note:`` Reference [config.yaml](samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml). Please update the `model_path` of config.yaml with your local path.

</details>