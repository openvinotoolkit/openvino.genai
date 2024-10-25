# Building openvino_llm:latest genai docker image
```Bash
git clone --branch ct-beam-search https://github.com/ilya-lavrenov/openvino.genai.git
git submodule update --remote --init
cd text_generation/causal_lm/cpp/continuous_batching/
make
```

```Bash
cd ../../../..
docker run -it -v `pwd`:/workspace/openvino.genai/ openvino_llm:latest
cd /workspace/openvino.genai/
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/ && cmake --build ./build/ -j
```

# Downloading LLM models
```Bash
cd /workspace/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/
optimum-cli export openvino --model facebook/opt-125m ./ov_model
```

# Running throuput benchmark application
```Bash
cd /workspace/openvino.genai/
./build/text_generation/causal_lm/cpp/continuous_batching/apps/throughput_benchmark --model /workspace/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/ov_model --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --dynamic_split_fuse --num_prompts 100 --device CPU --plugin_config {/"ENABLE_PROFILING/":true}
```


# How to create environment to debug and develop continuous batching project with OpenVINO:

1. Build OpenVINO with python bindings:
```
cd /path/to/openvino
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE={ov_build_type} ..
make -j24
```
2. Set PYTHONPATH, LD_LIBRARY_PATH and OpenVINO_DIR environment variables:
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/openvino/bin/intel64/{ov_build_type}
export PYTHONPATH=${PYTHONPATH}:/path/to/openvino/bin/intel64/Release/python:/path/to/openvino/tools/ovc
export OpenVINO_DIR=/path/to/openvino/{ov_build_type}
```
3. Build OpenVINO tokenizers:
```
cd /path/to/openvino.genai/thirdparty/openvino_tokenizers
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE={ov_build_type} ..
make -j24
```
4. Create virtual environment to generate models and run python tests:
> NOTE: Comment installation of `openvino` and `openvino_tokenizers` to your env in `/path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/python/tests/requirements.txt
```
cd /path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching
python3 -m venv .env
source .env/bin/activate
pip3 install -r python/tests/requirements.txt
```
5. Install `openvino_tokenizers` to your virtual environment:
```
cd /path/to/openvino.genai/thirdparty/openvino_tokenizers
export OpenVINO_DIR=/path/to/openvino/build
pip install --no-deps .
```
6. Create build directory in `continuous batching` project:
```
mkdir /path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/build
```
7. Generate cmake project:
```
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DOpenVINO_DIR=/path/to/openvino/build ..
```
8. Build the project
```
make -j24
```
9. Extend `PYTHONPATH` by `continuous batching`:
```
export PYTHONPATH=${PYTHONPATH}:/path/to/openvino.genai/text_generation/causal_lm/cpp/continuous_batching/build/python
```
10. Run python tests:
```
cd python/tests
pytest .
```
