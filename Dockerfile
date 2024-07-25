FROM ubuntu:22.04

ARG JOBS
WORKDIR /workspace
RUN apt-get update -y && apt-get install -y python3-pip python3-venv git

# Install OpenVINO
RUN git clone --branch master https://github.com/openvinotoolkit/openvino.git && \
    cd /workspace/openvino && \
    git submodule update --init -- /workspace/openvino/thirdparty/xbyak /workspace/openvino/thirdparty/pugixml /workspace/openvino/thirdparty/open_model_zoo \
        /workspace/openvino/thirdparty/protobuf /workspace/openvino/thirdparty/snappy /workspace/openvino/thirdparty/telemetry /workspace/openvino/src/plugins/intel_cpu/thirdparty/mlas \
        /workspace/openvino/src/plugins/intel_cpu/thirdparty/onednn /workspace/openvino/src/bindings/python/thirdparty/pybind11 && cd -

RUN /workspace/openvino/install_build_dependencies.sh
RUN python3 -m pip install -r /workspace/openvino/src/bindings/python/wheel/requirements-dev.txt
RUN cmake -DENABLE_PYTHON=ON -DENABLE_PYTHON_PACKAGING=ON -DENABLE_WHEEL=ON -DENABLE_CPPLINT=OFF -DENABLE_SAMPLES=OFF -DENABLE_INTEL_GPU=OFF \
        -DENABLE_INTEL_NPU=OFF -DENABLE_TEMPLATE=OFF -DENABLE_AUTO=OFF -DENABLE_HETERO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_OV_TF_FRONTEND=ON -DENABLE_OV_ONNX_FRONTEND=OFF \
        -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -S /workspace/openvino -B /workspace/openvino_build
RUN cmake --build /workspace/openvino_build --parallel $JOBS
RUN cmake -P /workspace/openvino_build/cmake_install.cmake
RUN python3 -m pip install /workspace/openvino_build/wheels/openvino-2024* 
ENV OpenVINO_DIR=/workspace/openvino_build

# Download dataset
RUN wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Build GenAI library with dependencies
RUN git clone https://github.com/Wovchena/openvino.genai-public.git -b reuse-Tokenizer openvino.genai && \
        cd /workspace/openvino.genai/thirdparty && git submodule update --remote --init && \
        mkdir /workspace/openvino.genai/build && cd /workspace/openvino.genai/build && \
        cmake -DCMAKE_BUILD_TYPE=Release .. && \
        make -j${JOBS}

# Install test dependencies
RUN python3 -m pip install --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly/ /workspace/openvino.genai/thirdparty/openvino_tokenizers
RUN PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python3 -m pip install -r /workspace/openvino.genai/tests/python_tests/continuous_batching/requirements.txt
ENV PYTHONPATH=/workspace/openvino.genai/build/
ENV LD_LIBRARY_PATH=/workspace/openvino.genai/build/
