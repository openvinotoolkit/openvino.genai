# Quick Setup (Reproducible)

This guide reproduces a working SmolVLM + OpenVINO GenAI setup on a fresh Linux machine with OpenVINO and OpenVINO-genai 2025.4 version. 

## 1) System prerequisites

```bash
sudo apt update
sudo apt install -y git build-essential cmake ninja-build 
```

## 2) Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install --upgrade pip setuptools wheel
```

## 3) Install Python dependencies in venv

For OpenVINO 2026
```bash
pip install --pre --upgrade openvino==2026.1.0.dev20260217 openvino-tokenizers==2026.1.0.0.dev20260217 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
For OpenVINO 2025.4
```bash
pip install -U openvino==2025.4.1 openvino-tokenizers==2025.4.1.0 
```
```bash
pip install transformers pillow numpy optimum-intel nncf
```

## Download model
```
# Export the model
optimum-cli export openvino \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --weight-format int4 \
    --sym \
    --group-size 128 \
    --ratio 1.0 \
    --trust-remote-code \
    SmolVLM-instruct_int4_sym_group-128
```

## 4) Clone OpenVINO GenAI and initialize submodules

For OpenVINO 2026
```bash
git clone --recursive  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai
```
For OpenVINO 2025.4
```bash
git clone --recursive --branch 2025.4.1.0 https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai
``` 

## 5) Apply patch

For OpenVINO 2026
```bash
git apply ../smolvlm_idefics3_support_ov2026.patch
git status
```
For OpenVINO 2025.4
```bash
git apply ../smolvlm_idefics3_support_ov2025-4.patch
git status
```

## 6) Build OpenVINO GenAI (library + Python bindings)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
cmake --build build --parallel
cd ..
```


## 7) Use built bindings without `pip install .`

This avoids expensive source wheel builds and works reliably on low-memory systems.
use-built-python-bindings.sh

```
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${REPO_DIR}/openvino.genai/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO_DIR}/openvino.genai/build/openvino_genai:${LD_LIBRARY_PATH:-}"

exec "$@"
```

Run Python commands through wrapper:

```bash
./use-built-python-bindings.sh python -c "import openvino_genai; print(openvino_genai.__version__)"
```


## 8) Test inference

Python script to test the inference
```bash
#!/usr/bin/env python3
"""
Test script for SmolVLM model with OpenVINO GenAI
"""
import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
import sys

def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_smolvlm.py <model_dir> <image_path> [device] [prompt]")
        print("Example: python test_smolvlm.py SmolVLM-instruct_int4_sym_group-1 test.jpg CPU")
        sys.exit(1)

    model_dir = sys.argv[1]
    image_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "CPU"
    prompt = sys.argv[4] if len(sys.argv) > 4 else "Describe this image in detail."

    print(f"Loading SmolVLM model from: {model_dir}")
    print(f"Device: {device}")

    # Load the model
    pipe = openvino_genai.VLMPipeline(model_dir, device)

    # Read the image
    print(f"Reading image: {image_path}")
    image = read_image(image_path)

    # Configure generation
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.do_sample = False  # Use greedy decoding
    config.apply_chat_template = False  # Disable chat template and format manually

    # Disable chat template and format manually
    # SmolVLM uses format: <|im_start|>User:<image>\n{prompt}<end_of_utterance>\n<|im_start|>Assistant:
    formatted_prompt = f"User:<image>\n{prompt}\nAssistant:"

    # Test with the provided prompt
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    # Generate response
    result = pipe.generate(formatted_prompt, images=[image], generation_config=config)

    print(f"\nResponse: {result.texts[0]}")
    print("\nSuccess! SmolVLM is working with OpenVINO GenAI!")

if __name__ == "__main__":
    main()
```

```bash
./use-built-python-bindings.sh \
  python test-smolvlm.py SmolVLM-instruct_int4_sym_group-128 cat.jpg GPU "Describe this image"
```

(Try with GPU / NPU / NPU)

---


