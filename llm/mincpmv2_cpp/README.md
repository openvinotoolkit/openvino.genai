# multimodal LLMs

This application showcases inference of a multimodal LLMs. It doesn't have much of configuration options to encourage the reader to explore and modify the source code. 

> Note  
This pipeline is not for production use.

## How it works

The program loads multimodal model to OpenVINO™. The models take image and text as inputs and provide high-quality text outputs. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Install OpenVINO Runtime

Install OpenVINO Runtime from an archive: [Linux](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html) or [Windows](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html). `<INSTALL_DIR>` below refers to the extraction location.

## Supported model

1. minicpmv2
   1. https://hf-mirror.com/openbmb/MiniCPM-V-2


Build the pipelines and `openvino_tokenizers`

1.Clone submodules:

```sh
git clone https://github.com/wenyi5608/openvino.genai.git -b wenyi5608-stateful
git submodule update --init
```

2.Build

For Linux

Compile the project using CMake:

```sh
source <OpenVINO dir>/setupvars.sh
cmake -B build
cmake --build build -j --config Release
```

For Windows

Compile the project using CMake:

```sh
<OpenVINO dir>\setupvars.bat
cmake -B build
cmake --build build -j --config Release
```

3.Download and convert the model and tokenizers

For linux

3.1 Download python code
```sh
git clone https://github.com/wenyi5608/MiniCPM-V.git -b ov_runtime
cd eval_mm/openvinoruntime
```

3.2 create python environment
```sh
conda create -n ov_minicpmv2 python=3.10
conda activate ov_minicpmv2
pip install -r requirements.txt
pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
3.3 Convert minicpmv2 model to OpenVINO™ IR(Intermediate Representation).
```sh
python ov_convert_minicpm-v-2.py -m /path/to/minicpmv2 -o /path/to/minicpmv2_ov
```

4. Run

Paramters:
* `-h, --help `             show this help message and exit\n"
* `-m, --model` PATH        minicpm OpenVINO model path (default: openvino_model.xml)\n"
* `-vision` PATH            minicpmv vision model path (default: minicpm-v-2_vision.xml)\n"
* `-resam` PATH             minicpmv resampler model path (default: minicpm-v-2_resampler.xml)\n"
* `-embed` PATH             minicpmv embedding model path (default: minicpm-v-2_embedding.xml)\n"
* `-token` PATH             Tokenizer model path (default: tokenizer.xml)\n"
* `-detoken` PATH           DeTokenizer model path (default: detokenizer.xml)\n"
* `-d, --device`            Device (default: CPU)\n"
* `--image P`ATH            path to an image file. use with multimodal models. Specify multiple times for batching\n";
   
For Linux


Usage: `minicpmv2 -m <minicpm-v-2_openvino-int4.xml> -vision <minicpm-v-2_vision.xml> -resam <minicpm-v-2_resampler.xml> -embed <minicpm-v-2_embedding.xml> -token <minicpm-v-2_openvino_tokenizer.xml> -detoken <minicpm-v-2_openvino_detokenizer.xml> --image <airplane.jpeg> -d GPU`

Example: `./build/llm/mincpmv2_cpp/minicpmv2 -m "\path\to\minicpm-v-2_openvino-int4.xml" -vision "\path\to\minicpm-v-2_vision.xml" -resam "\path\to\minicpm-v-2_resampler.xml" -embed "\path\to\minicpm-v-2_embedding.xml" -token "\path\to\minicpm-v-2_openvino_tokenizer.xml" -detoken "\path\to\minicpm-v-2_openvino_detokenizer.xml"  --image "\path\to\airplane.jpeg" -d GPU`

For Windows

Usage: `minicpmv2.exe -m <minicpm-v-2_openvino-int4.xml> -vision <minicpm-v-2_vision.xml> -resam <minicpm-v-2_resampler.xml> -embed <minicpm-v-2_embedding.xml> -token <minicpm-v-2_openvino_tokenizer.xml> -detoken <minicpm-v-2_openvino_detokenizer.xml> --image <airplane.jpeg> -d GPU`

Example: `build\llm\mincpmv2_cpp\Release\minicpmv2.exe -m "\path\to\minicpm-v-2_openvino-int4.xml" -vision "\path\to\minicpm-v-2_vision.xml" -resam "\path\to\minicpm-v-2_resampler.xml" -embed "\path\to\minicpm-v-2_embedding.xml" -token "\path\to\minicpm-v-2_openvino_tokenizer.xml" -detoken "\path\to\minicpm-v-2_openvino_detokenizer.xml"  --image "\path\to\airplane.jpeg" -d GPU`

 

