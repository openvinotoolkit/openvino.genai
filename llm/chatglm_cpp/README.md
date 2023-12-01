# LLM

This application showcases inference of a large language model (LLM). It doesn't have much of configuration options to encourage the reader to explore and modify the source code. There's a Jupyter notebook which corresponds to this pipeline and discusses how to create an LLM-powered Chatbot: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot.

> Note  
This pipeline is not for production use.

## How it works

The program loads a tokenizer, detokenizer and a model (`.xml` and `.bin`) to OpenVINO™. The model is reshaped to batch 1 and variable prompt length. A prompt is tokenized and passed to the model. The model greedily generates token by token until the special end of sequence (EOS) token is obtained. The predicted tokens are converted to chars and printed in a streaming fashion.

## Supported models

1. chatglm
   1. https://hf-mirror.com/THUDM/chatglm2-6b
   2. https://hf-mirror.com/THUDM/chatglm3-6b

### Download and convert the model and tokenizers

```sh
python -m pip install --upgrade-strategy eager thirdparty/openvino_contrib/modules/custom_operations/user_ie_extensions/tokenizer/python/[transformers] onnx "optimum[openvino]>=1.14.0" --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r ./llm_bench/python/requirements.txt
python ./llm_bench/python/convert.py --model_id --model_id <model_id_or_path> --output_dir <out_dir>
source <OpenVINO dir>/setupvars.sh
python ./llm/cpp/convert_tokenizers.py ./build/thirdparty/openvino_contrib/modules/custom_operations/user_ie_extensions/libuser_ov_extensions.so ./chatglm3-6b/
```

### Run

For Linux
Usage: `chatglm <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "device" "<prompt>"`

Example: `./build/llm/chatglm_cpp/chatglm ./chatglm3-6b/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "CPU" "Why is the Sun yellow?"`

For Windows
copy openvino dll and tbb12.dll to chatglm.exe directory
```sh
$copy <OpenVINO dir>\runtime\bin\intel64\Release\* build\llm\chatglm_cpp\Release\
$copy <OpenVINO dir>\runtime\3rdparty\tbb\bin\tbb12.dll build\llm\chatglm_cpp\Release\
```

copy user_ov_extensions.dll and its dependent dll to chatglm.exe directory
```sh
$copy build\thirdparty\openvino_contrib\modules\custom_operations\user_ie_extensions\Release\user_ov_extensions.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\third_party\lib\icudt70.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\third_party\lib\icuuc70.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\lib\core_tokenizers.dll build\llm\chatglm_cpp\Release\
```
Usage: `chatglm.exe <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "device" "<prompt>"`

Example: `build\llm\chatglm_cpp\Release\chatglm.exe .\chatglm3-6b\openvino_model.xml .\tokenizer.xml .\detokenizer.xml "CPU" "Why is the Sun yellow?"`
