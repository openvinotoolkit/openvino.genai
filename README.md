GenAI contains pipelines that implement image and text generation tasks. The implementation exploits OpenVINO capabilities to optimize the pipelines. Each sample covers a family of models and suggests that its implementation can be modified to adapt for a specific need.

> Note  
This project is not for production use.

Every pipeline requires https://github.com/openvinotoolkit/openvino for C++ to be installed.

Build the pipelines and `user_ov_extensions`

1.Clone submodules:
```sh
git clone https://github.com/wenyi5608/openvino.genai.git -b wenyi5608-chatglm
cd openvino.genai
git submodule update --init
```

2.Build & Run

For Linux

Compile the project using CMake:
```sh
source <OpenVINO dir>/setupvars.sh
cmake -B build
cmake --build build -j --config Release
```
Usage: `chatglm <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "device"`

Example: `./build/llm/chatglm_cpp/chatglm ./chatglm3-6b/openvino_model.xml ./tokenizer.xml ./detokenizer.xml "CPU"`

For Windows

Compile the project using CMake:
```sh
<OpenVINO dir>\setupvars.bat
cmake -B build
cmake --build build -j --config Release
```

copy user_ov_extensions.dll and its dependent dll to chatglm.exe directory

```sh
$copy build\thirdparty\openvino_contrib\modules\custom_operations\user_ie_extensions\Release\user_ov_extensions.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\third_party\lib\icudt70.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\third_party\lib\icuuc70.dll build\llm\chatglm_cpp\Release\
$copy build\_deps\fast_tokenizer-src\lib\core_tokenizers.dll build\llm\chatglm_cpp\Release\
```
Usage: `chatglm.exe <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> "device"`

Example: `build\llm\chatglm_cpp\Release\chatglm.exe .\chatglm3-6b\openvino_model.xml .\tokenizer.xml .\detokenizer.xml "CPU"`

To enable non ASCII characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
