# Benchmarking script for large language models

This script provides a unified approach to estimate performance for Large Language Models.
It is based on pipelines provided by Optimum-Intel and allows to estimate performance for
pytorch and openvino models, using almost the same code and precollected models.

## Usage 

### 1. Start a Python virtual environment
   
``` bash
python3 -m venv python-env
source python-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
> Note:
> If you are using an existing python environment, recommend following command to use all the dependencies with latest versions:  
> pip install -U --upgrade-strategy eager -r requirements.txt

### 2. Convert a model to OpenVINO IR
   
The optimum-cli tool allows you to convert models from Hugging Face to the OpenVINO IR format

Prerequisites:
install conversion dependencies using `requirements.txt`

Usage:

```bash
optimum-cli export openvino --model <MODEL_NAME> --weight-format <PRECISION> <NEW_MODEL_NAME>
```

Paramters:
* `--model <MODEL_NAME>` - <MODEL_NAME> model_id for downloading from huggngface_hub (https://huggingface.co/models) or path with directory where pytorch model located. 
* `--weight-format` - precision for model conversion FP16 or INT8 or INT4
* `<NEW_MODEL_NAME>` - output directory for saving OpenVINO model.

Usage example:
```bash
optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format fp16 models/llama-2-7b-chat
```

the result of running the command will have the following file structure:

    |-llama-2-7b-chat
      |-pytorch
        |-dldt
           |-FP16
              |-openvino_model.xml
              |-openvino_model.bin
              |-config.json
              |-generation_config.json
              |-tokenizer_config.json
              |-tokenizer.json
              |-tokenizer.model
              |-special_tokens_map.json

### 3. Benchmarking

Prerequisites:
install benchmarking dependencies using `requirements.txt`

``` bash
pip install -r requirements.txt
```
note: **You can specify the installed OpenVINO version through pip install**
``` bash
# e.g. 
pip install openvino==2023.3.0
```

### 4. Run the following command to test the performance of one LLM model
``` bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters>
# e.g.
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -n 2
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -p "What is openvino?" -n 2
python benchmark.py -m models/llama-2-7b-chat/pytorch/dldt/FP32 -pf prompts/llama-2-7b-chat_l.jsonl -n 2
```
Parameters:
* `-m` - model path
* `-d` - inference device (default=cpu)
* `-r` - report csv
* `-f` - framework (default=ov)
* `-p` - interactive prompt text
* `-pf` - path of JSONL file including interactive prompts
* `-n` - number of benchmarking iterations, if the value greater 0, will exclude the first iteration. (default=0)
* `-ic` - limit the output token size (default 512) of text_gen and code_gen models.


``` bash
python ./benchmark.py -h # for more information
```

## Running `torch.compile()`

The option `--torch_compile_backend` uses `torch.compile()` to speed up
the PyTorch code by compiling it into optimized kernels using a selected backend.

Prerequisites: install benchmarking dependencies using requirements.txt

``` bash
pip install -r requirements.txt
```

In order to run the `torch.compile()` on CUDA GPU, install additionally the nightly PyTorch version:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Add the option `--torch_compile_backend` with the desired backend: `pytorch` or `openvino` (default) while running the benchmarking script:

```bash
python ./benchmark.py -m models/llama-2-7b-chat/pytorch -d CPU --torch_compile_backend openvino
```

## Run on 2 sockets platform

benchmark.py sets openvino.properties.streams.num(1) by default

| OpenVINO version    | Behaviors                                       |
|:--------------------|:------------------------------------------------|
| Before 2024.0.0 | streams.num(1) <br>execute on 2 sockets. |
| 2024.0.0 | streams.num(1) <br>execute on the same socket as the APP is running on. |

numactl on Linux or --load_config for benchmark.py can be used to change the behaviors.

For example, --load_config config.json as following in OpenVINO 2024.0.0 will result in streams.num(1) and execute on 2 sockets.
```
{"INFERENCE_NUM_THREADS":<NUMBER>}
```
`<NUMBER>` is the number of total physical cores in 2 sockets

## Additional Resources
### 1. NOTE
> If you encounter any errors, please check **[NOTES.md](./doc/NOTES.md)** which provides solutions to the known errors.
### 2. Image generation
> To configure more parameters for image generation models, reference to **[IMAGE_GEN.md](./doc/IMAGE_GEN.md)**
