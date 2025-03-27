# Benchmarking Script for Large Language Models

This script provides a unified approach to estimate performance for Large Language Models (LLMs). It leverages pipelines provided by Optimum-Intel and allows performance estimation for PyTorch and OpenVINO models using nearly identical code and pre-collected models.


### 1. Prepare Python Virtual Environment for LLM Benchmarking
   
``` bash
python3 -m venv ov-llm-bench-env
source ov-llm-bench-env/bin/activate
pip install --upgrade pip

git clone  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt  
```

> Note:
> For existing Python environments, run the following command to ensure that all dependencies are installed with the latest versions:  
> `pip install -U --upgrade-strategy eager -r requirements.txt`

#### (Optional) Hugging Face Login :

Login to Hugging Face if you want to use non-public models:

```bash
huggingface-cli login
```

### 2. Convert Model to OpenVINO IR Format
   
The `optimum-cli` tool simplifies converting Hugging Face models to OpenVINO IR format. 
- Detailed documentation can be found in the [Optimum-Intel documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export). 
- To learn more about weight compression, see the [NNCF Weight Compression Guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).
- For additional guidance on running inference with OpenVINO for LLMs, see the [OpenVINO Generative AI workflow](https://docs.openvino.ai/2025/openvino-workflow-generative.html).

**Usage:**

```bash
optimum-cli export openvino --model <MODEL_ID> --weight-format <PRECISION> <OUTPUT_DIR>

optimum-cli export openvino -h # For detailed information
```

* `--model <MODEL_ID>` : model_id for downloading from [huggingface_hub](https://huggingface.co/models) or path with directory where pytorch model located. 
* `--weight-format <PRECISION>` : precision for model conversion. Available options: `fp32, fp16, int8, int4, mxfp4`
* `<OUTPUT_DIR>`: output directory for saving generated OpenVINO model.

**NOTE:** 
- Models larger than 1 billion parameters are exported to the OpenVINO format with 8-bit weights by default. You can disable it with `--weight-format fp32`.

**Example:**
```bash
optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format fp16 models/llama-2-7b-chat
```
**Resulting file structure:**

```console
    models
    └── llama-2-7b-chat
        ├── config.json
        ├── generation_config.json
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── tokenizer.model
```

### 3. Benchmark LLM Model

To benchmark the performance of the LLM, use the following command:

``` bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters>
# e.g.
python benchmark.py -m models/llama-2-7b-chat/ -n 2
python benchmark.py -m models/llama-2-7b-chat/ -p "What is openvino?" -n 2
python benchmark.py -m models/llama-2-7b-chat/ -pf prompts/llama-2-7b-chat_l.jsonl -n 2
```

**Parameters:**
- `-m`: Path to the model.
- `-d`: Inference device (default: CPU).
- `-r`: Path to the CSV report.
- `-f`: Framework (default: ov).
- `-p`: Interactive prompt text.
- `-pf`: Path to a JSONL file containing prompts.
- `-n`: Number of iterations (default: 0, the first iteration is excluded).
- `-ic`: Limit the output token size (default: 512) for text generation and code generation models.

**Additional options:**
``` bash
python ./benchmark.py -h # for more information
```

#### Benchmarking the Original PyTorch Model:
To benchmark the original PyTorch model, first download the model locally and then run benchmark by specifying PyTorch as the framework with parameter `-f pt`

```bash
# Download PyTorch Model
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b-chat/pytorch
# Benchmark with PyTorch Framework
python benchmark.py -m models/llama-2-7b-chat/pytorch -n 2 -f pt
```

> **Note:** If needed, You can install a specific OpenVINO version using pip:
> ``` bash
> # e.g. 
> pip install openvino==2024.4.0
> # Optional, install the openvino nightly package if needed.
> # OpenVINO nightly is pre-release software and has not undergone full release validation or qualification. 
> pip uninstall openvino
> pip install --upgrade --pre openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
> ```

## 4. Benchmark LLM with `torch.compile()`

The `--torch_compile_backend` option enables you to use `torch.compile()` to accelerate PyTorch models by compiling them into optimized kernels using a specified backend.

Before benchmarking, you need to download the original PyTorch model. Use the following command to download the model locally:

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b-chat/pytorch
```

To run the benchmarking script with `torch.compile()`, use the `--torch_compile_backend` option to specify the backend. You can choose between `pytorch` or `openvino` (default). Example:

```bash
python ./benchmark.py -m models/llama-2-7b-chat/pytorch -d CPU --torch_compile_backend openvino
```

> **Note:** To use `torch.compile()` with CUDA GPUs, you need to install the nightly version of PyTorch:
>
> ```bash
> pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
> ```


## 5. Running on 2-Socket Platforms

The benchmarking script sets `openvino.properties.streams.num(1)` by default. For multi-socket platforms, use `numactl` on Linux or the `--load_config` option to modify behavior.

| OpenVINO Version    | Behaviors                                       |
|:--------------------|:------------------------------------------------|
| Before 2024.0.0     | streams.num(1) <br>execute on 2 sockets.        |
| 2024.0.0            | streams.num(1) <br>execute on the same socket as the APP is running on. |

For example, `--load_config config.json` as following will result in streams.num(1) and execute on 2 sockets.
```json
{
  "INFERENCE_NUM_THREADS": <NUMBER>
} 
``` 
`<NUMBER>` is the number of total physical cores in 2 sockets.

## 6. Execution on CPU device

OpenVINO is by default built with [oneTBB](https://github.com/oneapi-src/oneTBB/) threading library, while Torch uses [OpenMP](https://www.openmp.org/). Both threading libraries have ['busy-wait spin'](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html) by default. When running LLM pipeline on CPU device, there is threading overhead in the switching between inference on CPU with OpenVINO (oneTBB) and postprocessing (For example: greedy search or beam search) with Torch (OpenMP). The default benchmarking scenarion uses OpenVINO GenAI that implements own postprocessing api without additional dependencies.

**Alternative solutions**
1. With --optimum option which uses optimum-intel API, set environment variable [OMP_WAIT_POLICY](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html) to PASSIVE which will disable OpenMP 'busy-wait', and benchmark.py will limit the Torch thread number by default to avoid using CPU cores which is in 'busy-wait' by OpenVINO inference. Users can also set the number with --set_torch_thread option.

## 7. Additional Resources

- **Error Troubleshooting:** Check the [NOTES.md](./doc/NOTES.md) for solutions to known issues.
- **Syntax and attributes of prompt file:** Refer to [PROMPT.md](./doc/PROMPT.md) for writing a prompt file.
