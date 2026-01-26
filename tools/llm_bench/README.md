# Benchmarking Script for Large Language Models

This script provides a unified approach to estimate performance for Large Language Models (LLMs). It leverages pipelines provided by Optimum-Intel and allows performance estimation for PyTorch and OpenVINO models using nearly identical code and pre-collected models.

This tool is designed for performance estimation, not accuracy validation. For accuracy checks, refer to the [wwb tool](https://github.com/openvinotoolkit/openvino.genai/blob/master/tools/who_what_benchmark/README.md).

For Text Generation Pipeline prompt modifications are turned on by default from iteration to iteration. It's enabled to avoid prefix caching. If you need to have equal results on each iteration, please, run tool with --disable_prompt_permutation.

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

### 3. Benchmark OpenVINO IRs Models

To benchmark the performance of the LLM, use the following command:

``` bash
python benchmark.py -m <model> -d <device> -r <report_csv> -f <framework> -p <prompt text> -n <num_iters> --task <pipeline_type>
# e.g.
python benchmark.py -m models/llama-2-7b-chat/ -n 2 --task text_gen
python benchmark.py -m models/llama-2-7b-chat/ -p "What is openvino?" -n 2 --task text_gen
python benchmark.py -m models/llama-2-7b-chat/ -pf prompts/llama-2-7b-chat_l.jsonl -n 2 --task text_gen
```

**General parameters:**
- `-m`: Path to the model.
- `-d`: Inference device (default: CPU).
- `-r`: Path to the CSV report.
- `-rj`: Report in JSON format.
- `-f`: Framework (default: ov).
- `-p`: Interactive prompt text.
- `-pf`: Path to a JSONL file containing prompts.
- `-n`: Number of iterations (default: 0, the first iteration is excluded).
- `-ic`: Limit the output token size (default: 512) for text generation and code generation models.
- `-lc`: Path to JSON file to load customized configurations.
- `--optimum`: Use Optimum Intel pipelines for benchmarking.
- `--from_onnx`: Allow initialize Optimum OpenVINO model using ONNX.
- `--pruning_ratio`: Percentage of visual tokens to prune (valid range: 0-100). If this option is not provided, pruning is disabled.
- `--relevance_weight`: Float value from 0 to 1, control the trade-off between diversity and relevance for visual tokens pruning, a value of 0 disables relevance weighting, while higher values (up to 1.0) emphasize relevance, making pruning more conservative on borderline tokens.

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
python benchmark.py -m models/llama-2-7b-chat/pytorch -n 2 -f pt --task text_gen
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

## 4. Benchmark models with `torch.compile()`

The `--torch_compile_backend` option enables you to use `torch.compile()` to accelerate PyTorch models by compiling them into optimized kernels using a specified backend.

Before benchmarking, you need to download the original PyTorch model. Use the following command to download the model locally:

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b-chat/pytorch
```

To run the benchmarking script with `torch.compile()`, use the `--torch_compile_backend` option to specify the backend. You can choose between `pytorch` or `openvino` (default). Example:

```bash
python ./benchmark.py -m models/llama-2-7b-chat/pytorch -d CPU --torch_compile_backend openvino
```

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

OpenVINO is by default built with [oneTBB](https://github.com/oneapi-src/oneTBB/) threading library, while Torch uses [OpenMP](https://www.openmp.org/). Both threading libraries have ['busy-wait spin'](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html) by default. When running LLM pipeline on CPU device, there is threading overhead in the switching between inference on CPU with OpenVINO (oneTBB) and postprocessing (For example: greedy search or beam search) with Torch (OpenMP). The default benchmarking scenario uses OpenVINO GenAI that implements own postprocessing api without additional dependencies.

**Alternative solutions**
1. With --optimum option which uses optimum-intel API, set environment variable [OMP_WAIT_POLICY](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html) to PASSIVE which will disable OpenMP 'busy-wait', and benchmark.py will limit the Torch thread number by default to avoid using CPU cores which is in 'busy-wait' by OpenVINO inference. Users can also set the number with --set_torch_thread option.

## 7. Supported use cases
### Text Generation Models (LLMs)
```sh
python benchmark.py -m ./models/llama-2-7b-chat/ -p "What is openvino?" -n 2 --task text_gen
```
**Some additional parameters:**
- `--cb_config`: Path to file with Continuous Batching Scheduler settings or dict".
- `--disable_prompt_permutation`: "Disable modification prompt from run to run for avoid prefix caching"
- `--apply_chat_template`: "Apply chat template for LLM. By default chat template is not applied"
- `--from_onnx`: "Load the model from an ONNX file instead of a pre-converted OpenVINO IR."

```sh
optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/TinyLlama-1.1B-Chat-v1.0
# speculative decoding
python benchmark.py -m models/llama-2-7b-chat/ --draft_model models/TinyLlama-1.1B-Chat-v1.0 -p "What is openvino?" -n 2 --task text_gen --num_assistant_tokens 5
```

**Some additional parameters:**
- `--draft_device`: Inference device for Speculative decoding of draft model.
- `--draft_cb_config`: Path to file with Continuous Batching Scheduler settings or dict for Speculative decoding of draft model.
- `--assistant_confidence_threshold`: The lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number.

```sh
# prompt lookup decoding
python benchmark.py -m models/llama-2-7b-chat/ -p "What is openvino?" -n 2 --task text_gen --max_ngram_size 3 --num_assistant_tokens 5
```

> **Supported LLM model types:** arcee, decoder, falcon, glm, aquila, gpt2, open-llama, openchat, neural-chat, llama, tiny-llama, tinyllama, opt, pythia, stablelm, stable-zephyr, rocket, vicuna, dolly, bloom, red-pajama, xgen, longchat, jais, orca-mini, baichuan, qwen, zephyr, mistral, mixtral, phi2, minicpm, gemma, deci, phi3, deci, internlm, olmo, starcoder, instruct-gpt, granite, granitemoe, gptj, t5, gpt, mpt, blenderbot, chatglm, yi, phi

### Visual Language Models (VLMs)
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code models/MiniCPM-V-2_6
# run benchmark.py
python benchmark.py -m models/MiniCPM-V-2_6/ -p "What is openvino?" -n 2 --task visual_text_gen -i ./image.png
```

> **Supported VLM model types:** llava, llava-next, qwen2-vl, llava-qwen2, internvl-chat, minicpmv, phi3-v, minicpm-v, minicpmo, maira2, qwen2-5-vl

### Image Generation Models
```sh
# convert model to OpenVINO IRs format
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16 models/dreamlike_anime_1_0_ov/FP16
# text to image
python benchmark.py -m models/dreamlike_anime_1_0_ov/FP16 -p "scat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney" -n 2 --task text-to-image
# image to image
python benchmark.py -m models/dreamlike_anime_1_0_ov/FP16 -p "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney" -n 2 --task image-to-image --media ./image.png
# inpainting
python benchmark.py -m models/dreamlike_anime_1_0_ov/FP16 -p "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney" -n 2 --task inpainting --media ./image.png --mask_image ./mask.png
```

**Some additional parameters:**
- `--height`: Generated image height.
- `--width`: Generated image width.
- `--num_steps`: Number of inference steps for image generation.
- `--static_reshape`: Reshape image generation pipeline to specific width & height at pipline creation time.
- `--guidance_scale`: guidance_scale parameter for pipeline, supported via json JSON input only.
- `--images`: Like a `--media`, path to the directory or single image.

> **Supported Image Generation model types:** stable-diffusion, ssd, tiny-sd, small-sd, lcm, sdxl, dreamlike, flux

### Generation with LoRA
LoRA is supported for Text Generation and Image Generation OpenVINO GenAI Pipelines.
```sh
# load LoRA adapters
wget -O soulcard.safetensors https://civitai.com/api/download/models/72591
# run text to image pipeline with lora
python benchmark.py -m models/dreamlike_anime_1_0_ov/FP16 -p "curly-haired unicorn in the forest, anime, line" -n 2 --task text-to-image -i ./image.png --lora soulcard.safetensors --lora_alphas 0.7
```

**Some additional parameters:**
- `--lora`: The list of paths to LoRA adapters for using OpenVINO GenAI optimized pipelines with LoRA for benchmarking.
- `--lora_alphas`: The list of alphas params for LoRA adapters.
- `--lora_mode`: LoRA adapters loading mode: auto, fuse, static, static_rank, dynamic.
- `--empty_lora`: Inference with empty lora config.

### Text to Speech models
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" models/speecht5_tts
# load speaker embeddings
wget https://huggingface.co/datasets/Xenova/cmu-arctic-xvectors-extracted/resolve/main/cmu_us_awb_arctic-wav-arctic_a0001.bin
# run benchmark.py
python benchmark.py -m models/speecht5_tts/ -p "Hello OpenVINO GenAI" -n 2 --task text_to_speech --speaker_embeddings ./cmu_us_awb_arctic-wav-arctic_a0001.bin
```

**Some additional parameters:**
- `--vocoder_path`: Path to vocoder model

> **Supported Text to Speech model types:** speecht5

### Speech to Text models
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model openai/whisper-base models/whisper-base
# load audio
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav
# run benchmark.py
python benchmark.py -m models/whisper-base/ --media ./how_are_you_doing_today.wav -n 2 --task speech_to_text
```

> **Supported Text to Speech model types:** whisper

### Text Rerank models
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model cross-encoder/ms-marco-MiniLM-L2-v2 --task text-classification models/ms-marco-MiniLM-L2-v2
# run benchmark.py
python benchmark.py -m models/ms-marco-MiniLM-L2-v2/ -n 2 --task text_rerank
```

**Some additional parameters:**
- `-p`: Query.
- `--texts`: Text or list of texts of documents for reranking based on their relevance to a query.
- `--reranking_max_length`: Max length for text reranking. Input text will be padded or truncated to specified value.
- `--reranking_top_n`: Number of top results to return for text reranking
- `--texts_file`: Files with texts in JSONL format with candidates for reranking based on relevance to a prompt(query). Multiple files should be separated with space(s).

> **Supported Text Rerank model types:**: bge, bert, albert, roberta, xlm-roberta, qwen3

### Compare Text Embeddings models
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model BAAI/bge-small-en-v1.5 --task feature-extraction models/bge-small-en-v1.5
# run benchmark.py
python benchmark.py -m models/bge-small-en-v1.5/ -n 2 --task text_embed
```

**Some additional parameters:**
- `-p`: Text for creating embeddings
- `--embedding_pooling`: Pooling type CLS or MEAN for encoders, LAST_TOKEN for decoders. Different post-processing is applied depending on the padding side.
- `--embedding_normalize`: Normalize embeddings
- `--embedding_max_length`: Max length for text embeddings. Input text will be padded or truncated to specified value.
- `--embedding_padding_side`: Side to use for padding 'left' or 'right'.

> **Supported Text Embeddings model types:**: bge, bert, albert, roberta, xlm-roberta, qwen3

### Code Generation models
```sh
# convert model to OpenVINO IR format
optimum-cli export openvino --model Salesforce/codegen-350M-multi models/codegen-350M-multi
# run benchmark.py
python benchmark.py -m models/codegen-350M-multi -p "def hello_world():" -n 2 --task code_gen
```

> **Supported Code Generation model types:**: codegen, codegen2, stable-code, replit, codet5

### Video Generation Models
```sh
python benchmark.py -m models/LTX-Video/FP16 -p "A cat plays with ball on the christmas tree." --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" --num_frames 5 -n 2 --num_steps 25 --task text-to-video
```

**Some additional parameters:**
- `--height`: Generated video height.
- `--width`: Generated video width.
- `--num_steps`: Number of inference steps for video generation.
- `--num_frames`: Number of frames in generated video.
- `--frame_rate`: Frame rate for video generation and saving.
- `--static_reshape`: Reshape video generation pipeline to specific width & height at pipeline creation time.
- `--guidance_scale`: guidance scale parameter for pipeline, supported via json JSON input only.
- `--guidance_rescale`: guidance rescale parameter for pipeline, supported via json JSON input only. **Note:** Currently not supported by LTX Pipeline with OpenVINO GenAI.

> **Supported Video Generation model types:** Lightricks/LTX-Video

## 8. Memory consumption mode
Enables memory usage information collection mode. This mode affects execution time, so it is not recommended to run memory consumption and performance benchmarking at the same time. Effect on performance can be reduced by specifying a longer --memory_consumption_delay, but the impact is still expected.

```sh
# run benchmark.py in memory consumption mode
python benchmark.py -m models/llama-2-7b-chat/ -p "What is openvino?" -n 2 --task text_gen -mc 2 -mc_dir ./mem_output_info
```

**Parameters:**
- `-mc, --memory_consumption`: Enables memory usage information collection mode. If the value is 1, output the maximum memory consumption in warm-up iterations. If the value is 2, output the maximum memory consumption in all iterations.
- `--memory_consumption_delay`: Delay for memory consumption check in seconds, smaller value will lead to more precised memory consumption, but may affects performance.
- `-mc_dir, --memory_consumption_dir`: Path to store memory consamption logs and chart.

## 9. Additional Resources

- **Error Troubleshooting:** Check the [NOTES.md](./doc/NOTES.md) for solutions to known issues.
- **Syntax and attributes of prompt file:** Refer to [PROMPT.md](./doc/PROMPT.md) for writing a prompt file.
