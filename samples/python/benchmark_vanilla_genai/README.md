# Benchmark Vanilla GenAI

This sample script demonstrates how to benchmark an LLMModel in OpenVINO GenAI. The script includes functionality for warm-up iterations, generating text, and calculating various performance metrics.

# ov.genai.PerfMetrics structure
ov.genai.PerfMetrics is a structure which holds performance metric for each generate call. Each generate call calcualtes the following metrics:
- mean_ttft
 - std_ttft
 - mean_tpot
 - std_tpot
 - load_time
 - mean_generate_duration
 - std_generate_duration
 - mean_tokenization_duration
 - std_tokenization_duration
 - mean_detokenization_duration
 - std_detokenization_duration
 - mean_throughput
 - std_throughput
 - num_generated_tokens
 - num_input_tokens

Performance metrics can be added to one another and accumulated using the += operator or the + operator. In that case the mean values accumulated by several generate calls will be calculated.


## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../requirements.txt](../../requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
```

## Usage

```sh
python benchmark_vanilla_genai.py [OPTIONS]
```

### Options

- `-m, --model`: Path to the model and tokenizers base directory.
- `-p, --prompt` (default: `"The Sky is blue because"`): The prompt to generate text.
- `-nw, --num_warmup` (default: `1`): Number of warmup iterations.
- `-mt, --max_new_tokens` (default: `20`): Number of warmup iterations.
- `-n, --num_iter` (default: `3`): Number of iterations.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### Output:

```
python benchmark_vanilla_genai.py -m TinyLlama-1.1B-Chat-v1.0/
```

```
Load time: 3405.69 ms
Generate time: 1430.77 ± 3.04 ms
Tokenization time: 0.51 ± 0.02 ms
Detokenization time: 0.37 ± 0.01 ms
TTFT: 81.60 ± 0.54 ms
TPOT: 71.52 ± 2.72 ms
Throughput tokens/s: 13.98 ± 0.53
```
s