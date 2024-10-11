# Simple Accuracy Benchmark for Generative AI models

## Features

* Simple and quick accuracy test for compressed, quantized, pruned, distilled LLMs. It works with any model that suppors HuggingFace Transformers text generation API including:
    * HuggingFace Transformers compressed models via [Bitsandbytes](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)
    * [GPTQ](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig) via HuggingFace API
    * Llama.cpp via [BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm)
    * [OpenVINO](https://github.com/openvinotoolkit/openvino) and [NNCF](https://github.com/openvinotoolkit/nncf) via [Optimum-Intel](https://github.com/huggingface/optimum-intel)
    * Support of custom datasets of the user choice
* Validation of text-to-image pipelines. Computes similarity score between generated images:
    * Supports Diffusers library and Optimum-Intel via `Text2ImageEvaluator` class.

The main idea is to compare similarity of text generation between baseline and optimized LLMs.

The API provides a way to access to investigate the worst generated text examples.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import whowhatbench

model_id = "facebook/opt-1.3b"
base_small = AutoModelForCausalLM.from_pretrained(model_id)
optimized_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

evaluator = whowhatbench.TextEvaluator(base_model=base_small, tokenizer=tokenizer)
metrics_per_prompt, metrics = evaluator.score(optimized_model)

metric_of_interest = "similarity"
print(metric_of_interest, ": ", metrics["similarity"][0])

worst_examples = evaluator.worst_examples(top_k=5, metric=metric_of_interest)
print("Metric: ", metric_of_interest)
for e in worst_examples:
    print("\t=========================")
    print("\tPrompt: ", e["prompt"])
    print("\tBaseline Model:\n ", "\t" + e["source_model"])
    print("\tOptimized Model:\n ", "\t" + e["optimized_model"])

```

Use your own list of prompts to compare (e.g. from a dataset):
```python
from datasets import load_dataset
val = load_dataset("lambada", split="validation[20:40]")
prompts = val["text"]
...
metrics_per_prompt, metrics = evaluator.score(optimized_model, test_data=prompts)
```

### Installing

* python -m venv eval_env
* source eval_env/bin/activate
* pip install -r requirements.txt

### CLI example for text-generation models

```sh
wwb --help

# Run ground truth generation for uncompressed model on the first 32 samples from squad dataset
# Ground truth will be saved in llama_2_7b_squad_gt.csv file
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_squad_gt.csv --dataset squad --split validation[:32] --dataset-field question

# Run comparison with compressed model on the first 32 samples from squad dataset
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_squad_gt.csv --dataset squad --split validation[:32] --dataset-field question

# Output will be like this
#   similarity        FDT        SDT  FDT norm  SDT norm
# 0    0.972823  67.296296  20.592593  0.735127  0.151505

# Run ground truth generation for uncompressed model on internal set of questions
# Ground truth will be saved in llama_2_7b_squad_gt.csv file
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv

# Run comparison with compressed model on internal set of questions
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_wwb_gt.csv

# Use --num-samples to control the number of samples
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --num-samples 10

# Use -v for verbose mode to see the difference in the results
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_wwb_gt.csv  --num-samples 10 -v

# Use --hf AutoModelForCausalLM to instantiate the model from model_id/folder
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --hf

# Use --language parameter to control the language of promts
# Autodetection works for basic Chinese models 
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --hf
```

### Example of Stable Diffusion comparison
```sh
# Export FP16 model
 optimum-cli export openvino -m SimianLuo/LCM_Dreamshaper_v7 --weight-format fp16 sd-lcm-fp16
# Export INT8 WOQ model
 optimum-cli export openvino -m SimianLuo/LCM_Dreamshaper_v7 --weight-format int8 sd-lcm-int8
# Collect the references
wwb --base-model sd-lcm-fp16 --gt-data lcm_test/sd_xl.json --model-type sd-lcm
# Compute the metric
wwb --target-model sd-lcm-int8 --gt-data lcm_test/sd_xl.json --model-type sd-lcm
```

### Supported metrics

* `similarity` - averaged similarity measured by neural network trained for sentence embeddings. The best is 1.0, the minimum is 0.0, higher-better.
* `FDT` - Average position of the first divergent token betwen sentences generated by differnrt LLMs. The worst is 0, higher-better. [Paper.](https://arxiv.org/abs/2311.01544)
* `FDT norm` - Average share of matched tokens until first divergent one betwen sentences generated by differnrt LLMs. The best is 1, higher-better.[Paper.](https://arxiv.org/abs/2311.01544)
* `SDT` - Average number of divergent tokens in the evaluated outputs betwen sentences generated by differnrt LLMs. The best is 0, lower-better. [Paper.](https://arxiv.org/abs/2311.01544)
* `SDT norm` - Average share of divergent tokens in the evaluated outputs betwen sentences generated by differnrt LLMs. The best is 0, the maximum is 1, lower-better. [Paper.](https://arxiv.org/abs/2311.01544)

### Notes

* The generation of ground truth on uncompressed model must be run before comparison with compressed model.
* WWB uses [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for similarity measurement but you can use other similar network.
