import openvino_genai as ov_genai
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
path = '/home/epavel/devel/openvino.genai/text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0'
device = 'CPU'
pipe = ov_genai.LLMPipeline(path, device)

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = 'table is made of'
generation_config = {'max_new_tokens': 10}

encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
hf_encoded_output = model.generate(encoded_prompt, **generation_config)
hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])



import os
build_dir = os.getenv('GENAI_BUILD_DIR', 'build')
ov_tokenizers_path = f'{build_dir}/openvino_tokenizers/src/'
# pipe = ov_genai.LLMPipeline(path, device, {}, ov_tokenizers_path)

ov_output = pipe.generate(prompt, **generation_config)
