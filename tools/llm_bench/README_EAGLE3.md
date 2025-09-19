# SPEULATIVE DECODING for EAGLE3

### 1. Prepare Python Virtual Environment for LLM Benchmarking
   
``` bash
python3 -m venv ov-llm-bench-env
source ov-llm-bench-env/bin/activate
pip install --upgrade pip

git clone  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt  
```

### 2. Get main and draft model in OpenVINO IR Format
the main and draft model downloaded from hugging face needs to be converted to openvino IR format.
For now, please get llama3 8B eagle3 main and draft model from below server (password: openvino):
``` bash
scp -r openvino-ci-97@10.67.108.171:~/bell/speculative_decoding/eagle3/llama-3.1-8b-instruct-ov-int4/ your_path_to_main/
scp -r openvino-ci-97@10.67.108.171:~/bell/speculative_decoding/eagle3/EAGLE3-LLaMA3.1-instruct-8B-ov-int4/ your_path_to_draft/
```

### 3. Benchmark LLM Model using eagle3 speculative decoding

To benchmark the performance of the LLM, use the following command:

python benchmark.py -m /home/openvino-ci-97/bell/speculative_decoding/eagle3/llama-3.1-8b-instruct-ov-int4 -d GPU -pf /home/openvino-ci-97/bell/openvino.genai/tools/llm_bench/test.jsonl -ic 129 --draft_model /home/openvino-ci-97/bell/speculative_decoding/eagle3/EAGLE3-LLaMA3.1-instruct-8B-ov-int4 --draft_device GPU --eagle_config ./eagle.config --disable_prompt_permutation --apply_chat_template

the content of eagle.config is as below:
{"eagle_mode":"EAGLE3", "branching_factor": 1, "tree_depth": 4, "total_tokens": 6}

to tune for better performance, fix the branching_factor to 1, adjust the tree_depth, and the total_tokens should be set as tree_depth + 2 for now (for example, in above config, ajust tree_depth to 5, and total_tokens set to 7)
