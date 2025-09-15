# OpenVINO™ GenAI Speculative Decoding using EAGLE3

paper reference: https://github.com/SafeAILab/EAGLE

## Getting Started

To run openvino inference in eagle3 mode, check the following check-points:
 - eagle3 draft model from hugging-face. The hugging face model cannot be used directly in ov. Some modifications are needed locally. If needed, check https://jira.devtools.intel.com/browse/CVS-171947 for model conversion reference. Currently, check with us to get modified llama3-8B model openvino model directly.
 - ov package:
 - genai package:


## Supported eagle3 parameters:
 - the verfied parameter sets: {"eagle_mode":"EAGLE3", "branching_factor": 1, "tree_depth": 4, "total_tokens": 6}
 - you can use c++ reference app for test (our current test data are tested in this method):
   ./eagle_speculative_lm /home/openvino-ci-97/bell/speculative_decoding/eagle3/llama-3.1-8b-instruct-ov-int4 /home/openvino-ci-97/bell/speculative_decoding/eagle3/EAGLE3-LLaMA3.1-instruct-8B-ov-int4/ "your_prompt_here"
- python test command as an option:
    python benchmark.py -m /home/openvino-ci-97/bell/speculative_decoding/eagle3/llama-3.1-8b-instruct-ov-int4 -d GPU -pf /home/openvino-ci-97/bell/openvino.genai/tools/llm_bench/test.jsonl -ic 129 --draft_model /home/openvino-ci-97/bell/speculative_decoding/eagle3/EAGLE3-LLaMA3.1-instruct-8B-ov-int4 --draft_device GPU --eagle_config ./eagle.config --disable_prompt_permutation --apply_chat_template
    sample content of eagle.config :
    {"eagle_mode":"EAGLE3", "branching_factor": 1, "tree_depth": 4, "total_tokens": 6}
    
