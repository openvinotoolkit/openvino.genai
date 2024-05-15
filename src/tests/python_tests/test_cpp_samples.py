
import pytest

model_ids = [
    # ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0-skip-special-tokens"),

    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/"),
    ("google/gemma-2b-it", "gemma-2b-it/pytorch/dldt/FP16/"),
    # ("meta-llama/Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf/pytorch/dldt/FP16/"),
]

def run_cpp_sample_command(command, cwd):
    import subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

def run_transformers_model(model_id, prompt, config=None, add_special_tokens=True):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    tokenized = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=add_special_tokens)
        
    default_config = dict( 
        num_beam_groups=3, 
        num_beams=15, 
        diversity_penalty=1.0, 
        num_return_sequences=15, 
        max_new_tokens=20, 
        early_stopping=False, 
        length_penalty=1.0, 
        no_repeat_ngram_size=9**9, 
        do_sample=False
    )

    if config is None:
        config = default_config
    print(tokenized)
    beams = model.generate(tokenized, **config)
    return map(lambda beam: tokenizer.decode(beam[tokenized.numel():], skip_special_tokens=True), beams)

@pytest.mark.parametrize("param", model_ids)
def test_model(param):
    model_id, path = param

    prompts = ["table is made of", "The Sun is yellow because"]
    # prompt = " ".join([f'"{item}"' for item in prompts])

    prompt = "table is made of"

    # cmd = 'build-Debug/greedy_causal_lm' // for old samples
    cmd = 'build-Debug/text_generation/causal_lm/cpp/'
    
    # beam search old
    cmd = 'build-Debug/beam_search_causal_lm'
    cwd = '/home/epavel/devel/openvino.genai_'
    config = None # None means greedy

    # greedy new
    cwd = '/home/epavel/devel/openvino.genai'
    cmd = 'build-Debug/text_generation/causal_lm/cpp/greedy_causal_lm'
    config = dict(max_new_tokens=75, do_sample=False)

    # beam search new
    cwd = '/home/epavel/devel/openvino.genai'
    cmd = 'build-Debug/text_generation/causal_lm/cpp/beam_search_causal_lm'
    config = None

    predictions, _, _ = run_cpp_sample_command([cmd, '/home/epavel/devel/openvino.genai/text_generation/causal_lm/' + path, prompt], cwd)
    print(predictions)
    
    beams = run_transformers_model(model_id, prompt, config)
    for beam in beams:
        idx = predictions.find(beam)
        if -1 == idx and beam and predictions:
            raise RuntimeError(f'Missing "{beam=}" from predictions')
        predictions = predictions[:idx] + predictions[idx + len(beam):]
    
    return True
    # with open('pred.txt', 'r') as file:
    #     predictions = file.read()

for model_id, path in model_ids:
    test_model((model_id, path))
