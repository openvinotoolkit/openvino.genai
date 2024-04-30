from typing import List, Tuple
from unittest import TestCase

from transformers import AutoTokenizer
from transformers import GenerationConfig as HFGenerationConfig
from optimum.intel import OVModelForCausalLM
import py_continuous_batching as pa
from py_continuous_batching import GenerationConfig, SchedulerConfig

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_groups = 3
    generation_config.group_size = 2
    return generation_config

def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_beam_search(),
        get_beam_search(),
        get_beam_search(),
        get_beam_search()
    ]
    return (prompts, generation_configs)

def get_scheduler_config() -> SchedulerConfig:
    scheduler_config = pa.SchedulerConfig()
    scheduler_config.dynamic_split_fuse = True

    return scheduler_config

def convert_to_hf(
    generation_config : GenerationConfig
) -> HFGenerationConfig:
    kwargs = {}

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    kwargs['max_new_tokens'] = generation_config.max_new_tokens

    if generation_config.num_groups * generation_config.group_size > 1:
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_groups
        kwargs['num_beams'] = generation_config.num_groups * generation_config.group_size
        kwargs['diversity_penalty'] = generation_config.diversity_penalty
        kwargs['repetition_penalty'] = generation_config.repetition_penalty
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
    elif generation_config.do_sample:
        # mulitinomial
        kwargs['temperature'] = generation_config.temperature
        kwargs['top_k'] = generation_config.top_k
        kwargs['top_p'] = generation_config.top_p
        kwargs['do_sample'] = generation_config.do_sample
    else:
        # greedy
        pass

    hf_generation_config = HFGenerationConfig(**kwargs)
    return hf_generation_config

def run_hugging_face(
    model_id : str,
    prompts: List[str],
    generation_configs: List[GenerationConfig]
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)
    generated_results: List[str] = []

    for prompt, generation_config in zip(prompts, generation_configs):
        inputs = tokenizer(prompt, return_tensors="pt")
        output_tokens = model.generate(**inputs, generation_config=convert_to_hf(generation_config))
        all_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        generated_text = all_text[len(prompt):-1]
        generated_results.append(generated_text)

    return generated_results

def run_continuous_batching(
    model_path : str,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig]
) -> List[str]:
    scheduler_config = pa.SchedulerConfig()
    scheduler_config.dynamic_split_fuse = True

    pipe = pa.ContinuousBatchingPipeline(model_path, scheduler_config)
    outputs = pipe.generate(prompts, generation_configs)

    generated_results: List[str] = []
    for output in outputs:
        # suppose that 0-th has maximum score
        generated_results.append(output.m_generation_ids[4])

    return generated_results

def test_check_greedy_search():
    prompts, generation_configs = get_test_dataset()
    hf_results = run_hugging_face("facebook/opt-125m", prompts, generation_configs)
    my_results = run_continuous_batching("/home/sandye51/Documents/Programming/git_repo/vllm", get_scheduler_config(), prompts, generation_configs)
    for prompt, hf_result, my_result in zip(prompts, hf_results, my_results):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nmy result = {my_result}")
        assert hf_result == my_result
