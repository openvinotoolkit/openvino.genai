from typing import List, Tuple
from unittest import TestCase

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig

from py_continuous_batching import ContinuousBatchingPipeline, GenerationConfig, SchedulerConfig, GenerationResult

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_groups = 3
    generation_config.group_size = 2
    generation_config.num_return_sequences = 6
    return generation_config

def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        # "What is OpenVINO?",
        # "How are you?",
        # "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_beam_search(),
        # get_beam_search(),
        # get_beam_search(),
        # get_beam_search()
    ]
    return (prompts, generation_configs)

def get_scheduler_config() -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.num_kv_blocks = 300

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
        kwargs['output_scores'] = True
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
) -> List[GenerationResult]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    generation_results: List[GenerationResult] = []

    for prompt, generation_config in zip(prompts, generation_configs):
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_outputs = model.generate(**inputs, generation_config=convert_to_hf(generation_config), return_dict_in_generate=True)
        all_text_batch = tokenizer.batch_decode(generate_outputs.sequences, skip_special_tokens=True)

        generation_result = GenerationResult()
        generation_result.m_generation_ids = [text[len(prompt):] for text in all_text_batch]
        generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
        generation_results.append(generation_result)

    return generation_results

def run_continuous_batching(
    model_path : str,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig]
) -> List[GenerationResult]:
    pipe = ContinuousBatchingPipeline(model_path, scheduler_config)
    return pipe.generate(prompts, generation_configs)

def test_check_greedy_search():
    prompts, generation_configs = get_test_dataset()
    hf_results : List[GenerationResult] = run_hugging_face("facebook/opt-125m", prompts, generation_configs)
    my_results : List[GenerationResult] = run_continuous_batching("/home/sandye51/Documents/Programming/git_repo/openvino.genai/build/opt125", get_scheduler_config(), prompts, generation_configs)

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(my_results)

    for prompt, hf_result, my_result in zip(prompts, hf_results, my_results):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nmy result = {my_result}")
        assert len(hf_result.m_generation_ids) == len(my_result.m_generation_ids)
        assert len(hf_result.m_scores) == len(my_result.m_scores)
        # for each num_return_sequences
        for hf_text, hf_score, my_text, my_score in zip(hf_result.m_generation_ids, hf_result.m_scores, my_result.m_generation_ids, my_result.m_scores):
            assert hf_text == my_text
            # assert hf_score == my_score
