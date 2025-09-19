# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from os.path import sep
from pathlib import Path
from typing import Type
from functools import lru_cache

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig

from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction, OVModelForSequenceClassification
from optimum.intel.openvino.modeling import OVModel

from huggingface_hub import hf_hub_download

from openvino import save_model
from openvino_genai import GenerationResult, GenerationConfig, StopCriteria
from openvino_tokenizers import convert_tokenizer

from utils.constants import get_default_llm_properties, extra_generate_kwargs, get_ov_cache_models_dir
from utils.network import retry_request
import pytest

def generation_config_to_hf(
    default_generation_config : HFGenerationConfig,
    generation_config : GenerationConfig
) -> HFGenerationConfig:
    if generation_config is None:
        return

    kwargs = {}
    kwargs['return_dict_in_generate'] = True

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    # has higher priority than 'max_length'
    SIZE_MAX = 2**64 - 1
    if generation_config.max_new_tokens != SIZE_MAX:
        kwargs['max_new_tokens'] = generation_config.max_new_tokens
    kwargs['min_new_tokens'] = generation_config.min_new_tokens
    if generation_config.stop_strings:
        kwargs['stop_strings'] = generation_config.stop_strings

    # copy default parameters
    kwargs['bos_token_id'] = default_generation_config.bos_token_id
    kwargs['pad_token_id'] = default_generation_config.pad_token_id

    if (generation_config.ignore_eos):
        kwargs['eos_token_id'] = []
    else:
        if len(generation_config.stop_token_ids) > 0:
            kwargs['eos_token_id'] = list(generation_config.stop_token_ids)
        elif generation_config.eos_token_id != -1:
            kwargs['eos_token_id'] = generation_config.eos_token_id
        else:
            kwargs['eos_token_id'] = default_generation_config.eos_token_id

    # copy penalties
    kwargs['repetition_penalty'] = generation_config.repetition_penalty

    if generation_config.is_beam_search():
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_beam_groups
        kwargs['num_beams'] = generation_config.num_beams
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
        kwargs['output_scores'] = True

        if generation_config.num_beam_groups > 1:
            kwargs['diversity_penalty'] = generation_config.diversity_penalty

        # in OpenVINO GenAI this parameter is called stop_criteria,
        # while in HF it's called early_stopping.
        # HF values True, False and "never" correspond to OV GenAI values "EARLY", "HEURISTIC" and "NEVER"
        STOP_CRITERIA_MAP = {
            StopCriteria.NEVER: "never",
            StopCriteria.EARLY: True,
            StopCriteria.HEURISTIC: False
        }

        kwargs['early_stopping'] = STOP_CRITERIA_MAP[generation_config.stop_criteria]
    elif generation_config.is_multinomial():
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
    opt_model,
    hf_tokenizer,
    prompts: list[str],
    generation_configs: list[GenerationConfig] | GenerationConfig,
) -> list[GenerationResult]:
    generation_results = []

    if type(generation_configs) is list:
        # process prompt by promp as we have multiple generation configs
        for prompt, generation_config in zip(prompts, generation_configs):
            hf_generation_config = generation_config_to_hf(opt_model.generation_config, generation_config)
            inputs = {}
            if hf_tokenizer.chat_template and generation_config.apply_chat_template:
                prompt = hf_tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                inputs = hf_tokenizer(prompt, return_tensors="pt")
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            prompt_len = 0 if generation_config.echo else input_ids.numel()

            generate_outputs = opt_model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer, **extra_generate_kwargs())
            all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)

            generation_result = GenerationResult()
            generation_result.m_generation_ids = all_text_batch
            # sequences_scores are available only for beam search case
            if generation_config.is_beam_search():
                generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
            generation_results.append(generation_result)
    else:
        inputs = {}
        if hf_tokenizer.chat_template and generation_configs.apply_chat_template:
            processed_prompts = []
            for prompt in prompts:
                processed_prompts.append(hf_tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True))
            # process all prompts as a single batch as we have a single generation config for all prompts
            inputs = hf_tokenizer(processed_prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False, padding_side='left')
        else:
            inputs = hf_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, padding_side='left')
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        hf_generation_config = generation_config_to_hf(opt_model.generation_config, generation_configs)
        hf_encoded_outputs = opt_model.generate(input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer, **extra_generate_kwargs())

        generation_ids = []
        scores = []

        for idx, hf_encoded_out in enumerate(hf_encoded_outputs.sequences):
            prompt_idx = idx // hf_generation_config.num_return_sequences
            prompt_len = 0 if generation_configs.echo else input_ids[prompt_idx].numel()
            decoded_text = hf_tokenizer.decode(hf_encoded_out[prompt_len:], skip_special_tokens=True)
            generation_ids.append(decoded_text)
            if generation_configs.is_beam_search():
                scores.append(hf_encoded_outputs.sequences_scores[idx])

            # if we need to move to next generation result
            if (idx + 1) // hf_generation_config.num_return_sequences != prompt_idx:
                generation_result = GenerationResult()
                generation_result.m_generation_ids = generation_ids
                generation_result.m_scores = scores
                generation_results.append(generation_result)
                generation_ids = []
                scores = []

    del hf_tokenizer
    del opt_model

    return generation_results


# download HF model or read converted model
def get_huggingface_models(model_id: str | Path, model_class: Type[OVModel], local_files_only=False):
    hf_tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only))
    opt_model = retry_request(lambda: model_class.from_pretrained(model_id, export=isinstance(model_id, str), compile=False, load_in_8bit=False, trust_remote_code=isinstance(model_id, str), ov_config=get_default_llm_properties(), local_files_only=local_files_only))
    return opt_model, hf_tokenizer


def convert_and_save_tokenizer(hf_tokenizer : AutoTokenizer,
                               models_path: Path,
                               **tokenizer_kwargs):
    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True, **tokenizer_kwargs)

    from utils.constants import OV_DETOKENIZER_FILENAME, OV_TOKENIZER_FILENAME
    save_model(tokenizer, models_path / OV_TOKENIZER_FILENAME)
    save_model(detokenizer, models_path / OV_DETOKENIZER_FILENAME)


def convert_models(opt_model : OVModelForCausalLM,
                   hf_tokenizer : AutoTokenizer,
                   models_path: Path,
                   **tokenizer_kwargs):
    opt_model.save_pretrained(models_path)
    # save generation config
    if opt_model.generation_config:
        opt_model.generation_config.save_pretrained(models_path)
    opt_model.config.save_pretrained(models_path)

    # to store tokenizer config jsons with special tokens
    hf_tokenizer.save_pretrained(models_path)
    # convert tokenizers as well
    convert_and_save_tokenizer(hf_tokenizer, models_path)


def download_and_convert_model(model_id: str, **tokenizer_kwargs):
    return _download_and_convert_model(model_id, OVModelForCausalLM, **tokenizer_kwargs)

@pytest.fixture(scope="module")
def download_and_convert_embeddings_models(request):
    model_id = request.param
    return _download_and_convert_model(model_id, OVModelForFeatureExtraction)


@pytest.fixture()
def download_and_convert_rerank_model(request):
    model_id = request.param
    return _download_and_convert_model(model_id, OVModelForSequenceClassification)


def _download_and_convert_model(model_id: str, model_class: Type[OVModel], **tokenizer_kwargs):
    dir_name = str(model_id).replace(sep, "_")
    ov_cache_models_dir = get_ov_cache_models_dir()
    models_path = ov_cache_models_dir / dir_name

    from utils.constants import OV_MODEL_FILENAME
    if (models_path / OV_MODEL_FILENAME).exists():
        opt_model, hf_tokenizer = get_huggingface_models(models_path, model_class, local_files_only=True)
    else:
        opt_model, hf_tokenizer = get_huggingface_models(model_id, model_class, local_files_only=False)
        convert_models(opt_model, hf_tokenizer, models_path)

    if "padding_side" in tokenizer_kwargs:
        hf_tokenizer.padding_side = tokenizer_kwargs.pop("padding_side")

    return opt_model, hf_tokenizer, models_path


def download_gguf_model(gguf_model_id: str,
                        gguf_filename: str):
    gguf_dir_name = str(gguf_model_id).replace(sep, "_")
    ov_cache_models_dir = get_ov_cache_models_dir()
    models_path_gguf = ov_cache_models_dir / gguf_dir_name

    gguf_path = hf_hub_download(
        repo_id=gguf_model_id,
        filename=gguf_filename,
        local_dir=models_path_gguf # Optional: Specify download directory
    )

    return gguf_path

@lru_cache(maxsize=None)
def load_hf_model_from_gguf(gguf_model_id, gguf_filename):
    return retry_request(lambda: AutoModelForCausalLM.from_pretrained(gguf_model_id, gguf_file=gguf_filename))

@lru_cache(maxsize=None)
def load_hf_tokenizer_from_gguf(gguf_model_id, gguf_filename):
    return retry_request(lambda: AutoTokenizer.from_pretrained(gguf_model_id, gguf_file=gguf_filename))
