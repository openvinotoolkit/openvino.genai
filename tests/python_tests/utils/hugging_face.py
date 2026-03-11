# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from optimum.modeling_base import OptimizedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig

from optimum.intel import OVModelForCausalLM, OVModelForSequenceClassification
from optimum.intel.openvino.modeling import OVModel

from huggingface_hub import hf_hub_download, snapshot_download

from openvino import save_model
from openvino_genai import GenerationResult, GenerationConfig, StopCriteria
from openvino_tokenizers import convert_tokenizer

from utils.constants import (
    get_default_llm_properties,
    extra_generate_kwargs,
    get_ov_cache_converted_models_dir,
    get_ov_cache_downloaded_models_dir,
)
from utils.network import retry_request
from utils.atomic_download import AtomicDownloadManager

from utils.constants import OV_MODEL_FILENAME


@dataclass(frozen=True)
class OVConvertedModelSchema:
    model_id: str
    opt_model: OptimizedModel
    hf_tokenizer: AutoTokenizer
    models_path: Path


def generation_config_to_hf(
    default_generation_config: HFGenerationConfig,
    generation_config: GenerationConfig | None,
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
    opt_model: OptimizedModel,
    hf_tokenizer: AutoTokenizer,
    prompts: list[str],
    generation_configs: list[GenerationConfig] | GenerationConfig,
) -> list[GenerationResult]:
    generation_results: list[GenerationResult] = []

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

    return generation_results


# download HF model or read converted model
def get_huggingface_models(
    model_id: str | Path,
    model_class: Type[OVModel],
    local_files_only=False,
    trust_remote_code=False,
) -> tuple[OptimizedModel, AutoTokenizer]:
    if not local_files_only and isinstance(model_id, str):
        model_id = snapshot_download(model_id)  # required to avoid HF rate limits

    def auto_tokenizer_from_pretrained() -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            model_id, 
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    is_eagle_model = "eagle3" in str(model_id).lower()

    def auto_model_from_pretrained() -> OptimizedModel:
        params = {
            "export": isinstance(model_id, str),
            "compile": False,
            "load_in_8bit": False,
            "ov_config": get_default_llm_properties(),
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        return model_class.from_pretrained(model_id, **params)

    opt_model = retry_request(auto_model_from_pretrained)

    if is_eagle_model:
        return opt_model, None
    else:
        hf_tokenizer = retry_request(auto_tokenizer_from_pretrained)
        return opt_model, hf_tokenizer

def convert_and_save_tokenizer(
    hf_tokenizer : AutoTokenizer,
    models_path: Path,
    **convert_args,
):
    tokenizer, detokenizer = convert_tokenizer(
        hf_tokenizer, with_detokenizer=True, **convert_args
    )

    from utils.constants import OV_DETOKENIZER_FILENAME, OV_TOKENIZER_FILENAME
    save_model(tokenizer, models_path / OV_TOKENIZER_FILENAME)
    save_model(detokenizer, models_path / OV_DETOKENIZER_FILENAME)


def convert_models(
    opt_model : OVModelForCausalLM,
    hf_tokenizer : AutoTokenizer,
    models_path: Path,
) -> None:
    opt_model.save_pretrained(str(models_path))
    # save generation config
    if opt_model.generation_config:
        opt_model.generation_config.save_pretrained(str(models_path))
    opt_model.config.save_pretrained(str(models_path))

    # to store tokenizer config jsons with special tokens
    if hf_tokenizer:
        hf_tokenizer.save_pretrained(str(models_path))
        # convert tokenizers as well
        convert_and_save_tokenizer(hf_tokenizer, models_path)


def download_and_convert_model(model_id: str, **tokenizer_kwargs) -> OVConvertedModelSchema:
    return download_and_convert_model_class(model_id, OVModelForCausalLM, **tokenizer_kwargs)


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


TRUST_REMOTE_CODE_MODELS = ("AngelSlim/Qwen3-1.7B_eagle3",)


def download_and_convert_model_class(
    model_id: str, 
    model_class: Type[OVModel],
    **tokenizer_kwargs,
) -> OVConvertedModelSchema:
    trust_remote_code = model_id in TRUST_REMOTE_CODE_MODELS
    dir_name = sanitize_model_id(model_id)
    if model_class.__name__ not in ["OVModelForCausalLM"]:
        dir_name = f"{dir_name}_{model_class.__name__}"
    ov_cache_converted_dir = get_ov_cache_converted_models_dir()
    models_path = ov_cache_converted_dir / dir_name

    manager = AtomicDownloadManager(models_path)

    if manager.is_complete() or (models_path / OV_MODEL_FILENAME).exists():
        opt_model, hf_tokenizer = get_huggingface_models(
            models_path, model_class, local_files_only=True, trust_remote_code=trust_remote_code
        )
    else:
        opt_model, hf_tokenizer = get_huggingface_models(
            model_id, model_class, local_files_only=False, trust_remote_code=trust_remote_code
        )
        if "padding_side" in tokenizer_kwargs:
            hf_tokenizer.padding_side = tokenizer_kwargs.pop("padding_side")

        def convert_to_temp(temp_path: Path) -> None:
            convert_models(opt_model, hf_tokenizer, temp_path)

        manager.execute(convert_to_temp)

    if "padding_side" in tokenizer_kwargs:
        hf_tokenizer.padding_side = tokenizer_kwargs["padding_side"]

    return OVConvertedModelSchema(
        model_id, 
        opt_model, 
        hf_tokenizer, 
        models_path,
    )


def download_gguf_model(
    gguf_model_id: str,
    gguf_filename: str,
):
    gguf_dir_name = sanitize_model_id(gguf_model_id)
    ov_cache_downloaded_dir = get_ov_cache_downloaded_models_dir()
    models_path_gguf = ov_cache_downloaded_dir / gguf_dir_name

    manager = AtomicDownloadManager(models_path_gguf)

    def download_to_temp(temp_path: Path) -> None:
        retry_request(
            lambda: hf_hub_download(
                repo_id=gguf_model_id,
                filename=gguf_filename,
                local_dir=temp_path
            )
        )

    manager.execute(download_to_temp)

    gguf_path = models_path_gguf / gguf_filename
    return gguf_path


def load_hf_model_from_gguf(gguf_model_id, gguf_filename):
    model_cached = snapshot_download(gguf_model_id)  # required to avoid HF rate limits
    return retry_request(lambda: AutoModelForCausalLM.from_pretrained(model_cached, gguf_file=gguf_filename))


def load_hf_tokenizer_from_gguf(gguf_model_id, gguf_filename):
    model_cached = snapshot_download(gguf_model_id)  # required to avoid HF rate limits
    return retry_request(lambda: AutoTokenizer.from_pretrained(model_cached, gguf_file=gguf_filename))
