# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import time
import torch
import logging as log
from torch import nn
from typing import Optional, Union
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.utils import ModelOutput
import llm_bench_utils.hook_beam_search as hook_beam


logger = log.getLogger(__name__)


class GenerateBeamDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


class GenerateBeamEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]


ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

# Transformers version: v5.13.0 and later (used for transformers >= 5.13.0)
# Add the function of collecting latency


# https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/utils.py#L3852
def new_prefill(
    self,
    input_ids: torch.LongTensor,
    generation_config: GenerationConfig,
    model_kwargs: dict,
    is_first_iteration: bool = True,
):
    """
    Perform the prefill stage of generation.

    Note that usually, the prefill stage is always the first iteration of a new input batch, and thus multimodal inputs etc
    should be treated as if it's the first iteration. However, for assisted decoding, assistants call `generate`
    several time in a row for a same batch of inputs, so we need to pass `is_first_iteration` here for such cases.
    """
    next_sequence_length = None
    inputs_embeds = model_kwargs.get("inputs_embeds")
    use_inputs_embeds = False
    if not self.config.is_encoder_decoder and inputs_embeds is not None and is_first_iteration:
        use_inputs_embeds = True
    if (cache := model_kwargs.get("past_key_values")) is not None:
        past_length = cache.get_seq_length()
        if use_inputs_embeds:
            next_sequence_length = model_kwargs["inputs_embeds"].shape[1] - past_length
        else:
            attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
            attention_mask = model_kwargs.get(attention_mask_key)
            if attention_mask is not None and input_ids.shape[1] == attention_mask.shape[1]:
                next_sequence_length = input_ids.shape[1] - past_length

    # Usual prefill
    if generation_config.prefill_chunk_size is None:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            is_first_iteration=is_first_iteration,
            **model_kwargs,
        )
        tic_infer = time.perf_counter()
        outputs = self(**model_inputs, return_dict=True)
        hook_beam.tm_infer_list.append(time.perf_counter() - tic_infer)
        return outputs

    # Chunked prefill (for very large contexts)
    else:
        getattr(torch, "_dynamo").config.cache_size_limit = 64

        chunk_size = generation_config.prefill_chunk_size
        input_chunks = torch.split(input_ids, chunk_size, dim=-1)

        if "past_key_values" not in model_kwargs:
            raise ValueError("Cannot use prefill chunking without a cache")

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        attention_mask = model_kwargs.pop("attention_mask", None)
        position_ids = model_kwargs.pop("position_ids", None)
        past_length = 0
        infer_time = 0
        for input_chunk in input_chunks:
            current_length = past_length + input_chunk.shape[-1]
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask[:, :current_length]
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids[:, past_length:current_length]
            model_inputs = self.prepare_inputs_for_generation(input_chunk, **model_kwargs)

            tic_infer = time.perf_counter()
            outputs = model_forward(**model_inputs, return_dict=True)
            infer_time += time.perf_counter() - tic_infer

            model_kwargs["past_key_values"] = outputs.past_key_values
            past_length = current_length

        hook_beam.tm_infer_list.append(infer_time)

        # Recreate the kwargs based on the full length
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["position_ids"] = position_ids

        # Latest outputs contain next token logits
        return outputs


# Copied from https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/utils.py#L3185
def new_beam_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    **model_kwargs,
) -> GenerateBeamOutput | torch.LongTensor:
    # 1. init beam_search values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample
    early_stopping = generation_config.early_stopping
    length_penalty = generation_config.length_penalty
    max_length = generation_config.max_length
    num_beams = generation_config.num_beams
    num_return_sequences = generation_config.num_return_sequences

    batch_size_unflattened, cur_len = input_ids.shape[:2]
    batch_size = batch_size_unflattened // num_beams
    # TODO (joao): standardize special cases
    if self.__class__.__name__ == "MoshiDepthDecoder":
        vocab_size = self.config.audio_vocab_size
    elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
        vocab_size = self.get_output_embeddings().out_features
    elif self.__class__.__name__ == "BarkSemanticModel":
        vocab_size = self.config.output_vocab_size
    else:
        vocab_size = self.config.get_text_config().vocab_size
    decoder_prompt_len = cur_len
    this_peer_finished = False

    n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
    beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
    top_num_beam_mask = torch.cat(
        (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
        dim=0,
    ).to(input_ids.device)

    sequential = generation_config.low_memory
    if sequential:
        raise ValueError(
            "`low_memory=True` is not supported after the beam search refactor. Please check the discussion in "
            "#35802 *after the PR got merged*, and add a comment there if your questions are not yet answered."
        )

    # 2. init output tuples
    all_scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # 3. init running tensors and static-shaped placeholders
    output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
    running_sequences = torch.full(
        (batch_size, num_beams, max_length),
        fill_value=output_fill_value,
        dtype=torch.int64,
        device=input_ids.device,
    )
    running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(input_ids, batch_size, num_beams)
    sequences = running_sequences.detach().clone()

    running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    running_beam_scores[:, 1:] = -1e9
    beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

    is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)

    is_early_stop_heuristic_unsatisfied = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)

    next_token_hits_stopping_criteria = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)

    running_beam_indices = torch.full(
        (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
    )
    beam_indices = running_beam_indices.detach().clone()

    flat_running_sequences = input_ids
    prefill_consumed = False
    tic = time.perf_counter()
    model_outputs = self._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    hook_beam.tm_list.append(time.perf_counter() - tic)

    # 4. run the generation loop
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        tic = time.perf_counter()
        if prefill_consumed:
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = self.prepare_inputs_for_generation(
                flat_running_sequences, next_sequence_length=next_sequence_length, **model_kwargs
            )
            tic_infer = time.perf_counter()
            model_outputs = self(**model_inputs, return_dict=True)
            hook_beam.tm_infer_list.append(time.perf_counter() - tic_infer)
        prefill_consumed = True

        model_kwargs = self._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        logits = model_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs = logits_processor(flat_running_sequences, log_probs)

        if return_dict_in_generate:
            if output_logits:
                raw_logits += (logits.clone(),)
            if return_dict_in_generate and output_scores:
                all_scores += (log_probs.clone(),)

            if output_attentions:
                decoder_attentions += (
                    (model_outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (model_outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (model_outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (model_outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (model_outputs.hidden_states,)
                )

        del model_outputs

        log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
        log_probs = log_probs + running_beam_scores[:, :, None]
        log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

        topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
            accumulated_log_probs=log_probs,
            running_sequences=running_sequences,
            running_beam_indices=running_beam_indices,
            cur_len=cur_len,
            decoder_prompt_len=decoder_prompt_len,
            do_sample=do_sample,
            beams_to_keep=beams_to_keep,
            num_beams=num_beams,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )

        next_token_hits_stopping_criteria = stopping_criteria(
            self._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),
            all_scores,
        )
        next_token_hits_stopping_criteria = self._unflatten_beam_dim(
            next_token_hits_stopping_criteria, batch_size, beams_to_keep
        )

        running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
            topk_log_probs=topk_log_probs,
            topk_running_sequences=topk_running_sequences,
            topk_running_beam_indices=topk_running_beam_indices,
            next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
            num_beams=num_beams,
        )

        sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
            sequences=sequences,
            topk_running_sequences=topk_running_sequences,
            beam_scores=beam_scores,
            topk_log_probs=topk_log_probs,
            beam_indices=beam_indices,
            topk_running_beam_indices=topk_running_beam_indices,
            is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
            is_sent_finished=is_sent_finished,
            next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
            top_num_beam_mask=top_num_beam_mask,
            num_beams=num_beams,
            cur_len=cur_len,
            decoder_prompt_len=decoder_prompt_len,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

        # pluck the cache from the beam indices that will be used in the next iteration
        # NOTE: we need to check if `self._reorder_cache` exists for special models like RAG, RecurrentGemma etc.
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            beam_idx = self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
            if hasattr(self, "_reorder_cache"):
                model_kwargs[cache_key] = self._reorder_cache(model_kwargs[cache_key], beam_idx)
            elif hasattr(model_kwargs[cache_key], "reorder_cache"):
                model_kwargs[cache_key].reorder_cache(beam_idx)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} cannot use beam search with a cache currently, as the cache cannot be reordered"
                )

        hook_beam.tm_list.append(time.perf_counter() - tic)
        cur_len = cur_len + 1
        is_early_stop_heuristic_unsatisfied = self._check_early_stop_heuristic(
            is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
            running_beam_scores=running_beam_scores,
            beam_scores=beam_scores,
            is_sent_finished=is_sent_finished,
            cur_len=cur_len,
            max_length=max_length,
            decoder_prompt_len=decoder_prompt_len,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
        this_peer_finished = not self._beam_search_has_unfinished_sequences(
            is_early_stop_heuristic_unsatisfied,
            is_sent_finished,
            next_token_hits_stopping_criteria,
            early_stopping,
        )

    # 5. prepare outputs
    sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
    beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
    beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

    max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
    output_length = decoder_prompt_len + max_generated_length
    sequences = sequences[:, :output_length]
    beam_indices = beam_indices[:, :max_generated_length]

    if return_dict_in_generate:
        if not output_scores:
            beam_scores = None

        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=all_scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequences,
                sequences_scores=beam_scores,
                scores=all_scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
    else:
        return sequences
