# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import logging as log
import time
import torch
from torch import nn
from typing import Optional, Union
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.streamers import BaseStreamer
from transformers.utils import ModelOutput
from transformers.generation.configuration_utils import GenerationConfig
import llm_bench_utils.hook_greedy_search as hook_greedy


logger = log.getLogger(__name__)


class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]


ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


# Transformers version: v5.4.0 through v5.10.x (used for transformers >= 5.4.0, < 5.11.0)
# Add the function of collecting latency


# https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/generation/utils.py#L3703
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
        hook_greedy.tm_infer_list.append(time.perf_counter() - tic_infer)
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

        hook_greedy.tm_infer_list.append(infer_time)

        # Recreate the kwargs based on the full length
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["position_ids"] = position_ids

        # Latest outputs contain next token logits
        return outputs


# Copied from https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/generation/utils.py#L2636
def new_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> GenerateNonBeamOutput | torch.LongTensor:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else self.__call__
    )

    prefill_consumed = False
    tic = time.perf_counter()
    outputs = self._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    hook_greedy.tm_list.append(time.perf_counter() - tic)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        tic = time.perf_counter()
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, next_sequence_length=next_sequence_length, **model_kwargs
            )
            with self._optimize_model_for_decode():
                tic_infer = time.perf_counter()
                outputs = model_forward(**model_inputs, return_dict=True)
                hook_greedy.tm_infer_list.append(time.perf_counter() - tic_infer)
        prefill_consumed = True
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                )

        # token selection
        tic_sample = time.perf_counter()
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        hook_greedy.tm_sample_list.append(time.perf_counter() - tic_sample)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        hook_greedy.tm_list.append(time.perf_counter() - tic)

        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
    else:
        return input_ids
