# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
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
from transformers.generation.utils import (
    _split_model_inputs,
    stack_model_outputs,
)
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


# Transformers version: v4.51.3
# Copied from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/utils.py#L3751
# Add the function of collecting latency
def new_beam_search_v51(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        If it's the first time you're diving into Beam Search, we recommend you read the following blog post:
        https://huggingface.co/blog/how-to-generate (especially the beam search section).

        You can recompute the sequence scores from the individual scores using the `compute_transition_scores` function
        (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores)

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size*num_beams, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

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

        batch_size_unflattened, cur_len = input_ids.shape
        batch_size = batch_size_unflattened // num_beams

        if self.__class__.__name__ == "MoshiDepthDecoder":
            vocab_size = self.config.audio_vocab_size
        elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
            vocab_size = self.get_output_embeddings().out_features
        else:
            vocab_size = self.config.get_text_config().vocab_size
        decoder_prompt_len = cur_len
        this_peer_finished = False

        # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
        # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
        # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
        # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0,
        ).to(input_ids.device)

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # (joao) feature lost in the refactor. Probably won't implement, hurts readbility with minimal gains (there
        # are newer low-memory alternatives like the offloaded cache)
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

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # 3. init running tensors and static-shaped placeholders

        # per batch, beam-item holding current token in loop and completed sequences
        output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
        running_sequences = torch.full(
            (batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=torch.int64,
            device=input_ids.device,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(input_ids, batch_size, num_beams)
        sequences = running_sequences.detach().clone()

        # per batch, beam-item score, logprobs
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

        # per batch, beam-item state bit indicating if sentence has finished.
        is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)

        # per batch, beam-item state bit indicating if there are valid continuations.
        next_token_hits_stopping_criteria = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
        )

        # per batch selected beam indices
        running_beam_indices = torch.full(
            (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
        )
        beam_indices = running_beam_indices.detach().clone()

        # 4. run the generation loop
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            tic = time.perf_counter()
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            tic_infer = time.perf_counter()
            model_outputs = self(**model_inputs, return_dict=True)
            hook_beam.tm_infer_list.append(time.perf_counter() - tic_infer)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref
            logits = model_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = logits_processor(flat_running_sequences, log_probs)

            # Store logits, attentions and hidden_states when required
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

            # This is needed to properly delete logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del model_outputs

            log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
            # continuations among all beams based on the accumulated scores.
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

            # d. Check which running sequences have finished
            next_token_hits_stopping_criteria = stopping_criteria(
                self._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),  # remove unfilled token indexes
                all_scores,
            )
            next_token_hits_stopping_criteria = self._unflatten_beam_dim(
                next_token_hits_stopping_criteria, batch_size, beams_to_keep
            )

            # e. Get the non-finished running `num_beams` sequences for the next generation step
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # f. Update the completed beams if a new high score in a finished sequence is found
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

            # g. Prepare remaining data for the next iteration, including computing the stopping condition for
            # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)

            # pluck the cache from the beam indices that will be used in the next iteration
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    past_key_values=model_kwargs["past_key_values"],
                    beam_idx=self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len]),
                )

            hook_beam.tm_list.append(time.perf_counter() - tic)
            cur_len = cur_len + 1
            this_peer_finished = not self._beam_search_has_unfinished_sequences(
                running_beam_scores,
                beam_scores,
                is_sent_finished,
                next_token_hits_stopping_criteria,
                cur_len,
                max_length,
                decoder_prompt_len,
                early_stopping,
                length_penalty,
            )

        # 5. prepare outputs
        # Take best beams for each batch (the score is sorted in descending order)
        sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
        beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        # Crop the static-shaped tensors to the actual size.
        # `beam_indices` is initialized with -1s, and is updated with the beam index of the generated token at each
        # step. We can use it to detect the generated length, which may be != `cur_len`  (e.g. selected beam is from a
        # previous decoding iteration)
        max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
        output_length = decoder_prompt_len + max_generated_length
        sequences = sequences[:, :output_length]
        beam_indices = beam_indices[:, :max_generated_length]

        if return_dict_in_generate:
            if not output_scores:
                beam_scores = None

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
                    past_key_values=model_kwargs.get("past_key_values"),
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
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequences