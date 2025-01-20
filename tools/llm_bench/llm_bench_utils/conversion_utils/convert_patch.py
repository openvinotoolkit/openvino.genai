# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Tuple, Optional, Union
import types
from transformers.modeling_outputs import BaseModelOutputWithPast


def _yi_prepare_decoder_attention_mask(
    attention_mask, input_ids, inputs_embeds, past_key_values_length
):
    input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
    return _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# Modified from transformers.models.bloom.modeling_bloom._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int,
    dtype: torch.dtype = torch.bool,
) -> torch.BoolTensor:
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.zeros(
        (target_length, target_length + past_key_values_length),
        dtype=dtype,
        device=device,
    )
    seq_ids = torch.arange(target_length, device=device)

    mask[:, past_key_values_length:] = (
        (seq_ids[:, None] < seq_ids[None, :]) * torch.finfo(dtype).min
        if torch.is_floating_point(mask)
        else seq_ids[:, None] < seq_ids[None, :]
    )

    return mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )


# Modified from transformers.models.llama.modeling_llama._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
        dtype=inputs_embeds.dtype,
    )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def stablelm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # Retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # Embed positions
    if getattr(self, "_use_flash_attention_2", False):
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    # Decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # Add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def patch_model_for_optimum_export(model):
    if model.config.model_type in ["stablelm_epoch"]:
        model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        model.model.forward = types.MethodType(stablelm_forward, model.model)
    elif model.config.model_type == "Yi":
        model.model._prepare_decoder_attention_mask = _yi_prepare_decoder_attention_mask
    return model
