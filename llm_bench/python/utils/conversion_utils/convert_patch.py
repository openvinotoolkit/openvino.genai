# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import types
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import PreTrainedModel


@torch.jit.script_if_tracing
def _chatglm2_get_context_layer(
    query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor
):
    mask = torch.zeros(
        (query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype
    )
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones(
            (query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool
        ).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer, key_layer, value_layer, attn_mask=mask
    )
    return context_layer


def _core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [
        k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]
    ]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (
        self.hidden_size_per_partition,
    )
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


@torch.jit.script_if_tracing
def _get_chatglm_attention_mask(input_ids, past_key):
    mask = torch.zeros(
        (input_ids.shape[1], past_key.shape[0] + input_ids.shape[1]),
        dtype=past_key.dtype,
    )
    if past_key.shape[0] == 0:
        tmp_mask = torch.ones(
            (input_ids.shape[1], past_key.shape[0] + input_ids.shape[1]),
            dtype=torch.bool,
        ).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))
    return mask


def _chatglm_transformer_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(
                batch_size=batch_size,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.new_ones((batch_size, self.pre_seq_len)),
                    attention_mask,
                ],
                dim=-1,
            )

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (
            past_key_values and seq_length != 1
        ):
            full_attention_mask = self.get_masks(
                input_ids, past_key_values, padding_mask=attention_mask
            )
        elif past_key_values is not None:
            full_attention_mask = torch.ones(
                batch_size,
                seq_length,
                seq_length,
                device=input_ids.device,
                dtype=torch.float,
            ) * float("-inf")
            full_attention_mask.triu_(diagonal=1)
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0][0].shape[0]
            if past_length:
                full_attention_mask = torch.cat(
                    (
                        torch.zeros(
                            batch_size, seq_length, past_length, device=input_ids.device
                        ),
                        full_attention_mask,
                    ),
                    dim=-1,
                )
            full_attention_mask.unsqueeze_(1)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds,
        full_attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
    )

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _patch_chatglm_core_attention_forward(model: "PreTrainedModel"):
    model.transformer.forward = types.MethodType(
        _chatglm_transformer_forward, model.transformer
    )
    for block in model.transformer.encoder.layers:
        block.self_attention.core_attention.forward = types.MethodType(
            _core_attention_forward, block.self_attention.core_attention
        )


def _update_qwen_rotary_embedding_cache(model):
    model.transformer.rotary_emb(2048)


def _yi_prepare_decoder_attention_mask(
    attention_mask, input_ids, inputs_embeds, past_key_values_length
):
    input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
    return _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )


# Modified from transformers.models.mistral.modeling_mistral._prepare_decoder_sliding_window_attention_mask
def _prepare_decoder_sliding_window_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int,
):
    from transformers.models.mistral.modeling_mistral import (
        _expand_mask,
        _make_sliding_window_causal_mask,
    )

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_sliding_window_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
        past_key_values_length=past_key_values_length,
        sliding_window=sliding_window,
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


def mixtral_sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


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
    if model.config.model_type in ["stablelm_epoch", "baichuan"]:
        model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        if model.config.model_type == "stablelm_epoch":
            model.model.forward = types.MethodType(stablelm_forward, model.model)
    elif model.config.model_type == "chatglm":
        _patch_chatglm_core_attention_forward(model)
    elif model.config.model_type == "qwen":
        _update_qwen_rotary_embedding_cache(model)
    elif model.config.model_type == "mistral":
        model.model._prepare_decoder_attention_mask = (
            _prepare_decoder_sliding_window_attention_mask
        )
    elif model.config.model_type == "Yi":
        model.model._prepare_decoder_attention_mask = _yi_prepare_decoder_attention_mask
    elif model.config.model_type == "mixtral":
        for layer in model.model.layers:
            layer.block_sparse_moe.forward = types.MethodType(mixtral_sparse_moe_block_forward, layer.block_sparse_moe)
    elif model.config.model_type == "gemma":
        for layer in model.model.layers:
            if layer.self_attn.rotary_emb.inv_freq is None:
                rotary_emb = layer.self_attn.rotary_emb
                layer.self_attn.rotary_emb.inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim))
    return model
