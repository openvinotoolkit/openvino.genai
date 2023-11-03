
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch

import copy

import torch
from torch import nn

import tqdm
import itertools
import string

from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

@torch.inference_mode()
def generate(self, inputs=None, **kwargs):
    r"""
    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
    """
    # generation_config (`~generation.GenerationConfig`, *optional*):
    #         The generation configuration to be used as base parametrization for the generation call. `**kwargs`
    #         passed to generate matching the attributes of `generation_config` will override them. If
    #         `generation_config` is not provided, the default will be used, which had the following loading
    #         priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
    #         configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
    #         default values, whose documentation should be checked to parameterize generation.
    generation_config = copy.deepcopy(self.generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    # logits_processor (`LogitsProcessorList`, *optional*):
    #         Custom logits processors that complement the default logits processors built from arguments and
    #         generation config. If a logit processor is passed that is already created with the arguments or a
    #         generation config an error is thrown. This feature is intended for advanced users.
    logits_processor = LogitsProcessorList()
    # stopping_criteria (`StoppingCriteriaList`, *optional*):
    #         Custom stopping criteria that complement the default stopping criteria built from arguments and a
    #         generation config. If a stopping criteria is passed that is already created with the arguments or a
    #         generation config an error is thrown. This feature is intended for advanced users.
    stopping_criteria = StoppingCriteriaList()

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    model_kwargs["use_cache"] = generation_config.use_cache

    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 8. prepare distribution pre_processing samplers
    prefix_allowed_tokens_fn = None
    negative_prompt_ids = None
    negative_prompt_attention_mask = None
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=generation_config.num_beams,
        device=inputs_tensor.device,
        length_penalty=generation_config.length_penalty,
        do_early_stopping=generation_config.early_stopping,
        num_beam_hyps_to_keep=generation_config.num_return_sequences,
        num_beam_groups=generation_config.num_beam_groups,
        max_length=generation_config.max_length,
    )
    # 12. interleave input_ids with `num_beams` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_beams,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )
    # 13. run beam search
    max_length = None  # deprecated
    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    output_attentions = False
    output_scores=generation_config.output_scores
    return_dict_in_generate=generation_config.return_dict_in_generate
    output_hidden_states = False
    """
    Generates sequences of token ids for models with a language modeling head using **diverse beam search
    decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.group_beam_search`] directly. Use
    generate() instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

        model_kwargs:
            Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if [`~generation.BeamSearchDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
        [`~generation.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     HammingDiversityLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run diverse beam search using 6 beams
    >>> num_beams = 6
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     max_length=model.config.max_length,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ...     num_beam_groups=3,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.group_beam_search(
    ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    # init values
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
    device = input_ids.device

    batch_beam_size, cur_len = input_ids.shape

    if return_dict_in_generate and output_scores:
        beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
    else:
        beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
    # the same group don't produce same tokens everytime.
    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    for _ in tqdm.tqdm(range(2044)):
        # predicted tokens in cur_len step
        current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

        # do one decoder step on all beams of all sentences in batch
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if output_scores:
            processed_score = torch.zeros_like(outputs.logits[:, -1, :])

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of current group only
            next_token_logits = outputs.logits[batch_group_indices, -1, :]

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * group_size, vocab_size)
            vocab_size = next_token_scores.shape[-1]

            next_token_scores_processed = logits_processor(
                group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
            )
            next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            if output_scores:
                processed_score[batch_group_indices] = next_token_scores_processed

            # reshape for beam search
            next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * group_size, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
                group_index=beam_group_idx,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if return_dict_in_generate and output_scores:
                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                )

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            current_tokens[batch_group_indices] = group_input_ids[:, -1]

            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                + group_start_idx
                + (beam_idx % group_size)
            )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (processed_score,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(
                model_kwargs["past_key_values"], reordering_indices
            )

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            raise "Break"
            break

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
    )
    return sequence_outputs["sequences"]

def main():
    model_path = r'C:\Users\vzlobin\r\tiny-llama-fast-tokenizer'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # add the EOS token as PAD token to avoid warnings
    model = LlamaForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model.generate = generate.__get__(model, transformers.GenerationMixin)

    for repeat in range(1, 9**9):
        for prod in itertools.product(string.printable + string.whitespace, repeat=repeat):
            print(f'{prod = }')
            tokens = tokenizer(prod, return_tensors='pt')
            model.generate(**tokens, max_new_tokens=9**9, num_beam_groups=2, num_beams=4, do_sample=False, early_stopping=True, no_repeat_ngram_size=2, num_return_sequences=4, top_k=50, diversity_penalty=1.0)
    # encode context the generation is conditioned on
    model_inputs = tokenizer('', return_tensors='pt')
    # transformers.set_seed(69)
    # no_sample = model.generate(**model_inputs, max_new_tokens=40, num_beams=3, do_sample=False, penalty_alpha=2.0, early_stopping=True, no_repeat_ngram_size=2, num_return_sequences=3)
    # do_sample = model.generate(**model_inputs, max_new_tokens=40, num_beams=3, do_sample=True, penalty_alpha=2.0, early_stopping=True, no_repeat_ngram_size=2, num_return_sequences=3, temperature=0.6, top_p=0.0001, top_k=1)
    group = model.generate(**model_inputs, max_new_tokens=9**9, num_beam_groups=2, num_beams=4, do_sample=False, early_stopping=True, no_repeat_ngram_size=2, num_return_sequences=4, top_k=50, diversity_penalty=1.0)

    for beam_output in group:
        print(tokenizer.decode(beam_output))


if '__main__' == __name__:
    main()
