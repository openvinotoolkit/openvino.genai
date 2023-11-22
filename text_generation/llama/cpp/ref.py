from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch

import torch
from torch import nn

from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)


@torch.inference_mode()
def generate(self, input_ids, **kwargs):
    batch_size = input_ids.shape[0]
    # 8. prepare distribution pre_processing samplers
    # instantiate processors list
    logits_processor = LogitsProcessorList([
        HammingDiversityLogitsProcessor(
            diversity_penalty=kwargs['diversity_penalty'],
            num_beams=kwargs['num_beams'],
            num_beam_groups=kwargs['num_beam_groups'],
        ),
        NoRepeatNGramLogitsProcessor(kwargs['no_repeat_ngram_size'])
    ])

    max_length = kwargs['max_new_tokens'] + input_ids.shape[-1]
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=kwargs['num_beams'],
        device='cpu',
        length_penalty=kwargs['length_penalty'],
        do_early_stopping=kwargs['early_stopping'],
        num_beam_hyps_to_keep=kwargs['num_return_sequences'],
        num_beam_groups=kwargs['num_beam_groups'],
        max_length=kwargs['max_new_tokens'] + input_ids.shape[-1],
    )
    # 12. interleave input_ids with `num_beams` additional sequences per batch
    input_ids = input_ids.repeat_interleave(kwargs['num_beams'], dim=0)

    # 13. run beam search
    eos_token_id = self.generation_config.eos_token_id
    pad_token_id = self.generation_config.pad_token_id
    output_attentions = False
    output_hidden_states = False

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    batch_size = len(beam_scorer._beam_hyps) // num_beam_groups

    # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
    # the same group don't produce same tokens everytime.
    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float)
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:
        # predicted tokens in cur_len step
        current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype)

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long)

        # do one decoder step on all beams of all sentences in batch
        model_inputs = self.prepare_inputs_for_generation(input_ids)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

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
            beam_indices = None
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                group_index=beam_group_idx,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

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

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        if input_ids.shape[-1] >= max_length:
            break
        if beam_scorer.is_done:
            break

    batch_size = len(beam_scorer._beam_hyps) // beam_scorer.num_beam_groups

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_group_idx, beam_hyp in enumerate(beam_scorer._beam_hyps):
        if beam_scorer._done[batch_group_idx]:
            continue

        # all open beam hypotheses are added to the beam hypothesis
        # beam hypothesis class automatically keeps the best beams
        for index_per_group in range(beam_scorer.group_size):
            batch_beam_idx = batch_group_idx * beam_scorer.group_size + index_per_group
            final_score = beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
            beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

    # select the best hypotheses
    sent_lengths = input_ids.new(batch_size * beam_scorer.num_beam_hyps_to_keep)
    best = []
    best_indices = []
    best_scores = torch.zeros(batch_size * beam_scorer.num_beam_hyps_to_keep, device=beam_scorer.device, dtype=torch.float32)

    # retrieve best hypotheses
    for i in range(batch_size):
        beam_hyps_in_batch = beam_scorer._beam_hyps[i * beam_scorer.num_beam_groups : (i + 1) * beam_scorer.num_beam_groups]
        candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
        sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
        for j in range(beam_scorer.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            best_index = best_hyp_tuple[2]
            sent_lengths[beam_scorer.num_beam_hyps_to_keep * i + j] = len(best_hyp)

            # append hyp to lists
            best.append(best_hyp)

            # append indices to list
            best_indices.append(best_index)

            best_scores[i * beam_scorer.num_beam_hyps_to_keep + j] = best_score

    # prepare for adding eos
    sent_lengths_max = sent_lengths.max().item() + 1
    sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
    decoded: torch.LongTensor = input_ids.new(batch_size * beam_scorer.num_beam_hyps_to_keep, sent_max_len)

    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        decoded.fill_(pad_token_id)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, hypo in enumerate(best):
        decoded[i, : sent_lengths[i]] = hypo

        if sent_lengths[i] < sent_max_len:
            # inserting only the first eos_token_id
            decoded[i, sent_lengths[i]] = eos_token_id[0]
    return decoded


def main():
    model_path = '/home/wov/r/tiny-llama-fast-tokenizer/' # '/home/wov/r/openvino.genai/tiny-llama-fast-tokenizer/ones'  #'/home/wov/r/tiny-llama-fast-tokenizer/'# '/home/wov/r/openvino.genai/TinyLlama-1.1B-intermediate-step-715k-1.5T/'  #r'C:\Users\vzlobin\r\tiny-llama-fast-tokenizer' '/home/wov/r/tiny-llama-fast-tokenizer/' '/home/wov/r/TinyLlama-1.1B-intermediate-step-715k-1.5T/'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    model.generate = generate.__get__(model, transformers.GenerationMixin)
    torch.set_printoptions(profile='full', linewidth=9**9)
    print(model.generate(tokenizer('asdf', return_tensors='pt')['input_ids'], max_new_tokens=100, num_beam_groups=9, num_beams=99, num_return_sequences=99, do_sample=False, early_stopping=True, no_repeat_ngram_size=3, diversity_penalty=1.0, length_penalty=1.0))  # default length_penalty is 1.0


if '__main__' == __name__:
    main()
