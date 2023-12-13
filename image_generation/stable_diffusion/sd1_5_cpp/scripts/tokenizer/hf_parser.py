# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Dict, Callable, Union, List

from openvino.runtime.exceptions import OVTypeError
from .tokenizer_pipeline import (
    TokenizerPipeline,
    NormalizationStep,
    NormalizeUnicode,
    NMTNormalizationStep,
    CaseFoldStep,
    RegexNormalizationStep,
    StripStringStep,
    PreTokenizatinStep,
    PunctuationSplitStep,
    RegexSplitStep,
    WhitespaceSplitStep,
    BytesToCharsStep,
    WordPieceTokenizationStep,
    BPETokenizationStep,
    TruncationStep,
    PaddingStep,
    CombineSegmentsStep,
    VocabDecoderStep,
    CharsToBytesStep,
    RegexDecodingStep,
)


def parse_replace_normalizer(normalizer_dict: Dict[str, Any]) -> RegexNormalizationStep:
    regex_search_pattern = normalizer_dict["pattern"].get("String") or normalizer_dict["pattern"]["Regex"]
    return RegexNormalizationStep(
        regex_search_pattern=regex_search_pattern,
        replace_term=normalizer_dict["content"],
    )


def parse_bert_normalizer(normalizer_dict: Dict[str, Any]) -> List[NormalizationStep]:
    steps: List[NormalizationStep] = []

    if normalizer_dict["clean_text"] is True:
        steps.append(RegexNormalizationStep.del_control_chars_regex())

    # https://github.com/huggingface/tokenizers/blob/8c9cfb0b689bce00b615b9557a9a767f286d7a33/tokenizers/src/normalizers/bert.rs#L127
    if normalizer_dict.get("strip_accents") or normalizer_dict["lowercase"]:
        steps.append(NormalizeUnicode("NFD"))
        steps.append(RegexNormalizationStep.strip_accents_regex())

    if normalizer_dict["lowercase"] is True:
        steps.append(CaseFoldStep())

    return steps


def parse_strip_step(split_dict: Dict[str, Any]) -> StripStringStep:
    return StripStringStep(
        left=split_dict["strip_left"],
        right=split_dict["strip_right"],
    )


def parse_split_step(pretokenizer_dict: Dict[str, Any]) -> RegexSplitStep:
    split_pattern = pretokenizer_dict["pattern"].get("String") or pretokenizer_dict["pattern"]["Regex"]
    return RegexSplitStep(
        split_pattern=split_pattern,
        invert=pretokenizer_dict["invert"],
        behaviour=pretokenizer_dict["behavior"].lower().rstrip("d")
    )


def parse_byte_level_pretokenization_step(
        pretokenizer_dict: Dict[str, Any]
) -> List[Union[NormalizationStep, PreTokenizatinStep]]:
    steps = []
    if pretokenizer_dict.get("add_prefix_space"):
        steps.append(RegexNormalizationStep.add_prefix_whitespace_regex())

    # regex is used by default, but it does not appear in config yet
    if pretokenizer_dict.get("use_regex", True):
        # re2 does not support negative lookahead, so there is two steps replicate the behaviour
        steps.append(RegexSplitStep.add_whitespace_to_the_next_word())
        steps.append(RegexSplitStep.byte_level_splitter())

    steps.append(BytesToCharsStep())
    return steps


class TransformersTokenizerPipelineParser:
    def __init__(self, tokenizer_object: Any, number_of_inputs: int = 1) -> None:
        assert tokenizer_object.is_fast

        self.original_tokenizer = tokenizer_object
        with TemporaryDirectory() as tmpdir:
            tokenizer_object.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "tokenizer.json") as tj:
                self.tokenizer_json = json.load(tj)
        self.pipeline = TokenizerPipeline()
        self.number_of_inputs = number_of_inputs
        self.num_of_added_tokens = 0

    def parse(self, number_of_inputs: Optional[int] = None) -> TokenizerPipeline:
        self.number_of_inputs = self.number_of_inputs if number_of_inputs is None else number_of_inputs
        self.pipeline.number_of_inputs = self.number_of_inputs
        for add_steps in [
            self.normalization,
            self.pre_tokenization,
            self.tokenization_model,
            self.post_tokenization,
            self.decoding,
        ]:
            add_steps()

        return self.pipeline

    normalizers_map: Dict[str, Callable[[Dict[str, Any]], Union[NormalizationStep, List[NormalizationStep]]]] = {
        "NFC": lambda step_dict: NormalizeUnicode("NFC"),
        "NFD": lambda step_dict: NormalizeUnicode("NFD"),
        "NFKC": lambda step_dict: NormalizeUnicode("NFKC"),
        "NFKD": lambda step_dict: NormalizeUnicode("NFKD"),
        "Nmt": lambda step_dict: NMTNormalizationStep(),
        "Lowercase": lambda step_dict: CaseFoldStep(),
        "StripAccents": lambda step_dict: RegexNormalizationStep.strip_accents_regex(),
        "BertNormalizer": parse_bert_normalizer,
        "Replace": parse_replace_normalizer,
        "Strip": parse_strip_step,
    }

    def parse_normalizer_step(self, step_dict: Dict[str, Any]) -> None:
        try:
            self.pipeline.add_steps(self.normalizers_map[step_dict["type"]](step_dict))
        except KeyError:
            raise OVTypeError(f"Normalizer type '{step_dict['type']}' is not supported")

    def normalization(self) -> None:
        if self.tokenizer_json["normalizer"] is None:
            return

        if self.tokenizer_json["normalizer"].get("type") == "Sequence":
            for normalizer in self.tokenizer_json["normalizer"]["normalizers"]:
                self.parse_normalizer_step(normalizer)
        else:
            self.parse_normalizer_step(self.tokenizer_json["normalizer"])

    pre_tokenization_map: Dict[str, Callable[[Dict[str, Any]], Union[PreTokenizatinStep, List[PreTokenizatinStep]]]] = {
        "BertPreTokenizer": lambda step_dict: RegexSplitStep.bert_splitter(),
        "Whitespace": lambda step_dict: RegexSplitStep.whitespace_splitter(),
        "WhitespaceSplit": lambda step_dict: WhitespaceSplitStep(),
        "Split": parse_split_step,
        "Punctuation": lambda step_dict: PunctuationSplitStep(step_dict["behavior"]),
        "ByteLevel": parse_byte_level_pretokenization_step,
        "Digits": lambda step_dict: RegexSplitStep.digits_splitter(
            "isolate" if step_dict["individual_digits"] else "contiguous"
        )
    }

    def parse_pre_tokenization_step(self, step_dict: Dict[str, Any]) -> None:
        try:
            self.pipeline.add_steps(self.pre_tokenization_map[step_dict["type"]](step_dict))
        except KeyError:
            raise OVTypeError(f"Pre-tokenizer type '{step_dict['type']}' is not supported")

    def pre_tokenization(self) -> None:
        if self.tokenizer_json["pre_tokenizer"] is None:
            return

        if self.tokenizer_json["pre_tokenizer"].get("type") == "Sequence":
            for pretokenizer in self.tokenizer_json["pre_tokenizer"]["pretokenizers"]:
                self.parse_pre_tokenization_step(pretokenizer)
        else:
            self.parse_pre_tokenization_step(self.tokenizer_json["pre_tokenizer"])

    def tokenization_model(self) -> None:
        if self.tokenizer_json["model"]["type"] == "WordPiece":
            self.pipeline.add_steps(WordPieceTokenizationStep.from_hf_json(self.tokenizer_json))
            self.pipeline.vocab = self.pipeline[-1].vocab
        elif self.tokenizer_json["model"]["type"] == "BPE":
            self.pipeline.add_steps(BPETokenizationStep.from_hf_json(self.tokenizer_json))
            self.pipeline.vocab = self.pipeline[-1].vocab
        else:
            raise OVTypeError(f"Tokenizer type '{self.tokenizer_json['model']['type']}' is not supported")

    def post_tokenization(self) -> None:
        if (
                self.tokenizer_json["post_processor"] is None
                or self.tokenizer_json["post_processor"]["type"] == "ByteLevel"
        ):
            self.add_truncation()
            self.add_padding()
            return

        if self.tokenizer_json["post_processor"]["type"] == "TemplateProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_template_postprocessor(
                self.tokenizer_json, self.number_of_inputs
            )
        elif self.tokenizer_json["post_processor"]["type"] == "BertProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_bert_postprocessor(
                self.tokenizer_json, self.number_of_inputs
            )
        elif self.tokenizer_json["post_processor"]["type"] == "RobertaProcessing":
            combine_segments_step = CombineSegmentsStep.from_hf_json_roberta_processor(
                self.tokenizer_json, self.number_of_inputs
            )
        else:
            raise OVTypeError(f"Post-processor type '{self.tokenizer_json['post_processor']['type']}' is not supported")

        self.num_of_added_tokens += combine_segments_step.number_of_added_tokens
        combine_segments_step.set_tokens_ids(self.pipeline.vocab)

        self.add_truncation()
        self.pipeline.add_steps(combine_segments_step)

        self.add_padding()

    def add_truncation(self) -> None:
        if self.tokenizer_json["truncation"] is not None:
            self.pipeline.add_steps(TruncationStep.from_hf_json(self.tokenizer_json, self.num_of_added_tokens))
        elif self.original_tokenizer.model_max_length is not None:
            self.pipeline.add_steps(TruncationStep.from_hf_object(self.original_tokenizer, self.num_of_added_tokens))

    def add_padding(self) -> None:
        if self.tokenizer_json["padding"] is not None:
            self.pipeline.add_steps(PaddingStep.from_hf_json(self.tokenizer_json))
            self.pipeline[-1].set_token_id(self.pipeline.vocab)
        elif self.original_tokenizer.pad_token is not None:
            self.pipeline.add_steps(PaddingStep(token=self.original_tokenizer.pad_token))
            self.pipeline[-1].set_token_id(self.pipeline.vocab)
        else:
            self.pipeline.add_steps(PaddingStep())

    def decoding(self) -> None:
        if self.tokenizer_json["decoder"] is None:
            return

        if self.tokenizer_json["decoder"]["type"] == "ByteLevel":
            self.pipeline.add_steps(VocabDecoderStep())
            self.pipeline.add_steps(CharsToBytesStep())

        if self.original_tokenizer.clean_up_tokenization_spaces:
            self.pipeline.add_steps(RegexDecodingStep.clean_up_tokenization_spaces())
        return
