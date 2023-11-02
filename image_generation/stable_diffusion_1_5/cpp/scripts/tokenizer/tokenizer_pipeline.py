# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field
from functools import singledispatchmethod
from itertools import chain
from typing import List, Optional, Any, Dict
from unittest.mock import MagicMock
import weakref

import numpy as np

from openvino.runtime.exceptions import UserInputError, OVTypeError
from openvino.runtime import Type, PartialShape, op, Model, Output, Node, opset10
from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime.utils.types import as_node, make_constant_node


def pack_strings(strings):
    assert isinstance(strings, list)
    to_bytes = lambda x: x.to_bytes(4, "little")
    batch_size = len(strings)
    if batch_size == 0:
        return to_bytes(0)
    offsets = to_bytes(0)
    symbols = bytes()
    for s in strings:
        symbols += bytes(s, "utf-8")
        offsets += to_bytes(len(symbols))
    return np.frombuffer(bytearray(to_bytes(batch_size) + offsets + symbols), np.uint8)


def pack_string(s):
    return np.frombuffer(
        bytes(s + " ", "utf-8"), dtype=np.uint8
    )  # + ' ' is WA for CPU bug

current_dir = os.path.dirname(os.path.abspath(__file__))
extension_path = os.path.join(current_dir, "../../extensions/libuser_ov_extensions.so")
factory = NodeFactory()
# TODO: Use relative path
factory.add_extension(extension_path)


class BasePipelineStep:
    _pipeline = field(default=None, init=False, repr=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {
            key: getattr(self, key)
            for key in dir(type(self))
            if not key.startswith("_") and isinstance(getattr(type(self), key), property)
        }
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline()

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = weakref.ref(pipeline)

    def get_ov_subgraph(self, *input_nodes: List[Output]) -> Node:
        raise NotImplementedError

    @staticmethod
    def create_string_constant_node(value: str) -> op.Constant:
        if isinstance(value, str):
            # string scalar
            ps = pack_string(value)
            return op.Constant(ps)
        else:
            # support only 1D strings for now
            ps = pack_strings(value)
            return factory.create("StringTensorUnpack", op.Constant(ps).outputs())


@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class NormalizeUnicode(NormalizationStep):
    normalization_form: str = "NFD"

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return factory.create(
            "NormalizeUnicode", input_nodes, {"normalization_form": self.normalization_form}
        ).outputs()


@dataclass
class CaseFoldStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return factory.create("CaseFold", input_nodes).outputs()


@dataclass
class RegexNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def strip_accents_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def add_prefix_whitespace_regex(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r"^(\S)", replace_term=r" \1")

    @classmethod
    def del_control_chars_regex(cls) -> "RegexNormalizationStep":
        # https://github.com/huggingface/tokenizers/blob/8c9cfb0b689bce00b615b9557a9a767f286d7a33/tokenizers/src/normalizers/bert.rs#L17
        return cls(regex_search_pattern=r"((?=[^\n\t\r])\p{Cc})|((?=[^\n\t\r])\p{Cf})", replace_term=" ")

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexNormalizationStep":
        return cls(regex_search_pattern=r" ([\.\?\!\,])| ('[ms])| (') | ('[rv]e)", replace_term="\1")

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.regex_search_pattern).outputs(),
                *self.create_string_constant_node(self.replace_term).outputs(),
            )
        )
        return factory.create(
            "RegexNormalization",
            input_nodes
        ).outputs()


@dataclass
class NMTNormalizationStep(NormalizationStep):
    """Normaization based on NMT task.

    https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/tokenizers/src/normalizers/unicode.rs#L44
    """


@dataclass
class StripAccentsStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return RegexNormalizationStep.strip_accents_regex().get_ov_subgraph(input_nodes)


@dataclass
class DelControlCharsStep(NormalizationStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return RegexNormalizationStep.del_control_chars_regex().get_ov_subgraph(input_nodes)


@dataclass
class StripStringStep(NormalizationStep):
    left: bool
    right: bool


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class RegexSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "remove"

    @classmethod
    def bert_whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(split_pattern=r"\s+", invert=False)

    @classmethod
    def bert_keep_delimeters_splitter(cls) -> "RegexSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
            invert=False,
            behaviour="isolate"
        )

    @classmethod
    def bert_splitter(cls) -> List["RegexSplitStep"]:
        return [cls.bert_whitespace_splitter(), cls.bert_keep_delimeters_splitter()]

    @classmethod
    def whitespace_splitter(cls) -> "RegexSplitStep":
        return cls(r"\w+|[^\w\s]+", invert=True)

    @classmethod
    def byte_level_splitter(cls) -> "RegexSplitStep":
        return cls(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+",
            invert=False,
            behaviour="isolate",
        )

    @classmethod
    def add_whitespace_to_the_next_word(cls):
        return cls(
            r"\s\S",
            invert=False,
            behaviour="merge_with_next"
        )

    @classmethod
    def digits_splitter(cls, behaviour="isolate") -> "RegexSplitStep":
        return cls(
            r"\p{Nd}|\p{Nl}|\p{No}",
            invert=False,
            behaviour=behaviour,
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        input_nodes.extend(
            self.create_string_constant_node(self.split_pattern).outputs()
        )
        return factory.create(
            "RegexSplit",
            input_nodes,
            {
                "behaviour": self.behaviour.lower(),
                "invert": self.invert,
            },
        ).outputs()


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""
    def get_ov_subgraph(self, input_nodes: List[Output]) -> List[Output]:
        return RegexSplitStep.whitespace_splitter().get_ov_subgraph(input_nodes)


@dataclass
class PunctuationSplitStep(PreTokenizatinStep):
    """Splits string on punctuation chars."""
    # behaviour: str = "Isolated"


@dataclass
class BytesToCharsStep(PreTokenizatinStep):
    """Maps chars to other chars for Byte-level BPE Tokenizer"""
    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return factory.create(
            "BytesToChars",
            input_nodes,
        ).outputs()


@dataclass
class TokenizationModelStep(BasePipelineStep):
    pass


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    unk_token: str = "[UNK]"
    suffix_indicator: str = "##"
    max_bytes_per_word: int = 100
    unk_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_id = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.vocab).outputs(),
                *as_node(self.unk_token_id).outputs(),
            )
        )
        return factory.create(
            "WordpieceTokenizer",
            input_nodes,
            {
                "suffix_indicator": self.suffix_indicator,
                "max_bytes_per_word": self.max_bytes_per_word,
            }
        ).outputs()


@dataclass
class BPETokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    merges: List[str] = field(repr=False)
    unk_token: str = ""
    fuse_unk: bool = False
    suffix_indicator: str = ""
    end_suffix: str = ""
    byte_fallback: bool = False

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "BPETokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"] or "",
            fuse_unk=tokenizer_json["model"]["fuse_unk"] or False,
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"] or "",
            end_suffix=tokenizer_json["model"]["end_of_word_suffix"] or "",
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
            merges=tokenizer_json["model"]["merges"],
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        pipeline = self.get_pipeline()
        pipeline.vocab_node_outputs = self.create_string_constant_node(self.vocab).outputs()
        input_nodes.extend(
            (
                *self.get_pipeline().vocab_node_outputs,
                *self.create_string_constant_node(self.merges).outputs(),
            )
        )
        return factory.create(
            "BPETokenizer",
            input_nodes,
            {
                "unk_token": self.unk_token,
                "fuse_unk": self.fuse_unk,
                "suffix_indicator": self.suffix_indicator,
                "end_suffix": self.end_suffix,
                "byte_fallback": self.byte_fallback,
            }
        ).outputs()


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class TruncationStep(PostTokenizationStep):
    max_length: int
    truncate_right: bool = True
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any], num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
            2**31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        max_length = min(
            tokenizer.model_max_length - num_of_added_tokens,
            2 ** 31 - 1 - num_of_added_tokens,
        )
        return cls(
            max_length=max_length,
            truncate_right=tokenizer.truncation_side == "right",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]):
        # FIXME: disabled for now
        # operation = string_ops.Truncation(
        #     *input_nodes,
        #     as_node(self.max_length),
        #     self.truncate_right,
        #     self.axis,
        # )
        # operation.configure_mock(**{"outputs.return_value": [MagicMock() for _ in range(len(input_nodes))]})
        # return operation
        #print('[ TOKENIZER PIPELINE CONVERSION ] WARNING: Truncation is not applied because it is not implemented')
        #print('Trunc max_length:', self.max_length)
        # FIXME: Truncation side (truncate_right) is ignored
        # TODO: Check if axis is the right-most dimension
        assert len(input_nodes) == 3, 'Only one input ragged tensor is supported as an input for TruncationStep'

        max_length = opset10.minimum(opset10.subtract(input_nodes[1], input_nodes[0]), make_constant_node(self.max_length, Type.i32))
        return [input_nodes[0], opset10.add(input_nodes[0], max_length).output(0), input_nodes[2]]


@dataclass
class SpecialTokenWithId:
    token: Optional[str] = None
    _token_id: Optional[int] = None

    def set_token_id(self, vocab: Optional[List[str]]) -> None:
        if vocab is not None and self.token in vocab:
            self._token_id = vocab.index(self.token)

    @property
    def token_id(self) -> Optional[int]:
        return self._token_id


@dataclass
class TokenWithTypeId:
    token_type_id: Optional[int] = None


@dataclass
class AddToken(TokenWithTypeId, SpecialTokenWithId):
    pass


@dataclass
class Sequence(TokenWithTypeId):
    pass


@dataclass
class CombineSegmentsStep(PostTokenizationStep):
    inputs: List[TokenWithTypeId] = field(default_factory=list)
    segment_ids: Optional[List[int]] = None
    axis: int = -1

    def __post_init__(self):
        if self.segment_ids is not None:
            return

        segment_ids_tensor = [node.token_type_id for node in self.inputs]
        if any(segment is None for segment in segment_ids_tensor):
            segment_ids_tensor = [0] * len(self.inputs)

        self.segment_ids = segment_ids_tensor

    def set_tokens_ids(self, vocab: Optional[List[int]]) -> None:
        for input_ in self.inputs:
            if isinstance(input_, AddToken):
                input_.set_token_id(vocab)

    @property
    def number_of_added_tokens(self) -> int:
        return sum(1 for input_ in self.inputs if isinstance(input_, AddToken))

    @classmethod
    def from_hf_json_template_postprocessor(
            cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        inputs: List[TokenWithTypeId] = []
        if number_of_inputs == 1:
            post_processor = tokenizer_json["post_processor"]["single"]
        else:
            post_processor = tokenizer_json["post_processor"]["pair"]

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                step = AddToken(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                )
                inputs.append(step)
            else:
                inputs.append(Sequence(token_type_id=template_dict["Sequence"]["type_id"]))

        return cls(inputs)

    @classmethod
    def from_hf_json_bert_postprocessor(
            cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        post_processor_dict = tokenizer_json["post_processor"]
        inputs: List[TokenWithTypeId] = [
            AddToken(
                token=post_processor_dict["cls"][0],
                token_type_id=0,
            ),
            Sequence(token_type_id=0),
            AddToken(
                token=post_processor_dict["sep"][0],
                token_type_id=0,
            ),
        ]

        if number_of_inputs == 2:
            inputs.extend(
                [
                    Sequence(token_type_id=1),
                    AddToken(
                        token=post_processor_dict["sep"][0],
                        token_type_id=1,
                    ),
                ]
            )
        return cls(inputs)

    @classmethod
    def from_hf_json_roberta_processor(
            cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        post_processor_dict = tokenizer_json["post_processor"]

        inputs: List[TokenWithTypeId] = [Sequence(token_type_id=0)]

        if not post_processor_dict.get("add_special_tokens", True):
            return cls(inputs)

        inputs.insert(
            0, AddToken(token=post_processor_dict["cls"][0], token_type_id=0)
        )
        inputs.append(AddToken(token=post_processor_dict["sep"][0], token_type_id=0))

        if number_of_inputs == 2:
            print("WARNING: Pair of inputs not supported for RoBERTa postprocessor")

        return cls(inputs)

    def get_ov_subgraph(self, input_nodes):
        number_of_sequence_inputs = sum(
            1 for input_ in self.inputs if isinstance(input_, Sequence)
        )
        #print('number_of_sequence_inputs:', number_of_sequence_inputs)
        if number_of_sequence_inputs != len(input_nodes)/3:
            raise UserInputError(
                f"Number of input nodes: {len(input_nodes)}, must be equal to {number_of_sequence_inputs}"
            )

        op_inputs = []
        i = 0

        for node in self.inputs:
            if isinstance(node, Sequence):
                op_inputs += input_nodes[3*i:3*(i+1)]
                i += 1
            else:
                # Put a scalar as a ragged tensor with scalar shape and a single element
                op_inputs += make_constant_node(0, Type.i32).outputs()
                op_inputs += make_constant_node(1, Type.i32).outputs()
                #print('Should be scalar:', op_inputs[-1])
                #print('token', node._token_id)
                op_inputs.append(make_constant_node(np.array([node._token_id]), Type.i32).output(0))

        op_inputs.append(make_constant_node(self.segment_ids, Type.i32).output(0))

        #print('op_inputs:', op_inputs)

        # FIXME: Disabled for now, no implementation
        # operation = string_ops.CombineSegments(
        #     *op_inputs,
        #     self.segment_ids,
        #     self.axis,
        # )
        # operation.configure_mock(**{"outputs.return_value": [MagicMock()]})
        # return operation

        # Decomposed implementation

        return factory.create('CombineSegments', op_inputs).outputs()
        print(input_nodes)
        assert len(input_nodes) == 3, '[ TOKENIZER PIPELINE CONVERSION ] CombineSegments can be converted for a single ragged input tensor only, this is temporary limitation'
        print('self.segment_ids:', self.segment_ids)
        # Make another ragged tensor with identical structure but with all values filled with self.segment_ids[0]
        segment_ids_output = [input_nodes[0], input_nodes[1], opset10.broadcast(make_constant_node(self.segment_ids[0], Type.i32), opset10.shape_of(input_nodes[2])).output(0)]
        print('[ TOKENIZER PIPELINE CONVERSION ] [ DEBUG ] CombineSegments outputs:', input_nodes + segment_ids_output)
        return input_nodes + segment_ids_output



@dataclass
class PaddingStep(PostTokenizationStep, SpecialTokenWithId):
    pad_right: bool = True
    token_type_id: Optional[int] = None
    max_length: int = -1
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "PaddingStep":
        padding_dict = tokenizer_json["padding"]
        return cls(
            token=padding_dict["pad_token"],
            pad_right=padding_dict["direction"] == "Right",
            token_type_id=padding_dict["pad_type_id"],
            # TODO: Initialize max_length
        )

    def get_ov_subgraph(self, input_nodes):
        # Suppose input_nodes may have multiple tuples each with 3 tensors represented decomposed ragged tensors
        # We suppose that all ragged tensors represent the same structure and produce the mask only once
        assert len(input_nodes) % 3 == 0
        assert len(input_nodes) >= 3

        #lens = opset10.subtract(input_nodes[1], input_nodes[2])
        #max_len = opset10.reduce_max(lens)
        #padded_len =
        outputs = []
        #print(self.token)
        #print('max_length =', self.max_length)
        #print('ERRROR: SETTING MAX_LENGTH = 100')
        #print('ERROR: Ignoring pad token and set it to id = 0')

        if self.max_length == -1 or self.max_length >= 2 ** 31:
            # Calculate max_length as the maximum ragged length
            max_length = opset10.reduce_max(opset10.subtract(input_nodes[1], input_nodes[0]), make_constant_node(0, Type.i32))
        else:
            max_length = make_constant_node(self.max_length, Type.i32)

        #if self.token_type_id == -1:
        #    self.token_type_id = 0
        names = ["input_ids", "token_type_ids"]
        for i, name in zip(range(len(input_nodes)//3), names):
            #print(input_nodes[3*i:3*(i+1)])
            #print(as_node(self.max_length).outputs())
            #print(as_node(np.array(0, dtype=int)).outputs())
            cur_outputs = factory.create(
                "RaggedToDense",
                input_nodes[3*i:3*(i+1)] + max_length.outputs() + make_constant_node(0, Type.i32).outputs()
            ).outputs()
            cur_outputs[0].tensor.add_names({name})

            outputs.append(cur_outputs[0])
            if i == 0:
                mask = opset10.convert(cur_outputs[1], "i32").output(0)  # TODO: Change RaggedToDense to generate mask of any type

        mask.tensor.add_names({"attention_mask"})
        outputs.append(mask)

        return outputs


@dataclass
class DecodingStep(BasePipelineStep):
    pass


@dataclass
class VocabDecoderStep(DecodingStep):
    def get_vocab_node_outputs(self) -> Optional[List[Output]]:
        return self.get_pipeline().vocab_node_outputs

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        input_nodes.extend(self.get_vocab_node_outputs())
        return factory.create("VocabDecoder", input_nodes, {}).outputs()


@dataclass
class CharsToBytesStep(DecodingStep):
    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        return factory.create("CharsToBytes", input_nodes, {}).outputs()


@dataclass
class RegexDecodingStep(DecodingStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def clean_up_tokenization_spaces(cls) -> "RegexDecodingStep":
        return cls(
            regex_search_pattern=r" ([\\.\\?\\!,])| ('[ms])| (') | ('[rv]e)| (n't)",
            replace_term=r"\1",
        )

    def get_ov_subgraph(self, input_nodes: List[Output]) -> Node:
        input_nodes.extend(
            (
                *self.create_string_constant_node(self.regex_search_pattern).outputs(),
                *self.create_string_constant_node(self.replace_term).outputs(),
            )
        )
        return factory.create(
            "RegexNormalization",
            input_nodes
        ).outputs()


@dataclass
class TokenizerPipeline:
    steps: List[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[List[str]] = field(default=None, repr=False)
    number_of_inputs: int = 1
    vocab_node_outputs: Optional[List[Output]] = field(default=None, repr=False)

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]

    @property
    def normalization_steps(self) -> List[NormalizationStep]:
        return [step for step in self.steps if isinstance(step, NormalizationStep)]

    @property
    def pretokenization_steps(self) -> List[PreTokenizatinStep]:
        return [step for step in self.steps if isinstance(step, PreTokenizatinStep)]

    @property
    def tokenization_steps(self) -> List[TokenizationModelStep]:
        return [step for step in self.steps if isinstance(step, TokenizationModelStep)]

    @property
    def post_tokenization_steps(self) -> List[PostTokenizationStep]:
        return [step for step in self.steps if isinstance(step, PostTokenizationStep)]

    @property
    def decoding_steps(self) -> List[DecodingStep]:
        return [step for step in self.steps if isinstance(step, DecodingStep)]

    def create_string_input(self) -> Node:
        return op.Parameter(Type.u8, PartialShape(["?"]))

    def create_int_input(self, input_type=Type.i32) -> Node:
        return op.Parameter(input_type, PartialShape(["?", "?", "?"]))

    def create_processing_pipeline(self, input_nodes: List[op.Parameter]) -> List[Node]:
        processing_pipelines_outputs = []

        for input_node in input_nodes:
            input_node = factory.create("StringTensorUnpack", input_node.outputs()).outputs()
            for step in self.normalization_steps:
                input_node = step.get_ov_subgraph(input_node)

            shape = opset10.shape_of(input_node[0])
            batch_size = opset10.gather(shape, as_node(0), as_node(0))

            # FIXME: Cannot create range with specific data type from python
            ragged_begins = opset10.convert(
                opset10.range(as_node(0), batch_size, as_node(1)),
                "i32",
            ).outputs()
            ragged_ends = opset10.convert(
                opset10.range(
                    as_node(1),
                    opset10.add(batch_size, as_node(1)),
                    as_node(1),
                ),
                "i32",
            ).outputs()
            input_node = ragged_begins + ragged_ends + input_node

            for step in chain(self.pretokenization_steps, self.tokenization_steps):
                input_node = step.get_ov_subgraph(input_node)

            processing_pipelines_outputs += input_node

        return processing_pipelines_outputs

    def create_post_tokenization_pipeline(self, input_nodes: List[op.Parameter]) -> List[Output]:
        for step in self.post_tokenization_steps:
            pipeline_step = step.get_ov_subgraph(input_nodes)
            input_nodes = pipeline_step

        return input_nodes

    def create_decoding_pipeline(self, input_nodes: List[Output]) -> List[Output]:
        for step in self.decoding_steps:
            pipeline_step = step.get_ov_subgraph(input_nodes)
            input_nodes = pipeline_step

        return factory.create("StringTensorPack", input_nodes).outputs()

    def get_encoder_ov_subgraph(self) -> Model:
        input_nodes = [self.create_string_input() for _ in range(self.number_of_inputs)]
        processing_outputs = self.create_processing_pipeline(input_nodes)
        outputs = self.create_post_tokenization_pipeline(processing_outputs)
        return Model(outputs, input_nodes, name="tokenizer_encoder")

    def get_greedy_decoding_ov_subgraph(self, input_node: op.Parameter) -> List[Output]:
        argmax = opset10.topk(
            data=input_node,
            k=1,
            axis=-1,
            mode="max",
            sort="none",
            name="ArgMax",
        )
        return opset10.squeeze(
            data=argmax.output(1),
            axes=-1,
        ).outputs()

    def get_decoder_ov_subgraph(self) -> Model:
        input_node = self.create_int_input()
        argmax = self.get_greedy_decoding_ov_subgraph(input_node)
        outputs = self.create_decoding_pipeline(argmax)
        model = Model(outputs, [input_node], name="tokenizer_decoder")
        model.output().tensor.add_names({"string_output"})
        return model

