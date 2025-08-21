// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::StopCriteria;
using ov::genai::StructuralTagItem;
using ov::genai::StructuralTagsConfig;
using ov::genai::StructuredOutputConfig;
using ov::genai::GenerationConfig;

namespace {

auto stop_criteria_docstring =  R"(
    StopCriteria controls the stopping condition for grouped beam search.

    The following values are possible:
        "openvino_genai.StopCriteria.EARLY" stops as soon as there are `num_beams` complete candidates.
        "openvino_genai.StopCriteria.HEURISTIC" stops when is it unlikely to find better candidates.
        "openvino_genai.StopCriteria.NEVER" stops when there cannot be better candidates.
)";

auto structured_output_config_docstring = R"(
    Structure to keep generation config parameters for structured output generation.
    It is used to store the configuration for structured generation, which includes
    the JSON schema and other related parameters.

    Structured output parameters:
    json_schema:           if set, the output will be a JSON string constraint by the specified json-schema.
    regex:          if set, the output will be constraint by specified regex.
    grammar:        if set, the output will be constraint by specified EBNF grammar.
    structural_tags_config: if set, the output will be constraint by specified structural tags configuration.
    compound_grammar:
        if set, the output will be constraint by specified compound grammar.
        Compound grammar is a combination of multiple grammars that can be used to generate structured outputs.
        It allows for more complex and flexible structured output generation.
        The compound grammar a Union or Concat of several grammars, where each grammar can be a JSON schema, regex, EBNF, Union or Concat.
)";

auto structured_tags_config_docstring = R"(
    Configures structured output generation by combining regular sampling with structural tags.

    When the model generates a trigger string, it switches to structured output mode and produces output
    based on the defined structural tags. Afterward, regular sampling resumes.

    Example:
      - Trigger "<func=" activates tags with begin "<func=sum>" or "<func=multiply>".

    Note:
      - Simple triggers like "<" may activate structured output unexpectedly if present in regular text.
      - Very specific or long triggers may be difficult for the model to generate,
      so structured output may not be triggered.

    Parameters:
    structural_tags: List of StructuralTagItem objects defining structural tags.
    triggers:        List of strings that trigger structured output generation.
                     Triggers may match the beginning or part of a tag's begin string.
)";

auto structured_tags_item_docstring = R"(
    Structure to keep generation config parameters for structural tags in structured output generation.
    It is used to store the configuration for a single structural tag item, which includes the begin string,
    schema, and end string.

    Parameters:
    begin:  the string that marks the beginning of the structural tag.
    schema: the JSON schema that defines the structure of the tag.
    end:    the string that marks the end of the structural tag.
)";

} // namespace

char generation_config_docstring[] = R"(
    Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
    and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
    be used while greedy and beam search parameters will not affect decoding at all.

    Parameters:
    max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                   max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
    max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
    min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
    ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
    eos_token_id:  token_id of <eos> (end of sentence)
    stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
    include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
    stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
    echo:           if set to true, the model will echo the prompt in the output.
    logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                    Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
    apply_chat_template: whether to apply chat_template for non-chat scenarios

    repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
    presence_penalty: reduces absolute log prob if the token was generated at least once.
    frequency_penalty: reduces absolute log prob as many times as the token was generated.

    Beam search specific parameters:
    num_beams:         number of beams for beam search. 1 disables beam search.
    num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
    diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
    length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
        likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
        length_penalty < 0.0 encourages shorter sequences.
    num_return_sequences: the number of sequences to return for grouped beam search decoding.
    no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
    stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
        "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
        "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
        "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).

    Random sampling parameters:
    temperature:        the value used to modulate token probabilities for random sampling.
    top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
    do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
    num_return_sequences: the number of sequences to generate from a single prompt.
)";


template <typename PyClass>
void add_grammar_operators(PyClass& py_cls) {
    py_cls
        .def("__add__", [](py::object self, py::object other) {
            return pyutils::py_obj_to_compound_grammar(self) + pyutils::py_obj_to_compound_grammar(other);
        })
        .def("__or__", [](py::object self, py::object other) {
            return pyutils::py_obj_to_compound_grammar(self) | pyutils::py_obj_to_compound_grammar(other);
        });
};

void init_generation_config(py::module_& m) {
    // Binding for StopCriteria
    py::enum_<StopCriteria>(m, "StopCriteria", stop_criteria_docstring)
        .value("EARLY", StopCriteria::EARLY)
        .value("HEURISTIC", StopCriteria::HEURISTIC)
        .value("NEVER", StopCriteria::NEVER);


    py::class_<StructuralTagItem>(m, "StructuralTagItem", structured_tags_item_docstring)
        .def(py::init<>(), "Default constructor for StructuralTagItem")
        .def(py::init([](py::kwargs kwargs) {
            return StructuralTagItem(pyutils::kwargs_to_any_map(kwargs));
        }), "Constructor that initializes the structured tags configuration with kwargs.")
        .def_readwrite("begin", &StructuralTagItem::begin, "Begin string for Structural Tag Item")
        .def_readwrite("schema", &StructuralTagItem::schema, "Json schema for Structural Tag Item")
        .def_readwrite("end", &StructuralTagItem::end, "End string for Structural Tag Item")
        .def("__repr__",
            [](const StructuralTagItem &self) {
                return "StructuralTagItem(begin=" + py::repr(py::cast(self.begin)).cast<std::string>() +
                       ", schema=" + py::repr(py::cast(self.schema)).cast<std::string>() +
                       ", end=" + py::repr(py::cast(self.end)).cast<std::string>() + ")";
            }
        );


    py::class_<StructuralTagsConfig>(m, "StructuralTagsConfig", structured_tags_config_docstring)
        .def(py::init<>(), "Default constructor for StructuralTagsConfig")
        .def(py::init([](py::kwargs kwargs) {
            return StructuralTagsConfig(pyutils::kwargs_to_any_map(kwargs));
        }), "Constructor that initializes the structured tags configuration with kwargs.")
        .def_readwrite("structural_tags", &StructuralTagsConfig::structural_tags, "List of structural tag items for structured output generation")
        .def_readwrite("triggers", &StructuralTagsConfig::triggers, "List of strings that will trigger generation of structured output")
        .def("__repr__",
            [](const StructuralTagsConfig &self) {
                return "StructuralTagsConfig(structural_tags=" + py::repr(py::cast(self.structural_tags)).cast<std::string>() +
                       ", triggers=" + py::repr(py::cast(self.triggers)).cast<std::string>() + ")";
            }
        );

    auto structured_output_config = py::class_<StructuredOutputConfig>(m, "StructuredOutputConfig", structured_output_config_docstring)
        .def(py::init<>(), "Default constructor for StructuredOutputConfig")
        .def(py::init([](py::kwargs kwargs) {
            return StructuredOutputConfig(pyutils::kwargs_to_any_map(kwargs));
        }), "Constructor that initializes the structured output configuration with kwargs.")
        .def_readwrite("json_schema", &StructuredOutputConfig::json_schema, "JSON schema for structured output generation")
        .def_readwrite("regex", &StructuredOutputConfig::regex, "Regular expression for structured output generation")
        .def_readwrite("grammar", &StructuredOutputConfig::grammar, "Grammar for structured output generation")
        .def_readwrite("structural_tags_config", &StructuredOutputConfig::structural_tags_config, "Configuration for structural tags in structured output generation")
        .def_readwrite("compound_grammar", &StructuredOutputConfig::compound_grammar, "Compound grammar for structured output generation")
        .def("__repr__",
            [](const StructuredOutputConfig &self) {
                return "StructuredOutputConfig(json_schema=" + py::repr(py::cast(self.json_schema)).cast<std::string>() +
                       ", regex=" + py::repr(py::cast(self.regex)).cast<std::string>() +
                       ", grammar=" + py::repr(py::cast(self.grammar)).cast<std::string>() +
                       ", structural_tags_config=" + py::repr(py::cast(self.structural_tags_config)).cast<std::string>() +
                       ", compound_grammar=" + py::repr(py::cast(self.compound_grammar)).cast<std::string>() + ")";
            }
        );

    auto regex = py::class_<StructuredOutputConfig::Regex>(structured_output_config, "Regex")
        .def(py::init<const std::string&>(), "Regex building block for compound grammar configuration.")
        .def_readwrite("value", &StructuredOutputConfig::Regex::value)
        .def("__repr__", [](const StructuredOutputConfig::Regex& self) { return self.to_string(); });
    add_grammar_operators(regex);

    auto json_schema = py::class_<StructuredOutputConfig::JSONSchema>(structured_output_config, "JSONSchema")
        .def(py::init<const std::string&>(), "JSON schema building block for compound grammar configuration.")
        .def_readwrite("value", &StructuredOutputConfig::JSONSchema::value)
        .def("__repr__", [](const StructuredOutputConfig::JSONSchema& self) { return self.to_string(); });
    add_grammar_operators(json_schema);

    auto ebnf = py::class_<StructuredOutputConfig::EBNF>(structured_output_config, "EBNF")
        .def(py::init<const std::string&>(), "EBNF grammar building block for compound grammar configuration.")
        .def_readwrite("value", &StructuredOutputConfig::EBNF::value)
        .def("__repr__", [](const StructuredOutputConfig::EBNF& self) { return self.to_string(); });
    add_grammar_operators(ebnf);

    auto concat = py::class_<StructuredOutputConfig::Concat, std::shared_ptr<StructuredOutputConfig::Concat>>(structured_output_config, "Concat")
        .def_static("__new__", [](py::object cls, py::object left, py::object right) {
            return std::make_shared<StructuredOutputConfig::Concat>(
                pyutils::py_obj_to_compound_grammar(left),
                pyutils::py_obj_to_compound_grammar(right)
            );
        }, "Concat combines two grammars sequentially, e.g. \"A B\" means A followed by B")
        .def_readwrite("left", &StructuredOutputConfig::Concat::left)
        .def_readwrite("right", &StructuredOutputConfig::Concat::right)
        .def("__repr__", [](const StructuredOutputConfig::Concat& self) { return self.to_string(); });
    add_grammar_operators(concat);

    auto union_ = py::class_<StructuredOutputConfig::Union, std::shared_ptr<StructuredOutputConfig::Union>>(structured_output_config, "Union")
        .def_static("__new__", [](py::object cls, py::object left, py::object right) {
            return std::make_shared<StructuredOutputConfig::Union>(
                pyutils::py_obj_to_compound_grammar(left),
                pyutils::py_obj_to_compound_grammar(right)
            );
        }, "Union combines two grammars in parallel, e.g. \"A | B\" means either A or B")
        .def_readwrite("left", &StructuredOutputConfig::Union::left)
        .def_readwrite("right", &StructuredOutputConfig::Union::right)
        .def("__repr__", [](const StructuredOutputConfig::Union& self) { return self.to_string(); });
    add_grammar_operators(union_);

    // Binding for GenerationConfig
    py::class_<GenerationConfig>(m, "GenerationConfig", generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](py::kwargs kwargs) { return *pyutils::update_config_from_kwargs(GenerationConfig(), kwargs); }))
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &GenerationConfig::ignore_eos)
        .def_readwrite("min_new_tokens", &GenerationConfig::min_new_tokens)
        .def_readwrite("num_beam_groups", &GenerationConfig::num_beam_groups)
        .def_readwrite("num_beams", &GenerationConfig::num_beams)
        .def_readwrite("diversity_penalty", &GenerationConfig::diversity_penalty)
        .def_readwrite("length_penalty", &GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &GenerationConfig::num_return_sequences)
        .def_readwrite("no_repeat_ngram_size", &GenerationConfig::no_repeat_ngram_size)
        .def_readwrite("stop_criteria", &GenerationConfig::stop_criteria)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id)
        .def_readwrite("presence_penalty", &GenerationConfig::presence_penalty)
        .def_readwrite("frequency_penalty", &GenerationConfig::frequency_penalty)
        .def_readwrite("rng_seed", &GenerationConfig::rng_seed)
        .def_readwrite("stop_strings", &GenerationConfig::stop_strings)
        .def_readwrite("echo", &GenerationConfig::echo)
        .def_readwrite("logprobs", &GenerationConfig::logprobs)
        .def_readwrite("assistant_confidence_threshold", &GenerationConfig::assistant_confidence_threshold)
        .def_readwrite("num_assistant_tokens", &GenerationConfig::num_assistant_tokens)
        .def_readwrite("max_ngram_size", &GenerationConfig::max_ngram_size)
        .def_readwrite("include_stop_str_in_output", &GenerationConfig::include_stop_str_in_output)
        .def_readwrite("stop_token_ids", &GenerationConfig::stop_token_ids)
        .def_readwrite("structured_output_config", &GenerationConfig::structured_output_config)
        .def_readwrite("adapters", &GenerationConfig::adapters)
        .def_readwrite("apply_chat_template", &GenerationConfig::apply_chat_template)
        .def("set_eos_token_id", &GenerationConfig::set_eos_token_id, py::arg("tokenizer_eos_token_id"))
        .def("is_beam_search", &GenerationConfig::is_beam_search)
        .def("is_greedy_decoding", &GenerationConfig::is_greedy_decoding)
        .def("is_multinomial", &GenerationConfig::is_multinomial)
        .def("is_assisting_generation", &GenerationConfig::is_assisting_generation)
        .def("is_prompt_lookup", &GenerationConfig::is_prompt_lookup)
        .def("validate", &GenerationConfig::validate)
        .def("update_generation_config", [](
            ov::genai::GenerationConfig& config,
            const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });
   }
