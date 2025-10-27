// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <limits>
#include <variant>
#include <string>
#include <sstream>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/lora_adapter.hpp"

namespace ov {
namespace genai {

/**
 * @brief controls the stopping condition for grouped beam search. The following values are possible:
 *        "EARLY" stops as soon as there are `num_beams` complete candidates.
          "HEURISTIC" stops when is it unlikely to find better candidates.
          "NEVER" stops when there cannot be better candidates.
 */
enum class StopCriteria { EARLY, HEURISTIC, NEVER };


/**
 * @brief StructuralTagItem is used to define a structural tag with its properties.
 * @param begin the string that marks the beginning of the structural tag.
 * @param schema JSON schema that defines the structure of the tag.
 * @param end the string that marks the end of the structural tag.
 */
struct OPENVINO_GENAI_EXPORTS StructuralTagItem {
    StructuralTagItem() = default;
    StructuralTagItem(const ov::AnyMap& properties);
    void update_config(const ov::AnyMap& properties);
    std::string to_string() const;

    bool operator==(const StructuralTagItem& other) const {
        return begin == other.begin && schema == other.schema && end == other.end;
    }

    std::string begin;
    std::string schema;
    std::string end;
};

/**
 * @brief Configures structured output generation by combining regular sampling with structural tags.
 *
 * When the model generates a trigger string, it switches to structured output mode and produces output
 * based on the defined structural tags. Afterward, regular sampling resumes.
 *
 * Example:
 *   - Trigger "<func=" activates tags with begin "<func=sum>" or "<func=multiply>".
 *
 * Note:
 *   - Simple triggers like "<" may activate structured output unexpectedly if present in regular text.
 *   - Very specific or long triggers may be difficult for the model to generate, so structured output may not be triggered.
 *
 * @param structural_tags List of StructuralTagItem objects defining structural tags.
 * @param triggers List of strings that trigger structured output generation. Triggers may match the beginning or part of a tag's begin string.
 */
struct OPENVINO_GENAI_EXPORTS StructuralTagsConfig {
public:
    StructuralTagsConfig() = default;
    StructuralTagsConfig(const ov::AnyMap& properties);
    void update_config(const ov::AnyMap& properties);
    std::string to_string() const;
    std::string to_json() const;

    bool operator==(const StructuralTagsConfig& other) const {
        return structural_tags == other.structural_tags && triggers == other.triggers;
    }

    std::vector<StructuralTagItem> structural_tags;
    std::vector<std::string> triggers;
};

/* 
* Structured output parameters:
* @param json_schema if set, the output will be a JSON string constrained by the specified json_schema.
* @param regex if set, the output will be constrained by specified regex.
* @param grammar if set, the output will be constrained by specified EBNF grammar.
* @param structural_tags_config if set, the output could contain substrings constrained by the specified structural tags.
* @param backend if set, the structured output generation will use specified backend, currently only "xgrammar" is supported.
* 
* If several parameters are set, e.g. json_schema and regex, then an error will be thrown when validating the configuration.
*/
class OPENVINO_GENAI_EXPORTS StructuredOutputConfig {
public:
    /* 
    * @brief Constructor that initializes the structured output configuration with properties.
    * @param properties A map of properties to initialize the structured output configuration.
    * 
    * Example: StructuredOutputConfig config({{ov::genai::json_schema(json_schema_str)}});
    */
    StructuredOutputConfig(const ov::AnyMap& properties);
    StructuredOutputConfig() = default;

    static std::string format_for_json(const std::string& input) {
        std::ostringstream stream;
        stream << '"';
        for (char character : input) {
            switch (character) {
                case '"': stream << "\\\""; break;
                case '\\': stream << "\\\\"; break;
                case '\b': stream << "\\b"; break;
                case '\f': stream << "\\f"; break;
                case '\n': stream << "\\n"; break;
                case '\r': stream << "\\r"; break;
                case '\t': stream << "\\t"; break;
                default: {
                    // Interpret `character` as a raw byte to avoid sign-extension on platforms where `char` is signed. 
                    unsigned char uc = static_cast<unsigned char>(character);
                    if (uc < 0x20) {
                        // control characters < 0x20 must be escaped as \uXXXX in JSON
                        stream << "\\u" << std::hex << std::uppercase << std::setfill('0') << std::setw(4)
                               << static_cast<int>(uc) << std::dec << std::nouppercase;
                    } else {
                        stream << character;
                    }
                }
            }
        }
        stream << '"';
        return stream.str();
    }

    // base grammar types for structural tags construction
    /**
     * @brief Regex structural tag constrains output using a regular expression.
     */
    struct Regex {
        std::string value;

        Regex() = default;
        Regex(const std::string& regex) : value(regex) {}
        std::string to_string() const {
            return "Regex(\"" + value + "\")";
        }
        std::string to_json() const {
            return std::string("{\"type\": \"regex\", \"pattern\": ") + format_for_json(value) + "}";
        }
        bool operator==(const Regex& other) const {
            return value == other.value;
        }
    };

    /**
     * @brief JSONSchema structural tag constrains output to a JSON document that
     *        must conform to the provided JSON Schema string.
     */
    struct JSONSchema {
        std::string value;

        JSONSchema() = default;
        JSONSchema(const std::string& schema) : value(schema) {}
        std::string to_string() const {
            return "JSONSchema(\"" + value + "\")";
        }
        std::string to_json() const {
            return std::string("{\"type\": \"json_schema\", \"json_schema\": ") + value + "}";
        }
        bool operator==(const JSONSchema& other) const {
            return value == other.value;
        }
    };

    /**
     * @brief EBNF structural tag constrains output using an EBNF grammar.
     */
    struct EBNF {
        std::string value;

        EBNF() = default;
        EBNF(const std::string& grammar) : value(grammar) {}
        std::string to_string() const {
            return "EBNF(\"" + value + "\")";
        }
        std::string to_json() const {
            return std::string("{\"type\": \"grammar\", \"grammar\": ") + format_for_json(value) + "}";
        }
        bool operator==(const EBNF& other) const {
            return value == other.value;
        }
    };

    /**
     * @brief ConstString structural tag forces the generator to produce exactly
     *        the provided constant string value.
     */
    struct ConstString {
        std::string value;

        ConstString() = default;
        ConstString(const std::string& str) : value(str) {}
        std::string to_string() const {
            return "ConstString(\"" + value + "\")";
        }
        std::string to_json() const {
            return std::string("{\"type\": \"const_string\", \"value\": ") + format_for_json(value) + "}";
        }
        bool operator==(const ConstString& other) const {
            return value == other.value;
        }
    };

    /**
     * @brief AnyText structural tag allows any text for the portion
     *        of output covered by this tag.
     */
    struct AnyText {
        AnyText() = default;
        std::string to_string() const {
            return "AnyText()";
        }
        std::string to_json() const {
            return "{\"type\": \"any_text\"}";
        }
        bool operator==(const AnyText& other) const {
            return true;
        }
    };

    /**
     * @brief QwenXMLParametersFormat instructs the generator to output an XML
     *        parameters block derived from the provided JSON schema. This is a
     *        specialized helper for Qwen-style XML parameter formatting.
     */
    struct QwenXMLParametersFormat {
        std::string json_schema;

        QwenXMLParametersFormat() = default;
        QwenXMLParametersFormat(const std::string& schema) : json_schema(schema) {};
        std::string to_json() const {
            return std::string("{\"type\": \"qwen_xml_parameter\", \"json_schema\": ") + json_schema + "}";
        };
        std::string to_string() const {
            return "QwenXMLParametersFormat(json_schema=" + json_schema + ")";
        };
        bool operator==(const QwenXMLParametersFormat& other) const {
            return json_schema == other.json_schema;
        }
    };

    // nested grammar types
    struct Concat;
    struct Union;
    struct Tag;
    struct TriggeredTags;
    struct TagsWithSeparator;

    using StructuralTag = std::variant<
        std::string,
        Regex,
        JSONSchema,
        EBNF,
        ConstString,
        AnyText,
        QwenXMLParametersFormat,
        std::shared_ptr<Concat>,
        std::shared_ptr<Union>,
        std::shared_ptr<Tag>,
        std::shared_ptr<TriggeredTags>,
        std::shared_ptr<TagsWithSeparator>
    >;
    using CompoundGrammar = StructuralTag;

    template <typename T>
    static std::string structural_tag_to_string(const T& g) {
        if constexpr (std::is_same_v<T, std::string>) {
                return g;
        } else if constexpr (std::is_same_v<T, ov::genai::StructuredOutputConfig::Regex> ||
                      std::is_same_v<T, ov::genai::StructuredOutputConfig::JSONSchema> ||
                      std::is_same_v<T, ov::genai::StructuredOutputConfig::EBNF> ||
                      std::is_same_v<T, ov::genai::StructuredOutputConfig::ConstString> ||
                      std::is_same_v<T, ov::genai::StructuredOutputConfig::AnyText> ||
                      std::is_same_v<T, ov::genai::StructuredOutputConfig::QwenXMLParametersFormat>) {
            return g.to_string();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Concat>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Union>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Tag>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::TriggeredTags>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::TagsWithSeparator>>) {
            return g ? g->to_string() : std::string("null");
        } else {
            OPENVINO_THROW("Unsupported structural tag, cannot convert to string:" + std::string(typeid(g).name()));
        }
    }

    template <typename T>
    static std::string structural_tag_to_json(const T& g) {
        if constexpr (std::is_same_v<T, std::string>) {
            return g;
        } else if constexpr (std::is_same_v<T, ov::genai::StructuredOutputConfig::Regex> ||
                             std::is_same_v<T, ov::genai::StructuredOutputConfig::JSONSchema> ||
                             std::is_same_v<T, ov::genai::StructuredOutputConfig::EBNF> ||
                             std::is_same_v<T, ov::genai::StructuredOutputConfig::ConstString> ||
                             std::is_same_v<T, ov::genai::StructuredOutputConfig::AnyText> ||
                             std::is_same_v<T, ov::genai::StructuredOutputConfig::QwenXMLParametersFormat>) {
            return g.to_json();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Concat>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Union>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::Tag>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::TriggeredTags>> ||
                             std::is_same_v<T, std::shared_ptr<ov::genai::StructuredOutputConfig::TagsWithSeparator>>) {
            return g ? g->to_json() : std::string("null");
        } else {
            OPENVINO_THROW("Unsupported structural tag, cannot convert to json:" + std::string(typeid(g).name()));
        }
    }

    // nested grammar types
    /**
     * @brief Concat composes multiple structural tags in sequence. Each element
     *        must be produced in the given order.
     *        Can be used indirectly with + operator.
     *
     * Example: Concat(ConstString("a"), ConstString("b")) produces "ab".
     *          ConstString("a") + ConstString("b") is equivalent.
     */
    struct Concat {
        std::vector<StructuralTag> elements;

        Concat() = default;
        Concat(StructuralTag left, StructuralTag right) : elements{std::move(left), std::move(right)} {};
        Concat(const std::vector<StructuralTag>& elems) : elements(elems) {};
        std::string to_json() const {
            std::ostringstream oss;
            oss << "{\"type\": \"sequence\", \"elements\": [";
            for (size_t i = 0; i < elements.size(); ++i) {
                oss << std::visit([](const auto& g) { return structural_tag_to_json(g); }, elements[i]);
                if (i != elements.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]}";
            return oss.str();
        };
        std::string to_string() const {
            std::ostringstream oss;
            oss << "Concat(";
            for (size_t i = 0; i < elements.size(); ++i) {
                oss << std::visit([](const auto& g) -> std::string { return structural_tag_to_string(g); }, elements[i]);
                if (i != elements.size() - 1) {
                    oss << ", ";
                }
            }
            oss << ")";
            return oss.str();
        }
        bool operator==(const Concat& other) const {
            return elements == other.elements;
        }
    };

    // Union combines two grammars in parallel, e.g. "A | B" means either A or B
    /**
     * @brief Union composes multiple structural tags as alternatives. The
     *        model may produce any one of the provided elements.
     *        Can be used indirectly with | operator.
     */
    struct Union {
        std::vector<StructuralTag> elements;

        Union() = default;
        Union(StructuralTag left, StructuralTag right) : elements{std::move(left), std::move(right)} {};
        Union(const std::vector<StructuralTag>& elems) : elements(elems) {};
        std::string to_json() const {
            std::ostringstream oss;
            oss << "{\"type\": \"or\", \"elements\": [";
            for (size_t i = 0; i < elements.size(); ++i) {
                oss << std::visit([](const auto& g) -> std::string { return structural_tag_to_json(g); }, elements[i]);
                if (i != elements.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]}";
            return oss.str();
        }
        std::string to_string() const {
            std::ostringstream oss;
            oss << "Union(";
            for (size_t i = 0; i < elements.size(); ++i) {
                oss << std::visit([](const auto& g) -> std::string { return structural_tag_to_string(g); }, elements[i]);
                if (i != elements.size() - 1) {
                    oss << ", ";
                }
            }
            oss << ")";
            return oss.str();
        }
        bool operator==(const Union& other) const {
            return elements == other.elements;
        }
    };

    /**
     * @brief Tag defines a begin/end wrapper with constrained inner content.
     *
     * The generator will output `begin`, then the `content` (a StructuralTag),
     * and finally `end`.
     *
     * Example: Tag("<think>", AnyText(), "</think>") represents thinking portion of the model output.
     */
    struct Tag {
        std::string begin;
        StructuralTag content;
        std::string end;

        Tag() = default;
        Tag(const std::string& begin, StructuralTag content, const std::string& end) : begin(begin), content(std::move(content)), end(end) {};
        std::string to_json() const {
            std::ostringstream oss;
            oss << "{\"type\": \"tag\", \"begin\": " << format_for_json(begin) << ", \"content\": " <<
                   std::visit([](const auto& g) -> std::string { return structural_tag_to_json(g); }, content) <<
                   ", \"end\": " << format_for_json(end) << "}";
            return oss.str();
        };
        std::string to_string() const {
            std::ostringstream oss;
            oss << "Tag(begin=\"" << begin << "\", content=" <<
                   std::visit([](const auto& g) -> std::string { return structural_tag_to_string(g); }, content) <<
                   ", end=\"" << end << "\")";
            return oss.str();
        };
        bool operator==(const Tag& other) const {
            return begin == other.begin && content == other.content && end == other.end;
        }
    };

    /**
     * @brief TriggeredTags associates a set of `triggers` with multiple `tags`.
     *
     * When the model generates any of the trigger strings the structured generation
     * activates to produce configured tags. Flags allow requiring
     * at least one tag and stopping structured generation after the first tag.
     */
    struct TriggeredTags {
        std::vector<std::string> triggers;
        std::vector<Tag> tags;
        bool at_least_one = false;  // if true, at least one tag must be generated after trigger
        bool stop_after_first = false; // if true, structured generation stops after first tag is generated
        
        TriggeredTags() = default;
        TriggeredTags(const std::vector<std::string>& triggers,
                      const std::vector<Tag>& tags,
                      bool at_least_one = false,
                      bool stop_after_first = false)
            : triggers(triggers), tags(tags), at_least_one(at_least_one), stop_after_first(stop_after_first) {};
        std::string to_json() const {
            std::ostringstream oss;
            oss << "{\"type\": \"triggered_tags\", \"triggers\": [";
            for (size_t i = 0; i < triggers.size(); ++i) {
                oss << format_for_json(triggers[i]);
                if (i != triggers.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], \"tags\": [";
            for (size_t i = 0; i < tags.size(); ++i) {
                oss << tags[i].to_json();
                if (i != tags.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], \"at_least_one\": " << (at_least_one ? "true" : "false") <<
                   ", \"stop_after_first\": " << (stop_after_first ? "true" : "false") << "}";
            return oss.str();
        };
        std::string to_string() const {
            std::ostringstream oss;
            oss << "TriggeredTags(triggers=[";
            for (size_t i = 0; i < triggers.size(); ++i) {
                oss << "\"" << triggers[i] << "\"";
                if (i != triggers.size() - 1) {
                    oss << ", ";
                }
            };
            oss << "], tags=[";
            for (size_t i = 0; i < tags.size(); ++i) {
                oss << tags[i].to_string();
                if (i != tags.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], at_least_one=" << (at_least_one ? "True" : "False") <<
                   ", stop_after_first=" << (stop_after_first ? "True" : "False") << ")";
            return oss.str();
        };
    };

    /**
     * @brief TagsWithSeparator configures generation of a sequence of tags
     *        separated by a fixed `separator` string.
     *
     * Can be used to produce repeated tagged elements like
     * "<f>A</f>;<f>B</f>" where `separator`=";".
     */
    struct TagsWithSeparator {
        std::vector<Tag> tags;
        std::string separator;
        bool at_least_one = false;  // if true, at least one tag must be generated
        bool stop_after_first = false; // if true, generation stops after first tag is generated

        TagsWithSeparator() = default;
        TagsWithSeparator(const std::vector<Tag>& tags, 
                          const std::string& separator,
                          bool at_least_one = false,
                          bool stop_after_first = false)
            : tags(tags), separator(separator), at_least_one(at_least_one), stop_after_first(stop_after_first) {};
        std::string to_json() const {
            std::ostringstream oss;
            oss << "{\"type\": \"tags_with_separator\", \"separator\": " << format_for_json(separator) << ", \"tags\": [";
            for (size_t i = 0; i < tags.size(); ++i) {
                oss << tags[i].to_json();
                if (i != tags.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], \"at_least_one\": " << (at_least_one ? "true" : "false") <<
                   ", \"stop_after_first\": " << (stop_after_first ? "true" : "false") << "}";
            return oss.str();
        };
        std::string to_string() const {
            std::ostringstream oss;
            oss << "TagsWithSeparator(separator=\"" << separator << "\", tags=[";
            for (size_t i = 0; i < tags.size(); ++i) {
                oss << tags[i].to_string();
                if (i != tags.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], at_least_one=" << (at_least_one ? "true" : "false") <<
                   ", stop_after_first=" << (stop_after_first ? "true" : "false") << ")";
            return oss.str();
        };
    };

    std::optional<std::string> json_schema;
    std::optional<std::string> regex;
    std::optional<std::string> grammar;
    std::optional<std::variant<StructuralTagsConfig, StructuralTag>> structural_tags_config;
    std::optional<CompoundGrammar> compound_grammar;
    std::optional<std::string> backend;
    void validate() const;
    void validate(Tokenizer& tokenizer) const;
    void update_config(const ov::AnyMap& properties);
};


OPENVINO_GENAI_EXPORTS std::shared_ptr<StructuredOutputConfig::Concat>
operator+(const StructuredOutputConfig::StructuralTag& lhs,
          const StructuredOutputConfig::StructuralTag& rhs);

OPENVINO_GENAI_EXPORTS std::shared_ptr<StructuredOutputConfig::Union>
operator|(const StructuredOutputConfig::StructuralTag& lhs,
          const StructuredOutputConfig::StructuralTag& rhs);

/**
 * @brief Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
 * and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
 * be used while greedy and beam search parameters will not affect decoding at all.
 *
 * Generic parameters:
 * @param max_length the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
 *        `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
 * @param max_new_tokens the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
 * @param ignore_eos if set to true, then generation will not stop even if <eos> token is met.
 * @param eos_token_id token_id of <eos> (end of sentence)
 * @param min_new_tokens set 0 probability for eos_token_id for the first eos_token_id generated tokens.
 *
 * @param stop_strings A set of strings that will cause pipeline to stop generating further tokens.
 * @param include_stop_str_in_output if set to true stop string that matched generation will be included in generation output (default: false)
 * @param stop_token_ids A set of tokens that will cause pipeline to stop generating further tokens.
 * @param echo if set to true, output will include user prompt (default: false).
 * @param logprobs number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
 *                 Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
 *
 * @param repetition_penalty the parameter for repetition penalty. 1.0 means no penalty.
 * @param presence_penalty reduces absolute log prob if the token was generated at least once.
 * @param frequency_penalty reduces absolute log prob as many times as the token was generated.
 *
 * Beam search specific parameters:
 * @param num_beams number of beams for beam search. 1 disables beam search.
 * @param num_beam_groups number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
 * @param diversity_penalty this value is subtracted from a beam's score if it generates the same token as any beam from other group at a
 *        particular time. See https://arxiv.org/pdf/1909.05858.
 * @param length_penalty exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
 *        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
 *        likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
 *        `length_penalty` < 0.0 encourages shorter sequences.
 * @param num_return_sequences the number of sequences to return for grouped beam search decoding per batch element. num_return_sequences must be less or equal to num_beams.
 * @param no_repeat_ngram_size if set to int > 0, all ngrams of that size can only occur once.
 * @param stop_criteria controls the stopping condition for grouped beam search. It accepts the following values:
 *        "EARLY", where the generation stops as soon as there are `num_beams` complete candidates; "HEURISTIC", where an
 *        "HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
 *        "NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
 *
 * Random (or multinomial) sampling parameters:
 * @param do_sample whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
 * @param temperature the value used to modulate token probabilities for random sampling.
 * @param top_p - if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
 * @param top_k the number of highest probability vocabulary tokens to keep for top-k-filtering.
 * @param rng_seed initializes random generator.
 *
 * Assisting generation parameters:
 * @param assistant_confidence_threshold the lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number update.
          NOTE: `assistant_confidence_threshold` is supported only by ContinuousBatching backend for Speculative Decode.
 * @param num_assistant_tokens the defined candidates number to be generated by draft model/prompt lookup in case of static strategy candidates number update.
 *        NOTE: ContinuousBatching backend for Speculative Decode uses `num_assistant_tokens` as is. Stateful backend for Speculative Decode uses `num_assistant_tokens`'s
 *        copy as initial value and adjusts it based on recent number of accepted tokens. If `num_assistant_tokens` is not set, it defaults to `5` for both backends.
 * @param max_ngram_size is maximum ngram to use when looking for matches in the prompt.
 *
 * @param structured_output_config if set, the output will be a string constrained by the specified json_schema, regex, or EBNF grammar.
 * 
 * @param apply_chat_template whether or not to apply chat_template for non-chat scenarios
 */
class OPENVINO_GENAI_EXPORTS GenerationConfig {
public:
    GenerationConfig() = default;
    explicit GenerationConfig(const std::filesystem::path& json_path);

    // Generic
    size_t max_new_tokens = SIZE_MAX;
    size_t max_length = SIZE_MAX;
    bool ignore_eos = false;
    size_t min_new_tokens = 0;
    bool echo = false;
    size_t logprobs = 0;

    // EOS special token
    int64_t eos_token_id = -1;
    std::set<std::string> stop_strings;
    // Default setting in vLLM (and OpenAI API) is not to include stop string in the output
    bool include_stop_str_in_output = false;
    std::set<int64_t> stop_token_ids;

    // penalties (not used in beam search)
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0;
    float frequency_penalty = 0.0f;

    // Beam search specific
    size_t num_beam_groups = 1;
    size_t num_beams = 1;
    float diversity_penalty = 0.0f;
    float length_penalty = 1.0f;
    size_t num_return_sequences = 1;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    StopCriteria stop_criteria = StopCriteria::HEURISTIC;

    // Multinomial
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = std::numeric_limits<size_t>::max();
    bool do_sample = false;
    size_t rng_seed = 0;

    // Assisting generation parameters
    float assistant_confidence_threshold = 0.f;
    size_t num_assistant_tokens = 0;
    size_t max_ngram_size = 0;

    // Structured output parameters
    std::optional<StructuredOutputConfig> structured_output_config;

    std::optional<AdapterConfig> adapters;

    // set to true if chat template should be applied for non-chat scenarios, set to false otherwise
    bool apply_chat_template = true;


    /** @brief sets eos_token_id to tokenizer_eos_token_id if eos_token_id is less than 0.
     * Otherwise verifies eos_token_id == tokenizer_eos_token_id.
     */
    void set_eos_token_id(size_t tokenizer_eos_token_id);
    size_t get_max_new_tokens(size_t prompt_length = 0) const;

    bool is_greedy_decoding() const;
    bool is_beam_search() const;
    bool is_multinomial() const;
    bool is_assisting_generation() const;
    bool is_prompt_lookup() const;
    bool is_structured_output_generation() const;

    OPENVINO_DEPRECATED("Please, use `is_assisting_generation()` instead of `is_speculative_decoding()`. This method will be removed in 2026.0.0 release")
    bool is_speculative_decoding() const;

    void update_generation_config(const ov::AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters, e.g. do_sample=true and num_beams > 1.
    /// @throws Exception if config is invalid.
    void validate() const;
};

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(input_ids, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
 * pipe(text, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
*/
static constexpr ov::Property<size_t> max_new_tokens{"max_new_tokens"};
static constexpr ov::Property<size_t> max_length{"max_length"};
static constexpr ov::Property<bool> ignore_eos{"ignore_eos"};
static constexpr ov::Property<size_t> min_new_tokens{"min_new_tokens"};
static constexpr ov::Property<std::set<std::string>> stop_strings{"stop_strings"};
static constexpr ov::Property<bool> include_stop_str_in_output{"include_stop_str_in_output"};
static constexpr ov::Property<std::set<int64_t>> stop_token_ids{"stop_token_ids"};

static constexpr ov::Property<size_t> num_beam_groups{"num_beam_groups"};
static constexpr ov::Property<size_t> num_beams{"num_beams"};
static constexpr ov::Property<float> diversity_penalty{"diversity_penalty"};
static constexpr ov::Property<float> length_penalty{"length_penalty"};
static constexpr ov::Property<size_t> num_return_sequences{"num_return_sequences"};
static constexpr ov::Property<size_t> no_repeat_ngram_size{"no_repeat_ngram_size"};
static constexpr ov::Property<StopCriteria> stop_criteria{"stop_criteria"};

static constexpr ov::Property<float> temperature{"temperature"};
static constexpr ov::Property<float> top_p{"top_p"};
static constexpr ov::Property<size_t> top_k{"top_k"};
static constexpr ov::Property<bool> do_sample{"do_sample"};
static constexpr ov::Property<float> repetition_penalty{"repetition_penalty"};
static constexpr ov::Property<int64_t> eos_token_id{"eos_token_id"};
static constexpr ov::Property<float> presence_penalty{"presence_penalty"};
static constexpr ov::Property<float> frequency_penalty{"frequency_penalty"};
extern OPENVINO_GENAI_EXPORTS ov::Property<size_t> rng_seed;

static constexpr ov::Property<float> assistant_confidence_threshold{"assistant_confidence_threshold"};
static constexpr ov::Property<size_t> num_assistant_tokens{"num_assistant_tokens"};
static constexpr ov::Property<size_t> max_ngram_size{"max_ngram_size"};

static constexpr ov::Property<StructuredOutputConfig> structured_output_config{"structured_output_config"};
static constexpr ov::Property<std::string> regex{"regex"};
static constexpr ov::Property<std::string> json_schema{"json_schema"};
static constexpr ov::Property<std::string> grammar{"grammar"};
static constexpr ov::Property<std::string> backend{"backend"};

static constexpr ov::Property<bool> apply_chat_template{"apply_chat_template"};

// Predefined Configs

OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig beam_search();
OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig greedy();
OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig multinomial();

}  // namespace genai
}  // namespace ov
