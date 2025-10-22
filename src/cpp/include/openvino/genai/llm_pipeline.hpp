// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <chrono>
#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov {
namespace genai {

// Return flag corresponds whether generation should be stopped. It could be:
// ov::genai::StreamingStatus flag, RUNNING means continue generation, STOP means stop generation, CANCEL means stop generation and remove last propmt and answer from history
// *DEPRECATED* bool flag, false means continue generation, true means stop. Please, use `ov::genai::StreamingStatus` instead.
using StreamerVariant = std::variant<std::function<bool(std::string)>, std::function<StreamingStatus(std::string)>, std::shared_ptr<StreamerBase>, std::monostate>;
using OptionalGenerationConfig = std::optional<GenerationConfig>;
using EncodedInputs = std::variant<ov::Tensor, TokenizedInputs>;
using StringInputs = std::variant<std::string, std::vector<std::string>>;

/**
* @brief Structure to store resulting batched tokens and scores for each batch sequence.
* The first num_return_sequences elements correspond to the first batch element.
* In the case if results decoded with beam search and random sampling scores contain
* sum of logarithmic probabilities for each token in the sequence. In the case
* of greedy decoding scores are filled with zeros.
*
* @param tokens sequence of resulting tokens
* @param scores sum of logarithmic probabilities of all tokens in the sequence
* @param perf_metrics performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics
* @param extended_perf_metrics pipeline specific performance metrics etc. of type ov::genai::PerfMetrics.
*        Applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
*        To get metrics, it should be cast to corresponding class for extended perf metrics from pipeline.
*        Cast to SDPerModelsPerfMetrics for SpeculativeDecoding.
*/
class EncodedResults {
public:
    std::vector<std::vector<int64_t>> tokens;
    std::vector<float> scores;
    PerfMetrics perf_metrics;
    std::shared_ptr<ExtendedPerfMetrics> extended_perf_metrics;
};

/**
* @brief Structure to store resulting batched text outputs and scores for each batch
* The first num_return_sequences elements correspond to the first batch element.
*
* @param texts vector of resulting sequences
* @param scores scores for each sequence
* @param perf_metrics performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics
* @param extended_perf_metrics pipeline specific performance metrics etc. of type ov::genai::PerfMetrics
*        Applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
*        To get metrics, it should be cast to corresponding class for extended perf metrics from pipeline.
*        Cast to SDPerModelsPerfMetrics for SpeculativeDecoding.
*/
class DecodedResults {
public:
    std::vector<std::string> texts;
    std::vector<float> scores;
    PerfMetrics perf_metrics;
    std::shared_ptr<ExtendedPerfMetrics> extended_perf_metrics;

    // @brief Convert DecodedResults to a string.
    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    // @brief Convert DecodedResults to a single string.
    // @return std::string containing the texts from the DecodedResults object.
    operator std::vector<std::string>() const {
        return texts;
    }

     // @brief Overloads operator<< to enhance output the contents of DecodedResults.
     // @return A reference to the output stream with the concatenated texts.
    friend std::ostream& operator<<(std::ostream& os, const DecodedResults& dr) {
        OPENVINO_ASSERT(
            dr.scores.size() == dr.texts.size(),
            "The number of scores and texts doesn't match in DecodedResults."
        );
        if (dr.texts.empty()) {
            return os;
        }
        if (dr.texts.size() == 1) {
            os << dr.texts[0];
            return os;
        }
        for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
            os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
        }
        return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
    }
};

class LLMPipelineImplBase;

/**
* @brief This class is used for generation with LLMs.
 */
class OPENVINO_GENAI_EXPORTS LLMPipeline {
public:
    /**
    * @brief Constructs an LLMPipeline from xml/bin files, tokenizers and configuration in the same dir.
    *
    * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
    * @param device optional device
    * @param properties optional properties
    * Add ov::genai::scheduler_config property to properties to create continuous batching pipeline.
    * Add ov::genai::adapters property to properties to register LoRA adapters.
    */
    LLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    LLMPipeline(
        const std::string& model_str,
        const ov::Tensor& weights_tensor,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {}
    );

    /**
    * @brief Constructs an LLMPipeline from xml/bin files, tokenizers and configuration in the same dir.
    * Accepts arbitrary list of optional properties.
    *
    * @param models_path Path to the dir model xml/bin files, tokenizers and generation_config.json
    * @param device optional device
    * @param properties optional plugin properties, ov::genai::adapters property for LoRA adapters and
    * ov::genai::scheduler_config property to create continuous batching pipeline. Properties can be
    * specified in any order.
    */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    LLMPipeline(
            const std::filesystem::path& models_path,
            const std::string& device,
            Properties&&... properties)
        : LLMPipeline(models_path, device,  AnyMap{std::forward<Properties>(properties)...}) {
    }

    /**
    * @brief Constructs an LLMPipeline from already existing infer InferRequest and Tokenizer
    *
    * @param request infer request of the model
    * @param tokenizer initialized Tokenizer
    * @param generation_config optional generation_config, be default will be initialized for greedy decoding
    */
    LLMPipeline(
        const ov::InferRequest& request,
        const ov::genai::Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config = std::nullopt
    );

    /**
    * @brief Constructs a LLMPipeline when ov::genai::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param models_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param tokenizer manually initialized ov::genai::Tokenizer
    * @param device optional device
    * @param properties optional plugin_config
    * Add ov::genai::scheduler_config property to plugin_config to create continuous batching pipeline
    */
    LLMPipeline(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~LLMPipeline();

    /**
    * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
    *
    * @param inputs input prompt or a vector of prompts
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return DecodedResults decoded resulting text
    * chat_template will be applied to the prompt, run pipe.get_tokenizer().set_chat_template(custom_chat_template) to update it.
    * To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    */
    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief High level generate that receives prompts as a string or a vector of strings and returns decoded output.
    * properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100), ov::genai::streamer(lambda_func)).
    *
    * @param inputs input prompt or a vector of prompts
    * @param properties properties
    * @return DecodedResults decoded resulting text
    * chat_template will be applied to the prompt, run pipe.get_tokenizer().set_chat_template(custom_chat_template) to update it.
    * To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
            StringInputs inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    DecodedResults generate(StringInputs inputs, const ov::AnyMap& config_map);


    DecodedResults operator()(
        StringInputs inputs,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    ) {
        return generate(inputs, generation_config, streamer);
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> operator()(
            StringInputs inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief High level generate that receives ChatHistory and returns decoded output.
    *
    * @param history ChatHistory with messages
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return DecodedResults decoded resulting text
    * 
    * Chat template will be applied to the prompt, run `pipe.get_tokenizer().set_chat_template(custom_chat_template)` to update it.
    * To disable chat template set `generation_config.apply_chat_template` to `false`.
    */
    DecodedResults generate(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief High level generate that receives ChatHistory and returns decoded output.
    * Properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100), ov::genai::streamer(lambda_func)).
    *
    * @param history ChatHistory with messages
    * @param properties properties
    * @return DecodedResults decoded resulting text
    * 
    * Chat template will be applied to the prompt, run `pipe.get_tokenizer().set_chat_template(custom_chat_template)` to update it.
    * To disable chat template set `generation_config.apply_chat_template` to `false`.
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
            const ChatHistory& history,
            Properties&&... properties) {
        return generate(history, AnyMap{std::forward<Properties>(properties)...});
    }
    DecodedResults generate(const ChatHistory& history, const ov::AnyMap& config_map);

    DecodedResults operator()(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    ) {
        return generate(history, generation_config, streamer);
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> operator()(
            const ChatHistory& history,
            Properties&&... properties) {
        return generate(history, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids or pair of (input_ids, attentino_mask) encoded input prompt tokens
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config = std::nullopt,
        StreamerVariant streamer=std::monostate()
    );

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids or pair of (input_ids, attentino_mask) encoded input prompt tokens
    * @param generation config params
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
            const EncodedInputs& inputs,
            Properties&&... properties) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }
    EncodedResults generate(const EncodedInputs& inputs, const ov::AnyMap& config_map);

    ov::genai::Tokenizer get_tokenizer();
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& config);


    /**
    * @brief start chat with keeping history in kv cache.
    * Turns on keeping KV cache between generate calls.
    * In case if beam search is used, KV cache is kept for the generated sequence with maximal scores.
    *
    * @param system_message optional system message.
    */
    void start_chat(const std::string& system_message = {});

    /**
    * @brief finish chat and clear kv cache.
    * Turns off keeping KV cache between generate calls.
    */
    void finish_chat();

private:
    std::string m_device;
    std::unique_ptr<LLMPipelineImplBase> m_pimpl;
};

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> streamer(StreamerVariant func);
OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> generation_config(const GenerationConfig& config);

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> draft_model(
    std::string& model_str,
    ov::Tensor& weights_tensor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device = {},
    const ov::AnyMap& properties = {},
    const ov::genai::GenerationConfig& generation_config = {});

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> draft_model(
    const std::filesystem::path& models_path,
    const std::string& device = {},
    const ov::AnyMap& properties = {});

template <typename... Properties,
          typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
inline std::pair<std::string, Any> draft_model(
    const std::filesystem::path& models_path,
    const std::string& device,
    Properties&&... properties) {
    return draft_model(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...});
}

template <typename... Properties,
          typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
inline std::pair<std::string, Any> draft_model(
    const std::filesystem::path& models_path,
    Properties&&... properties) {
    return draft_model(models_path, {}, ov::AnyMap{std::forward<Properties>(properties)...});
}

/**
* @brief scheduler_config property serves to activate continuous batching pipeline.
* Create SchedulerConfig and fill it with suitable values. Copy or move it to plugin_config.
* And create LLMPipeline instance with this config.
*/
static constexpr ov::Property<SchedulerConfig> scheduler_config{"scheduler_config"};

/**
* @brief enable prompt_lookup property serves to activate prompt lookup decoding.
* Set `true` to activate this mode.
* And create LLMPipeline instance with this config.
*/
static constexpr ov::Property<bool> prompt_lookup{"prompt_lookup"};

/**
* @brief enable enable_save_ov_model property serves to serialize ov model (xml/bin) generated from gguf model on disk for re-use.
* Set `true` to activate this mode.
* And create LLMPipeline instance with this config.
*/
static constexpr ov::Property<bool> enable_save_ov_model{"enable_save_ov_model"};


}  // namespace genai
}  // namespace ov
