// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <random>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "vlm_sampling.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/image_embedder.hpp"

#include <openvino/runtime/core.hpp>

#include "text_callback_streamer.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {
   
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

constexpr size_t BATCH_SIZE = 1;
} // namespace

class ov::genai::VLMPipeline::VLMPipelineImpl {
public:
    // A config to follow for LLM input construction.
    VLMConfig m_vlm_config;
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    ov::InferRequest m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    // Chat history
    ChatHistory m_history;
    // Templated chat history
    std::string m_templated_chat_history;
    // InputsEmbedder
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;

    VLMPipelineImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    ) :
        m_vlm_config{
            utils::from_config_json_if_exists<ov::genai::VLMConfig>(
                model_dir, "config.json"
            )
        },
        m_tokenizer{model_dir.string(), device_config},
        m_is_chat_conversation{false} {
        auto get_embedding_model_path = [] (ov::genai::VLMModelType model_type) -> std::filesystem::path {
            return model_type == ov::genai::VLMModelType::MINICPM ? "embed_tokens.xml" : "openvino_text_embeddings_model.xml";
        };
        auto get_language_model_path = [] (ov::genai::VLMModelType model_type) -> std::filesystem::path {
            return model_type == ov::genai::VLMModelType::MINICPM ? "language_model.xml" : "openvino_language_model.xml";
        };

        m_embedding = ov::Core{}.compile_model(
            model_dir / get_embedding_model_path(m_vlm_config.model_type), device, device_config
        ).create_infer_request();

        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            m_vlm_config, model_dir, device, device_config,
            m_embedding, m_tokenizer,
            m_is_chat_conversation, m_history, m_templated_chat_history);

        m_language = ov::Core{}.compile_model(
            model_dir / get_language_model_path(m_vlm_config.model_type), device, device_config
        ).create_infer_request();

        m_language.get_tensor("attention_mask").set_shape({1, 0});
    }

    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) {
        ov::Tensor inputs_embeds = m_inputs_embedder->get_inputs_embeds(prompt, rgbs);

        m_language.set_tensor("inputs_embeds", inputs_embeds);
        size_t history_len = m_language.get_tensor("attention_mask").get_shape().at(1);
        m_language.get_tensor("attention_mask").set_shape({1, history_len + inputs_embeds.get_shape()[1]});
        std::fill_n(m_language.get_tensor("attention_mask").data<int64_t>(), m_language.get_tensor("attention_mask").get_size(), 1);
        
        m_language.get_tensor("position_ids").set_shape({1, inputs_embeds.get_shape().at(1)});
        std::iota(m_language.get_tensor("position_ids").data<int64_t>(), m_language.get_tensor("position_ids").data<int64_t>() + m_language.get_tensor("position_ids").get_size(), history_len);
        
        m_language.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
        m_language.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        m_language.infer();

        ov::Shape logits_shape = m_language.get_tensor("logits").get_shape();
        auto attention_size = m_language.get_tensor("attention_mask").get_size();

        int64_t sequence_len = m_language.get_tensor("logits").get_shape().at(1) - 1;
        size_t vocab_size = m_language.get_tensor("logits").get_shape().back();
        float* logits = m_language.get_tensor("logits").data<float>() + sequence_len * vocab_size;
        int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

        m_language.get_tensor("inputs_embeds").set_shape({BATCH_SIZE, 1, m_vlm_config.hidden_size});
        m_language.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

        m_embedding.get_input_tensor().set_shape({ 1, 1 });

        int64_t eos_token_id = m_tokenizer.get_eos_token_id();
        std::shared_ptr<StreamerBase> streamer_ptr = std::visit(overloaded{
            [&m_tokenizer = m_tokenizer](
                const std::function<bool(std::string)>& callback
            ) -> std::shared_ptr<StreamerBase> {
                return std::make_shared<TextCallbackStreamer>(m_tokenizer, callback);
            },
            [](const std::shared_ptr<StreamerBase>& ptr) {
                return ptr;
            },
            [](std::monostate) {
                return std::shared_ptr<StreamerBase>{nullptr};
            },
        }, streamer);
        std::vector<int64_t> generated;
        while (true) {  //(out_token != eos_token_id)
            m_embedding.get_input_tensor().data<int64_t>()[0] = out_token;
            m_embedding.infer();
            const ov::Tensor& embed_prompt_tensor = m_embedding.get_output_tensor();
            float* embed_data = embed_prompt_tensor.data<float>();
            for (auto idx = 0; idx < embed_prompt_tensor.get_size(); idx++) {
                embed_data[idx] = embed_data[idx] * m_vlm_config.scale_emb;
            }

            m_language.set_tensor("inputs_embeds", embed_prompt_tensor);
            m_language.get_tensor("attention_mask").set_shape({ BATCH_SIZE, m_language.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(m_language.get_tensor("attention_mask").data<int64_t>(), m_language.get_tensor("attention_mask").get_size(), 1);
            m_language.get_tensor("position_ids").data<int64_t>()[0] = int64_t(m_language.get_tensor("attention_mask").get_size() - 1);

            m_language.infer();

            generated.push_back(out_token);
            if (streamer_ptr && streamer_ptr->put(out_token)) {
                break;
            }
            logits = m_language.get_tensor("logits").data<float>();

            out_token = std::max_element(logits, logits + vocab_size) - logits;
            if (out_token == eos_token_id) {
                break;
            }
        }

        if (streamer_ptr) {
            streamer_ptr->end();
        }

        std::string decoded_results = m_tokenizer.decode(generated);
        if (m_is_chat_conversation) {
            // Tail of chat template is missing in KV cache.
            // Find the tail to concatenate it with the next input prompt.
            m_templated_chat_history.append(decoded_results);
            m_history.push_back({{"role", "assistant"}, {"content", decoded_results}});
        } else {
            for (auto& variable : m_language.query_state()) {
                variable.reset();
            }
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }
        return {{std::move(decoded_results)}};
    }

    DecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        OPENVINO_ASSERT(
            config_map.end() == image || config_map.end() == images,
            "Only one property can be set: image of images."
        );
        std::vector<ov::Tensor> rgbs;
        if (config_map.end() != image) {
            rgbs = {image->second.as<ov::Tensor>()};
        } if (config_map.end() != images) {
            rgbs = images->second.as<std::vector<ov::Tensor>>();
        }
        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);
        return generate(
            prompt,
            rgbs,
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    void start_chat(const std::string& system_message) {
        m_is_chat_conversation = true;
        bool have_state = 0 != m_language.get_tensor("attention_mask").get_size();
        if (have_state) {
            // Resetting state may be slow.
            for (ov::VariableState& variable : m_language.query_state()) {
                variable.reset();
            }
            // Since if is already introduced, move all resetting here.
            m_language.get_tensor("attention_mask").set_shape({1, 0});
            m_history.clear();
            m_templated_chat_history.clear();
        }
        if (system_message.empty()) {
            return;
        }
        m_history = {{{"role", "system"}, {"content", system_message}}};
        constexpr bool add_generation_prompt = false;
        m_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
    }

    void finish_chat() {m_is_chat_conversation = false;}

    Tokenizer get_tokenizer() const {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) {
        m_tokenizer.set_chat_template(new_template);
    }

    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& new_config) {
        m_generation_config = new_config;
    }
};

VLMPipeline::VLMPipeline(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : m_pimpl{std::make_unique<VLMPipelineImpl>(model_dir, device, device_config)} {}

ov::genai::VLMPipeline::~VLMPipeline() = default;

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& rgbs,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, rgbs, generation_config, streamer);
}

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(prompt, config_map);
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void VLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void VLMPipeline::set_chat_template(const std::string& new_template) {
    m_pimpl->set_chat_template(new_template);
}

Tokenizer VLMPipeline::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

GenerationConfig VLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

void VLMPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_pimpl->set_generation_config(new_config);
}
