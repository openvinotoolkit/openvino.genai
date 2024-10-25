// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <random>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/image_embedder.hpp"
#include "visual_language/embedding_model.hpp"

#include "sampler.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {
   
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

constexpr size_t BATCH_SIZE = 1;

EncodedGenerationResult get_lm_encoded_results(
    ov::InferRequest& language,
    std::shared_ptr<EmbeddingsModel>& embeding,
    const ov::Tensor& inputs_embeds,
    const std::shared_ptr<StreamerBase>& streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> requests
) {
    SequenceGroup::Ptr request = requests.back();
    GenerationHandle generation = std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), request->get_sampling_parameters());

    language.set_tensor("inputs_embeds", inputs_embeds);

    size_t history_len = language.get_tensor("attention_mask").get_shape().at(1);
    language.get_tensor("attention_mask").set_shape({1, history_len + inputs_embeds.get_shape()[1]});
    std::fill_n(language.get_tensor("attention_mask").data<int64_t>(), language.get_tensor("attention_mask").get_size(), 1);

    language.get_tensor("position_ids").set_shape({1, inputs_embeds.get_shape().at(1)});
    std::iota(language.get_tensor("position_ids").data<int64_t>(), language.get_tensor("position_ids").data<int64_t>() + language.get_tensor("position_ids").get_size(), history_len);

    language.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
    language.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    language.infer();

    int64_t sequence_len = language.get_tensor("logits").get_shape().at(1);
    request->schedule_tokens(sequence_len);

    sampler.sample(requests, language.get_tensor("logits"));

    auto hidden_size = inputs_embeds.get_shape()[2];
    language.get_tensor("inputs_embeds").set_shape({BATCH_SIZE, 1, hidden_size});
    language.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

    while (!request->has_finished()) {
        request->schedule_tokens(1);
        size_t num_sequences = request->num_running_seqs();
        size_t total_num_tokens = request->get_num_scheduled_tokens() * num_sequences;

        ov::Tensor
            input_ids(ov::element::i64, {total_num_tokens, 1}),
            position_ids(ov::element::i64, {total_num_tokens, 1}),
            beam_idx(ov::element::i32, { total_num_tokens });

        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>();

        size_t num_scheduled_tokens = request->get_num_scheduled_tokens();
        size_t group_position_id = request->get_num_processed_tokens();
        for (Sequence::Ptr& sequence : request->get_running_sequences()) {
            for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                // compute token for current sequence
                input_ids_data[token_id] = position_id < request->get_prompt_len() ?
                    request->get_prompt_ids()[position_id] :
                    sequence->get_generated_ids()[position_id - request->get_prompt_len()];

                position_ids_data[token_id] = position_id;
            }
            // apply strides to shift to a next sequence
            input_ids_data += num_scheduled_tokens;
            position_ids_data += num_scheduled_tokens;
        }
        
        const ov::Tensor& embed_prompt_tensor = embeding->infer(input_ids);

        language.set_tensor("inputs_embeds", embed_prompt_tensor);

        language.get_tensor("attention_mask").set_shape({ total_num_tokens, language.get_tensor("attention_mask").get_shape()[1] + 1 });
        std::fill_n(language.get_tensor("attention_mask").data<int64_t>(), language.get_tensor("attention_mask").get_size(), 1);

        language.set_tensor("position_ids", position_ids);

        std::vector<int32_t> beam_idxs = sampler.get_beam_idxs(request);
        int32_t *beam_idx_data = beam_idx.data<int32_t>();
        copy(beam_idxs.begin(), beam_idxs.end(), beam_idx_data);
        language.set_tensor("beam_idx", beam_idx);

        language.infer();

        if (streamer_ptr) {
            // first sequence
            int64_t out_token = request.get()->operator[](0)->get_generated_ids().back();
            if (streamer_ptr->put(out_token)) {
                break;
            }
        }

        sampler.sample(requests, language.get_tensor("logits"));
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    EncodedGenerationResult result;
    result.m_request_id = 1;
    std::vector<GenerationOutput> generation_outputs = generation->read_all();
    std::sort(generation_outputs.begin(), generation_outputs.end(), [] (const GenerationOutput& r1, const GenerationOutput& r2) {
        return r1.score > r2.score;
    });

    auto num_outputs = std::min(request->get_sampling_parameters().num_return_sequences, generation_outputs.size());
    for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
        const auto& generation_output = generation_outputs[generation_output_idx];
        result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
        result.m_scores.push_back(generation_output.score);
    }
    result.m_status = generation->get_status();

    return result;
}

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
    std::shared_ptr<EmbeddingsModel> m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    // InputsEmbedder
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;

    VLMPipelineImpl(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) :
        m_vlm_config{
            utils::from_config_json_if_exists<ov::genai::VLMConfig>(
                models_dir, "config.json"
            )
        },
        m_is_chat_conversation{false} {
        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            m_vlm_config, models_dir, device, properties);

        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        m_language = utils::singleton_core().compile_model(
            models_dir / "openvino_language_model.xml", device, properties
        ).create_infer_request();

        m_language.get_tensor("attention_mask").set_shape({1, 0});
    }

    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) {
        // If eos_token_id was not provided, take value
        if (generation_config.eos_token_id == -1) {
            generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        ov::Tensor inputs_embeds = m_inputs_embedder->get_inputs_embeds(prompt, rgbs);

        Sampler sampler = Sampler(m_tokenizer);

        std::vector<SequenceGroup::Ptr> requests;
        size_t request_id = 0;
        size_t block_size = 1; // not used
        bool enable_prefix_caching = false;
        size_t history_size = m_language.get_tensor("attention_mask").get_shape().at(1);
        size_t inputs_embeds_size = inputs_embeds.get_shape().at(1);
        ov::Tensor prompt_ids(ov::element::i64, { history_size + inputs_embeds_size });
        std::fill_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), 0);

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, prompt_ids, generation_config, block_size, enable_prefix_caching);
        sequence_group->update_processed_tokens_num(history_size);
        sequence_group->set_sequence_group_ptr(sequence_group);
        requests.push_back(sequence_group);

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

        OPENVINO_ASSERT((generation_config.is_greedy_decoding() || generation_config.is_multinomial() || !streamer_ptr),
                        "Currently streaming is possible only for greedy or multinomial decoding");

        EncodedGenerationResult encoded_result = get_lm_encoded_results(m_language, m_embedding, inputs_embeds, streamer_ptr, sampler, requests);

        DecodedResults decoded;
        for (size_t idx = 0; idx < encoded_result.m_generation_ids.size(); ++idx) {
            decoded.texts.push_back(m_tokenizer.decode(encoded_result.m_generation_ids.at(idx)));
            decoded.scores.push_back(encoded_result.m_scores.at(idx));
        }

        std::string decoded_results = decoded.texts.at(0);
        if (m_is_chat_conversation) {
            m_inputs_embedder->update_chat_history(decoded_results);
        } else {
            m_language.reset_state();
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }
        return decoded;
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
            m_language.reset_state();
            // Since if is already introduced, move all resetting here.
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }
        m_inputs_embedder->start_chat(system_message);
    }

    void finish_chat() {
        m_is_chat_conversation = false;
        // Resetting state may be slow.
        m_language.reset_state();
        // clear all chat history
        m_inputs_embedder->finish_chat();
    }

    Tokenizer get_tokenizer() const {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) {
        OPENVINO_ASSERT(!m_is_chat_conversation, "Chat template cannot be changed once start_chat() is called. Please, finish current chat via finish_chat()");
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
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) : m_pimpl{std::make_unique<VLMPipelineImpl>(models_dir, device, properties)} {}

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
