// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <optional>
#include <string_view>
#include <system_error>
#include <tuple>

#include "decoder_model_split.hpp"
#include "openvino/core/model.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool isLanguageModelOnlyProperty(std::string_view name) {
    if (startsWith(name, "++")) {
        name.remove_prefix(2);
    }

    if (name == "MAX_PROMPT_LEN" || name == "MIN_RESPONSE_LEN" || name == "BLOB_PATH" || name == "EXPORT_BLOB") {
        return true;
    }

    return name == "NPU_USE_NPUW" || startsWith(name, "NPUW_") || startsWith(name, "PREFILL_") ||
           startsWith(name, "GENERATE_") || startsWith(name, "SHARED_");
}

ov::AnyMap getTextEmbeddingProperties(const ov::AnyMap& properties) {
    ov::AnyMap text_embedding_properties;
    for (const auto& [name, value] : properties) {
        if (!isLanguageModelOnlyProperty(name)) {
            text_embedding_properties.emplace(name, value);
        }
    }
    return text_embedding_properties;
}

std::optional<std::filesystem::path> resolveNpuDumpDir(const std::filesystem::path& models_path) {
    const char* flag = std::getenv("OV_GENAI_QWEN3ASR_NPU_DUMP");
    if (flag == nullptr) {
        return std::nullopt;
    }
    const std::string_view value{flag};
    if (value.empty() || value == "0" || value == "false" || value == "FALSE") {
        return std::nullopt;
    }
    auto dump_dir = models_path / "npu_static_dump";
    std::filesystem::create_directories(dump_dir);
    return dump_dir;
}

// Restores the process working directory on destruction so NPUW_DUMP_FULL
// output (written to the current directory) lands in the dump folder only
// while the language model is compiled.
class ScopedWorkingDirectory final {
public:
    explicit ScopedWorkingDirectory(const std::filesystem::path& directory)
        : _previous(std::filesystem::current_path()) {
        std::filesystem::current_path(directory);
    }

    ScopedWorkingDirectory(const ScopedWorkingDirectory&) = delete;
    ScopedWorkingDirectory& operator=(const ScopedWorkingDirectory&) = delete;

    ~ScopedWorkingDirectory() {
        std::error_code error_code;
        std::filesystem::current_path(_previous, error_code);
    }

private:
    std::filesystem::path _previous;
};

class SamplerRequestCleanup final {
public:
    SamplerRequestCleanup(Sampler& sampler, const std::vector<SequenceGroup::Ptr>& sequence_groups)
        : _sampler(sampler),
          _sequenceGroups(sequence_groups) {}

    SamplerRequestCleanup(const SamplerRequestCleanup&) = delete;
    SamplerRequestCleanup& operator=(const SamplerRequestCleanup&) = delete;

    ~SamplerRequestCleanup() {
        for (const auto& sequence_group : _sequenceGroups) {
            _sampler.clear_request_info(sequence_group->get_request_id());
        }
    }

private:
    Sampler& _sampler;
    const std::vector<SequenceGroup::Ptr>& _sequenceGroups;
};

}  // namespace

Qwen3ASRDecoder::Qwen3ASRDecoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model;
    if (device == "NPU") {
        m_is_npu = true;

        auto decoder_model = core.read_model(models_path / "openvino_decoder_model.xml");
        auto decoder_models = splitQwen3ASRDecoderModel(decoder_model);

        auto language_model_properties = utils::get_model_properties(properties, "language_model", "NPU");
        if (language_model_properties.count("PREFILL_HINT") == 0 &&
            language_model_properties.count("NPUW_LLM_PREFILL_HINT") == 0) {
            // Experimental A/B knob: default STATIC; env can switch to DYNAMIC chunked prefill.
            std::string prefill_hint = "STATIC";
            if (const char* override_hint = std::getenv("OV_GENAI_QWEN3ASR_PREFILL_HINT")) {
                if (override_hint[0] != '\0') {
                    prefill_hint = override_hint;
                }
            }
            language_model_properties.emplace("PREFILL_HINT", prefill_hint);
            if (const char* chunk_size = std::getenv("OV_GENAI_QWEN3ASR_PREFILL_CHUNK")) {
                if (chunk_size[0] != '\0') {
                    language_model_properties.emplace("NPUW_LLM_PREFILL_CHUNK_SIZE", std::string(chunk_size));
                }
            }
        }
        if (language_model_properties.count("GENERATE_HINT") == 0 &&
            language_model_properties.count("NPUW_LLM_GENERATE_HINT") == 0) {
            language_model_properties.emplace("GENERATE_HINT", "BEST_PERF");
        }
        if (language_model_properties.count("GENERATE_CONFIG") == 0 &&
            language_model_properties.count("NPUW_LLM_GENERATE_CONFIG") == 0 &&
            language_model_properties.count("++GENERATE_CONFIG") == 0 &&
            language_model_properties.count("++NPUW_LLM_GENERATE_CONFIG") == 0) {
            // Override NPUW's own BEST_PERF-hint default (NPUW_ONLINE_PIPELINE=NONE)
            // so the generate stage is online-partitioned into multiple subgraphs too,
            // matching the prefill stage's default (REP) behavior.
            language_model_properties.emplace(
                "++GENERATE_CONFIG", ov::AnyMap{{"NPUW_DQ", "YES"}, {"NPUW_ONLINE_PIPELINE", "REP"}, {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"}});
        }
        language_model_properties.emplace("NPUW_LLM_SHARED_HEAD", "NO");
        language_model_properties.emplace("NPUW_LLM_CACHE_ROPE", "NO");
        language_model_properties.emplace("NPUW_LLM_OPTIMIZE_V_TENSORS", "NO");
        language_model_properties.emplace("NPUW_DEVICES", "NPU");
        language_model_properties.emplace("NPUW_FALLBACK_EXEC", "NO");

        const auto npu_dump_dir = resolveNpuDumpDir(models_path);
        if (npu_dump_dir) {
            // Dump the statically-reshaped NPUW models. NPUW_DUMP_FULL writes the full
            // prefill/generate models to the current working directory (redirected below).
            // NPUW_DUMP_SUBS writes the online-partitioned subgraphs into a subfolder.
            language_model_properties.emplace("NPUW_DUMP_FULL", "YES");
            const auto partitions_dir = *npu_dump_dir / "partitions";
            std::filesystem::create_directories(partitions_dir);
            language_model_properties.emplace("NPUW_DUMP_SUBS", "YES");
            language_model_properties.emplace("NPUW_DUMP_SUBS_DIR", partitions_dir.string());
        }

        const auto kv_pos = utils::get_kv_axes_pos(decoder_models.languageModel);
        utils::KVDesc kv_desc;
        {
            std::optional<ScopedWorkingDirectory> dump_cwd;
            if (npu_dump_dir) {
                dump_cwd.emplace(*npu_dump_dir);
            }
            std::tie(compiled_model, kv_desc) =
                utils::compile_decoder_for_npu(decoder_models.languageModel, language_model_properties, kv_pos);
        }

        m_max_prompt_len = kv_desc.max_prompt_len;
        m_max_kv_cache_size =
            static_cast<size_t>(kv_desc.max_prompt_len) + static_cast<size_t>(kv_desc.min_response_len);
        m_audio_token_id = decoder_models.audioTokenId;
        m_hidden_size = decoder_models.hiddenSize;
        OPENVINO_ASSERT(m_max_prompt_len > 0, "Qwen3-ASR NPU decoder requires a positive maximum prompt length");
        OPENVINO_ASSERT(m_max_kv_cache_size >= m_max_prompt_len,
                        "Qwen3-ASR NPU decoder has an invalid KV-cache capacity");
        OPENVINO_ASSERT(m_audio_token_id >= 0, "Qwen3-ASR NPU decoder requires a valid audio token ID");
        OPENVINO_ASSERT(m_hidden_size > 0, "Qwen3-ASR NPU decoder requires a positive hidden size");

        const auto text_embedding_properties =
            getTextEmbeddingProperties(utils::get_model_properties(properties, "text_embeddings", "NPU"));

        // Prefill embeds the whole prompt in one shot, so it needs the full
        // [1, max_prompt_len] lookup.
        decoder_models.textEmbedding->reshape(
            {{"input_ids", ov::PartialShape{1, static_cast<int64_t>(m_max_prompt_len)}}});
        if (npu_dump_dir) {
            ov::save_model(decoder_models.textEmbedding,
                           (*npu_dump_dir / "npu_text_embedding_static.xml").string());
        }
        auto compiled_text_embedding =
            core.compile_model(decoder_models.textEmbedding, "NPU", text_embedding_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_text_embedding,
                                                          "qwen3-asr text embedding model (prefill)");
        m_text_embedding_request = compiled_text_embedding.create_infer_request();
        m_text_embedding_input = ov::Tensor(ov::element::i64, {1, m_max_prompt_len});
        // Untouched positions must always contain a valid vocabulary ID.
        std::fill_n(m_text_embedding_input.data<int64_t>(), m_max_prompt_len, 0);
        m_text_embedding_request.set_tensor("input_ids", m_text_embedding_input);

        // Decode embeds a single new token per step. A dedicated [1, 1] lookup avoids
        // recomputing all max_prompt_len embedding rows and discarding all but the last.
        decoder_models.textEmbedding->reshape({{"input_ids", ov::PartialShape{1, 1}}});
        if (npu_dump_dir) {
            ov::save_model(decoder_models.textEmbedding,
                           (*npu_dump_dir / "npu_text_embedding_decode_static.xml").string());
        }
        auto compiled_text_embedding_decode =
            core.compile_model(decoder_models.textEmbedding, "NPU", text_embedding_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_text_embedding_decode,
                                                          "qwen3-asr text embedding model (decode)");
        m_text_embedding_decode_request = compiled_text_embedding_decode.create_infer_request();
        m_text_embedding_decode_input = ov::Tensor(ov::element::i64, {1, 1});
        m_text_embedding_decode_input.data<int64_t>()[0] = 0;
        m_text_embedding_decode_request.set_tensor("input_ids", m_text_embedding_decode_input);
    } else {
        compiled_model = core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);
    }
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr decoder model");
    m_request = compiled_model.create_infer_request();
}

void Qwen3ASRDecoder::set_seed(size_t seed) {
    m_sampler.set_seed(seed);
}

EncodedResults Qwen3ASRDecoder::generate(const ov::Tensor& input_ids,
                                         const ov::Tensor& encoder_hidden_states,
                                         const ASRGenerationConfig& config,
                                         RawPerfMetrics& raw_metrics,
                                         ASRRawPerfMetrics& asr_raw_metrics,
                                         const std::shared_ptr<StreamerBase>& streamer_ptr) {
    const ov::Shape prompts_shape = input_ids.get_shape();
    if (m_is_npu) {
        OPENVINO_ASSERT(prompts_shape.size() == 2, "Qwen3-ASR NPU input_ids must be rank 2");
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64,
                        "Qwen3-ASR NPU input_ids must have i64 element type");
    }
    const size_t batch_size = prompts_shape[0];
    OPENVINO_ASSERT(batch_size == 1 || !streamer_ptr, "Streaming is only supported with batch_size == 1");
    if (m_is_npu) {
        OPENVINO_ASSERT(batch_size == 1, "Qwen3-ASR NPU decoder only supports batch size 1");
        OPENVINO_ASSERT(!config.is_beam_search(), "Qwen3-ASR NPU decoder does not support beam search");
    }

    // Reset decoder state for fresh generation
    m_request.reset_state();

    std::vector<SequenceGroup::Ptr> sequence_groups;
    sequence_groups.reserve(batch_size);
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const size_t prompt_len = prompts_shape[1];
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::vector<int64_t> prompt_tokens(input_ids_data + batch * prompt_len,
                                           input_ids_data + (batch + 1) * prompt_len);
        auto seq_group = std::make_shared<SequenceGroup>(batch, prompt_tokens, config);
        sequence_groups.push_back(seq_group);
    }
    SamplerRequestCleanup sampler_request_cleanup(m_sampler, sequence_groups);

    // Streaming handle (only for batch_size == 1)
    std::shared_ptr<GenerationHandleImpl> handle;
    if (streamer_ptr) {
        handle = std::make_shared<GenerationHandleImpl>(sequence_groups[0]->get_generation_stream(),
                                                        sequence_groups[0]->get_sampling_parameters());
    }

    auto stream_generated_tokens = [&]() {
        if (!streamer_ptr || !handle || !handle->can_read()) {
            return;
        }
        std::unordered_map<uint64_t, GenerationOutput> token = handle->read();
        auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
        if (streaming_status == StreamingStatus::CANCEL) {
            handle->cancel();
        } else if (streaming_status == StreamingStatus::STOP) {
            handle->stop();
        }
    };

    ov::Tensor current_encoder_hidden_states = encoder_hidden_states;
    if (!m_is_npu) {
        m_request.set_tensor("encoder_hidden_states", current_encoder_hidden_states);
    }

    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);
    m_request.set_tensor("beam_idx", beam_idx);

    // Prefill: run decoder with full prompt
    size_t context_len = prompt_len;
    std::chrono::steady_clock::time_point infer_start;
    if (m_is_npu) {
        infer_start = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(prompt_len > 0, "Qwen3-ASR NPU prompt must contain at least one token");
        OPENVINO_ASSERT(prompt_len <= m_max_prompt_len,
                        "Qwen3-ASR NPU prompt length ",
                        prompt_len,
                        " exceeds the configured maximum ",
                        m_max_prompt_len);

        auto* static_input_ids_data = m_text_embedding_input.data<int64_t>();
        // The embedding model has a static sequence length on NPU, so place the
        // prompt at the end. Earlier valid IDs may be stale because their output
        // rows are not consumed.
        const size_t prompt_offset = m_max_prompt_len - prompt_len;
        std::copy_n(input_ids_data, prompt_len, static_input_ids_data + prompt_offset);

        m_text_embedding_request.infer();
        const ov::Tensor static_embeddings = m_text_embedding_request.get_tensor("inputs_embeds");
        const ov::Shape static_embeddings_shape = static_embeddings.get_shape();
        OPENVINO_ASSERT(static_embeddings.get_element_type() == ov::element::f32,
                        "Qwen3-ASR text embeddings must have f32 element type");
        OPENVINO_ASSERT(static_embeddings_shape.size() == 3 && static_embeddings_shape[0] == 1 &&
                            static_embeddings_shape[1] == m_max_prompt_len &&
                            static_embeddings_shape[2] == m_hidden_size,
                        "Unexpected Qwen3-ASR static text embedding shape: ",
                        static_embeddings_shape);

        const ov::Shape encoder_shape = encoder_hidden_states.get_shape();
        OPENVINO_ASSERT(encoder_hidden_states.get_element_type() == ov::element::f32,
                        "Qwen3-ASR encoder hidden states must have f32 element type");
        OPENVINO_ASSERT(encoder_shape.size() == 3 && encoder_shape[0] == 1 && encoder_shape[2] == m_hidden_size,
                        "Unexpected Qwen3-ASR encoder hidden state shape: ",
                        encoder_shape);

        const size_t audio_token_count = std::count(input_ids_data, input_ids_data + prompt_len, m_audio_token_id);
        OPENVINO_ASSERT(audio_token_count == encoder_shape[1],
                        "Qwen3-ASR audio placeholder count ",
                        audio_token_count,
                        " does not match encoder token count ",
                        encoder_shape[1]);

        ov::Tensor merged_embeddings(ov::element::f32, {1, prompt_len, m_hidden_size});
        auto* merged_embeddings_data = merged_embeddings.data<float>();
        const auto* static_embeddings_data = static_embeddings.data<const float>();
        std::memcpy(merged_embeddings_data,
                    static_embeddings_data + prompt_offset * m_hidden_size,
                    prompt_len * m_hidden_size * sizeof(float));

        const auto* encoder_data = encoder_hidden_states.data<const float>();
        size_t encoder_token_index = 0;
        // Replace audio placeholders with encoder rows in prompt order.
        for (size_t prompt_index = 0; prompt_index < prompt_len; ++prompt_index) {
            if (input_ids_data[prompt_index] != m_audio_token_id) {
                continue;
            }
            std::memcpy(merged_embeddings_data + prompt_index * m_hidden_size,
                        encoder_data + encoder_token_index * m_hidden_size,
                        m_hidden_size * sizeof(float));
            ++encoder_token_index;
        }
        OPENVINO_ASSERT(encoder_token_index == audio_token_count,
                        "Qwen3-ASR did not consume all encoder hidden states");

        ov::Tensor attention_mask(ov::element::i64, {1, prompt_len});
        std::fill_n(attention_mask.data<int64_t>(), prompt_len, 1);
        ov::Tensor position_ids(ov::element::i64, {1, prompt_len});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + prompt_len, int64_t{0});

        m_request.set_tensor("inputs_embeds", merged_embeddings);
        m_request.set_tensor("attention_mask", attention_mask);
        m_request.set_tensor("position_ids", position_ids);
    } else {
        m_request.set_tensor("input_ids", input_ids);
        infer_start = std::chrono::steady_clock::now();
    }
    m_request.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(batch_size);
    asr_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

    ov::Tensor logits = m_request.get_tensor("logits");
    const int64_t output_sequence_len = logits.get_shape().at(1);

    // Schedule prompt tokens and sample
    for (auto& seq_group : sequence_groups) {
        seq_group->schedule_tokens(seq_group->get_prompt_len());
        seq_group->set_output_seq_len(output_sequence_len);
    }

    // Beam offsets: maps request_id -> starting position in flattened batch
    std::map<size_t, size_t> beam_offsets;
    for (size_t i = 0; i < sequence_groups.size(); ++i) {
        beam_offsets.insert({sequence_groups[i]->get_request_id(), i});
    }

    const auto sample_start = std::chrono::steady_clock::now();
    m_sampler.sample(sequence_groups, logits);
    raw_metrics.m_sampling_durations.emplace_back(
        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    stream_generated_tokens();

    // Track active (not yet finished) sequence groups
    auto active_sequence_groups = sequence_groups;

    auto free_finished_requests = [this, &active_sequence_groups]() {
        if (m_is_npu) {
            for (const auto& sequence_group : active_sequence_groups) {
                if (sequence_group->is_running() && sequence_group->get_num_processed_tokens() >= m_max_kv_cache_size) {
                    sequence_group->set_out_of_memory();
                }
            }
        }

        auto removed_it =
            std::remove_if(active_sequence_groups.begin(),
                           active_sequence_groups.end(),
                           [](const SequenceGroup::Ptr& sg) {
                               return sg->has_finished() || sg->handle_stopped() || sg->handle_cancelled();
                           });
        active_sequence_groups.erase(removed_it, active_sequence_groups.end());
    };

    free_finished_requests();

    // Generation loop
    while (!active_sequence_groups.empty()) {
        size_t total_num_tokens = 0;
        for (auto& seq_group : active_sequence_groups) {
            seq_group->schedule_tokens(1);
            total_num_tokens += seq_group->get_num_scheduled_tokens() * seq_group->num_running_seqs();
        }

        ov::Tensor new_input_ids(ov::element::i64, {total_num_tokens, 1});
        int64_t* input_ids_data = new_input_ids.data<int64_t>();
        std::vector<int32_t> next_beams;

        for (auto& seq_group : active_sequence_groups) {
            std::vector<Sequence::Ptr> running_sequences = seq_group->get_running_sequences();
            const size_t num_scheduled_tokens = seq_group->get_num_scheduled_tokens();
            const size_t group_position_id = seq_group->get_num_processed_tokens();

            std::map<size_t, int32_t> beam_idxs = m_sampler.get_beam_idxs(seq_group);

            for (size_t seq_id = 0; seq_id < running_sequences.size(); ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens;
                     ++token_id, ++position_id) {
                    input_ids_data[token_id] =
                        position_id < seq_group->get_prompt_len()
                            ? seq_group->get_prompt_ids()[position_id]
                            : sequence->get_generated_ids()[position_id - seq_group->get_prompt_len()];
                }

                input_ids_data += num_scheduled_tokens;
                next_beams.push_back(beam_idxs[sequence->get_id()] + beam_offsets.at(seq_group->get_request_id()));
            }
        }

        // Update beam offsets for next iteration
        for (size_t i = 0; i < active_sequence_groups.size(); ++i) {
            beam_offsets[active_sequence_groups[i]->get_request_id()] =
                i == 0 ? 0
                       : (active_sequence_groups[i - 1]->num_running_seqs() +
                          beam_offsets[active_sequence_groups[i - 1]->get_request_id()]);
        }

        std::chrono::steady_clock::time_point infer_start;
        if (m_is_npu) {
            infer_start = std::chrono::steady_clock::now();
            OPENVINO_ASSERT(total_num_tokens == 1 && next_beams.size() == 1 && next_beams.front() == 0,
                            "Qwen3-ASR NPU decoder only supports one non-beam sequence");

            m_text_embedding_decode_input.data<int64_t>()[0] = new_input_ids.data<const int64_t>()[0];

            // Dedicated [1, 1] embedding lookup: only the single new token is embedded.
            m_text_embedding_decode_request.infer();
            const ov::Tensor static_embeddings = m_text_embedding_decode_request.get_tensor("inputs_embeds");
            const ov::Shape static_embeddings_shape = static_embeddings.get_shape();
            OPENVINO_ASSERT(static_embeddings.get_element_type() == ov::element::f32 &&
                                static_embeddings_shape.size() == 3 && static_embeddings_shape[0] == 1 &&
                                static_embeddings_shape[1] == 1 && static_embeddings_shape[2] == m_hidden_size,
                            "Unexpected Qwen3-ASR decode token embedding tensor");

            ov::Tensor token_embedding(ov::element::f32, {1, 1, m_hidden_size});
            std::memcpy(token_embedding.data<float>(),
                        static_embeddings.data<const float>(),
                        m_hidden_size * sizeof(float));

            ++context_len;
            ov::Tensor attention_mask(ov::element::i64, {1, context_len});
            std::fill_n(attention_mask.data<int64_t>(), context_len, 1);
            ov::Tensor position_ids(ov::element::i64, {1, 1});
            position_ids.data<int64_t>()[0] = static_cast<int64_t>(context_len - 1);

            m_request.set_tensor("inputs_embeds", token_embedding);
            m_request.set_tensor("attention_mask", attention_mask);
            m_request.set_tensor("position_ids", position_ids);
        } else {
            m_request.set_tensor("input_ids", new_input_ids);
        }
        m_request.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {total_num_tokens}, next_beams.data()});
        // for beam search investigate encoder batches reordering based on
        // next_beams
        if (!m_is_npu) {
            infer_start = std::chrono::steady_clock::now();
        }
        m_request.infer();
        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
        raw_metrics.m_new_token_times.emplace_back(infer_end);
        raw_metrics.m_batch_sizes.emplace_back(total_num_tokens);
        asr_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

        logits = m_request.get_tensor("logits");

        const auto sample_start = std::chrono::steady_clock::now();
        m_sampler.sample(active_sequence_groups, logits);
        raw_metrics.m_sampling_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        stream_generated_tokens();
        free_finished_requests();
    }

    // Flush streamer cache
    stream_generated_tokens();

    // Collect results
    EncodedResults results;
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& sequences = sequence_groups[b]->get_finished_sequences();
        OPENVINO_ASSERT(!sequences.empty(), "No finished sequences for batch element ", b);

        const auto& sampling_params = sequence_groups[b]->get_sampling_parameters();
        const auto& sequence = sequences[0];
        const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params)
                                                             : sequence->get_cumulative_log_prob();

        results.tokens.push_back(sequence->get_generated_ids());
        results.scores.push_back(score);
    }
    return results;
}

}  // namespace ov::genai
