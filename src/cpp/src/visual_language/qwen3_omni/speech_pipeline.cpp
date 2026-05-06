// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_omni/speech_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>

#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace {

ov::genai::StreamingStatus invoke_audio_streamer(const ov::genai::AudioStreamerVariant& streamer,
                                                 const ov::Tensor& chunk) {
    return std::visit(
        [&chunk](auto&& arg) -> ov::genai::StreamingStatus {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::function<ov::genai::StreamingStatus(ov::Tensor)>>) {
                return arg(chunk);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::AudioStreamerBase>>) {
                return arg->write(chunk);
            } else {
                return ov::genai::StreamingStatus::RUNNING;
            }
        },
        streamer);
}

void end_audio_streamer(const ov::genai::AudioStreamerVariant& streamer) {
    std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::AudioStreamerBase>>) {
                arg->end();
            }
        },
        streamer);
}

bool is_audio_streamer_active(const ov::genai::AudioStreamerVariant& streamer) {
    return !std::holds_alternative<std::monostate>(streamer);
}

}  // namespace

namespace ov::genai {

// --- Qwen3OmniSpeechConfig ---

Qwen3OmniSpeechConfig Qwen3OmniSpeechConfig::from_vlm_config(const VLMConfig& config) {
    Qwen3OmniSpeechConfig sc;
    sc.codec_bos_id = config.talker_codec_bos_id;
    sc.codec_eos_token_id = config.talker_codec_eos_token_id;
    sc.codec_pad_id = config.talker_codec_pad_id;
    sc.codec_nothink_id = config.talker_codec_nothink_id;
    sc.codec_think_bos_id = config.talker_codec_think_bos_id;
    sc.codec_think_eos_id = config.talker_codec_think_eos_id;
    sc.tts_bos_token_id = config.tts_bos_token_id;
    sc.tts_eos_token_id = config.tts_eos_token_id;
    sc.tts_pad_token_id = config.tts_pad_token_id;
    sc.im_start_token_id = config.im_start_token_id;
    sc.im_end_token_id = config.im_end_token_id;
    sc.system_token_id = config.system_token_id;
    sc.user_token_id = config.user_token_id;
    sc.assistant_token_id = config.assistant_token_id;
    sc.audio_token_id = config.audio_token_id;
    sc.image_token_id = config.image_token_id;
    sc.video_token_id = config.video_token_id;
    sc.num_code_groups = config.talker_num_code_groups;
    sc.thinker_hidden_size = config.talker_thinker_hidden_size;
    sc.speaker_ids = config.speaker_ids;
    return sc;
}

// --- Qwen3OmniSpeechPipeline ---

Qwen3OmniSpeechPipeline::Qwen3OmniSpeechPipeline(const std::filesystem::path& model_dir,
                                                 const VLMConfig& config,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties)
    : m_config(Qwen3OmniSpeechConfig::from_vlm_config(config)) {
    // Load all 6 speech sub-models (all optional)
    auto load_model = [&](const std::string& filename) -> ov::InferRequest {
        auto path = model_dir / (filename + ".xml");
        if (!std::filesystem::exists(path)) {
            return {};
        }
        auto model = utils::singleton_core().read_model(path);
        auto compiled = utils::singleton_core().compile_model(model, device, properties);
        return compiled.create_infer_request();
    };

    // Load thinker text embeddings for TTS special token embedding
    m_thinker_text_embeddings = load_model("openvino_text_embeddings_model");

    m_talker = load_model("openvino_talker_model");
    m_talker_text_embeddings = load_model("openvino_talker_text_embeddings_model");
    m_talker_projections = load_model("openvino_talker_projections_model");
    m_code_predictor = load_model("openvino_code_predictor_model");
    m_code2wav = load_model("openvino_code2wav_model");

    // All speech models must be present for speech generation
    m_talker_available = m_talker && m_talker_text_embeddings && m_talker_projections && m_code_predictor &&
                         m_code2wav && m_thinker_text_embeddings;

    if (m_talker_available) {
        auto output_pshape = m_talker_projections.get_compiled_model().output("text_projection").get_partial_shape();
        OPENVINO_ASSERT(output_pshape.rank().is_static() && output_pshape.rank().get_length() >= 2,
                        "Talker text projection output must have at least 2 dimensions");
        auto last_dim = output_pshape[output_pshape.rank().get_length() - 1];
        OPENVINO_ASSERT(last_dim.is_static(), "Talker text projection output last dimension must be static");
        m_config.talker_hidden_size = static_cast<size_t>(last_dim.get_length());
        OPENVINO_ASSERT(m_config.talker_hidden_size > 0,
                        "Failed to detect talker hidden size from text projection model");

        // Pre-allocate scratch buffers that are reused across generate_speech() calls
        m_cp_embed_sum = ov::Tensor(ov::element::f32, {1, 1, m_config.talker_hidden_size});

        if (m_config.speaker_ids.empty()) {
            m_talker_available = false;
        }

        // Load talker generation parameters from generation_config.json
        auto gen_config_path = model_dir / "generation_config.json";
        if (std::filesystem::exists(gen_config_path)) {
            std::ifstream f(gen_config_path);
            auto gen_data = nlohmann::json::parse(f);
            utils::read_json_param(gen_data, "talker_temperature", m_config.talker_temperature);
            utils::read_json_param(gen_data, "talker_top_k", m_config.talker_top_k);
            utils::read_json_param(gen_data, "talker_repetition_penalty", m_config.talker_repetition_penalty);
            utils::read_json_param(gen_data, "talker_max_new_tokens", m_config.talker_max_new_tokens);
            GENAI_INFO("Speech: talker params: temp=%.2f, top_k=%zu, rep_penalty=%.2f, max_tokens=%zu",
                       m_config.talker_temperature,
                       m_config.talker_top_k,
                       m_config.talker_repetition_penalty,
                       m_config.talker_max_new_tokens);
        }

        // Detect vocab_size from talker logits output and build suppress_tokens list
        // Suppress tokens in [vocab_size-1024, vocab_size) except codec_eos_token_id
        auto logits_pshape = m_talker.get_compiled_model().output("logits").get_partial_shape();
        if (logits_pshape.rank().is_static()) {
            auto vocab_dim = logits_pshape[logits_pshape.rank().get_length() - 1];
            if (vocab_dim.is_static()) {
                m_config.talker_vocab_size = static_cast<size_t>(vocab_dim.get_length());
                auto suppress_start = static_cast<int64_t>(m_config.talker_vocab_size) - 1024;
                if (suppress_start < 0)
                    suppress_start = 0;
                for (int64_t i = suppress_start; i < static_cast<int64_t>(m_config.talker_vocab_size); i++) {
                    if (i != m_config.codec_eos_token_id) {
                        m_config.talker_suppress_tokens.push_back(i);
                    }
                }
                GENAI_INFO("Speech: suppressing %zu special tokens in range [%lld, %zu)",
                           m_config.talker_suppress_tokens.size(),
                           (long long)suppress_start,
                           m_config.talker_vocab_size);
            }
        }

        // Load CodePredictor codec_embedding weights (exported by optimum-intel)
        auto cp_embed_path = model_dir / "code_predictor_codec_embedding.npy";
        if (std::filesystem::exists(cp_embed_path)) {
            std::ifstream npy_file(cp_embed_path, std::ios::binary);
            OPENVINO_ASSERT(npy_file.good(), "Failed to open ", cp_embed_path.string());

            // Parse .npy header
            char magic[6];
            npy_file.read(magic, 6);
            OPENVINO_ASSERT(std::string(magic, 6) == std::string("\x93NUMPY", 6), "Invalid npy magic");
            uint8_t major_version, minor_version;
            npy_file.read(reinterpret_cast<char*>(&major_version), 1);
            npy_file.read(reinterpret_cast<char*>(&minor_version), 1);
            OPENVINO_ASSERT(major_version == 1 && minor_version == 0,
                            "Only NumPy format v1.0 supported, got v",
                            (int)major_version,
                            ".",
                            (int)minor_version);
            unsigned short header_len;
            npy_file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
            std::string header(header_len, ' ');
            npy_file.read(&header[0], header_len);

            // Parse shape from header
            auto shape_start = header.find("'shape': (");
            OPENVINO_ASSERT(shape_start != std::string::npos, "No shape in npy header");
            shape_start += 10;
            auto shape_end = header.find(')', shape_start);
            OPENVINO_ASSERT(shape_end != std::string::npos, "Malformed npy header: no closing parenthesis for shape");
            auto shape_str = header.substr(shape_start, shape_end - shape_start);

            ov::Shape shape;
            std::istringstream ss(shape_str);
            std::string dim;
            while (std::getline(ss, dim, ',')) {
                dim.erase(std::remove(dim.begin(), dim.end(), ' '), dim.end());
                if (!dim.empty())
                    shape.push_back(std::stoull(dim));
            }

            OPENVINO_ASSERT(shape.size() == 3, "Expected 3D codec_embedding weights, got ", shape.size(), "D");
            auto shape_product = ov::shape_size(shape);
            auto data_size = shape_product * sizeof(float);
            OPENVINO_ASSERT(data_size <= 1024ULL * 1024 * 1024,
                            "NPY file shape too large: ",
                            data_size / (1024 * 1024),
                            " MB");
            m_cp_codec_embeddings = ov::Tensor(ov::element::f32, shape);
            npy_file.read(reinterpret_cast<char*>(m_cp_codec_embeddings.data()), data_size);
            OPENVINO_ASSERT(npy_file.gcount() == static_cast<std::streamsize>(data_size),
                            "NPY file truncated: expected ",
                            data_size,
                            " bytes, read ",
                            npy_file.gcount());
            m_cp_vocab_size = shape[1];
            m_cp_hidden_size = shape[2];

            m_has_cp_embeds = true;
            GENAI_INFO("Speech: loaded codec_embedding [%zu, %zu, %zu] from %s",
                       shape[0],
                       shape[1],
                       shape[2],
                       cp_embed_path.string().c_str());
        } else {
            GENAI_INFO("Speech: codec_embedding.npy not found, using talker embeddings as fallback");
        }
    }
}

int64_t Qwen3OmniSpeechPipeline::resolve_speaker_id(const std::string& speaker) const {
    if (!speaker.empty() && !m_config.speaker_ids.empty()) {
        std::string lower_speaker = speaker;
        std::transform(lower_speaker.begin(), lower_speaker.end(), lower_speaker.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        for (const auto& [name, id] : m_config.speaker_ids) {
            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
            if (lower_name == lower_speaker) {
                return id;
            }
        }
    }
    // Fallback: use first available speaker
    if (!m_config.speaker_ids.empty()) {
        return m_config.speaker_ids.begin()->second;
    }
    OPENVINO_THROW("No speaker IDs available in model config. "
                   "Ensure 'talker_config.speaker_id' is defined in config.json");
}

ov::Tensor Qwen3OmniSpeechPipeline::embed_thinker_token(int64_t token_id) {
    ov::Tensor input(ov::element::i64, {1, 1});
    input.data<int64_t>()[0] = token_id;
    m_thinker_text_embeddings.set_tensor("input", input);
    m_thinker_text_embeddings.infer();
    auto result = m_thinker_text_embeddings.get_tensor("inputs_embeds");
    ov::Tensor copy(result.get_element_type(), result.get_shape());
    result.copy_to(copy);
    return copy;
}

ov::Tensor Qwen3OmniSpeechPipeline::embed_talker_token(int64_t token_id) {
    auto map_it = m_embedding_lru_map.find(token_id);
    if (map_it != m_embedding_lru_map.end()) {
        // Move to front (most recently used)
        m_embedding_lru_list.splice(m_embedding_lru_list.begin(), m_embedding_lru_list, map_it->second);
        return map_it->second->second;  // ref-counted handle, not a data copy
    }

    ov::Tensor input(ov::element::i64, {1, 1});
    input.data<int64_t>()[0] = token_id;
    m_talker_text_embeddings.set_tensor("input", input);
    m_talker_text_embeddings.infer();
    auto result = m_talker_text_embeddings.get_tensor("inputs_embeds");

    // One copy from inference output to owned cache entry
    ov::Tensor cached(result.get_element_type(), result.get_shape());
    result.copy_to(cached);

    // Evict least recently used if cache is full
    if (m_embedding_lru_list.size() >= kMaxEmbeddingCacheSize) {
        auto& evicted = m_embedding_lru_list.back();
        m_embedding_lru_map.erase(evicted.first);
        m_embedding_lru_list.pop_back();
    }

    m_embedding_lru_list.emplace_front(token_id, cached);
    m_embedding_lru_map[token_id] = m_embedding_lru_list.begin();
    return cached;
}

const float* Qwen3OmniSpeechPipeline::cp_token_weights(size_t step, int64_t code) const {
    OPENVINO_ASSERT(step < m_cp_codec_embeddings.get_shape()[0], "CP embed step ", step, " out of range");
    OPENVINO_ASSERT(code >= 0 && static_cast<size_t>(code) < m_cp_vocab_size,
                    "CP embed code ",
                    code,
                    " out of range [0, ",
                    m_cp_vocab_size,
                    ")");

    const auto* weights = m_cp_codec_embeddings.data<const float>();
    return weights + (step * m_cp_vocab_size + code) * m_cp_hidden_size;
}

ov::Tensor Qwen3OmniSpeechPipeline::project_text(const ov::Tensor& hidden_state) {
    m_talker_projections.set_tensor("hidden_state", hidden_state);
    m_talker_projections.infer();
    auto result = m_talker_projections.get_tensor("text_projection");
    ov::Tensor copy(result.get_element_type(), result.get_shape());
    result.copy_to(copy);
    return copy;
}

ov::Tensor Qwen3OmniSpeechPipeline::project_hidden(const ov::Tensor& hidden_state) {
    m_talker_projections.set_tensor("hidden_state", hidden_state);
    m_talker_projections.infer();
    auto result = m_talker_projections.get_tensor("hidden_projection");
    ov::Tensor copy(result.get_element_type(), result.get_shape());
    result.copy_to(copy);
    return copy;
}

void Qwen3OmniSpeechPipeline::reset_talker() {
    if (m_talker) {
        for (auto& state : m_talker.query_state()) {
            state.reset();
        }
    }
}

void Qwen3OmniSpeechPipeline::reset_code_predictor() {
    if (m_code_predictor) {
        for (auto& state : m_code_predictor.query_state()) {
            state.reset();
        }
    }
}

int64_t Qwen3OmniSpeechPipeline::sample_top_k(const float* logits,
                                              size_t vocab_size,
                                              float temperature,
                                              size_t top_k,
                                              float repetition_penalty,
                                              const std::vector<int64_t>& generated_tokens,
                                              const std::vector<int64_t>& suppress_tokens) {
    OPENVINO_ASSERT(vocab_size > 0, "Logits tensor has zero vocab dimension");
    OPENVINO_ASSERT(temperature > 0.0f, "sample_top_k: temperature must be > 0");
    OPENVINO_ASSERT(top_k > 0, "sample_top_k: top_k must be > 0");
    OPENVINO_ASSERT(static_cast<size_t>(top_k) <= vocab_size, "sample_top_k: top_k exceeds logits size");

    // Thread-local scratch buffers to avoid heap allocation per call
    static thread_local std::vector<float> scaled;
    static thread_local std::vector<size_t> indices;
    scaled.assign(logits, logits + vocab_size);
    indices.resize(vocab_size);

    // Suppress forbidden tokens (set to -inf before any other processing)
    for (auto token_id : suppress_tokens) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            scaled[token_id] = -std::numeric_limits<float>::infinity();
        }
    }

    // Apply repetition penalty to previously generated tokens
    if (repetition_penalty != 1.0f && !generated_tokens.empty()) {
        for (auto token_id : generated_tokens) {
            if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size)
                continue;
            if (scaled[token_id] >= 0.0f) {
                scaled[token_id] /= repetition_penalty;
            } else {
                scaled[token_id] *= repetition_penalty;
            }
        }
    }

    // Apply temperature
    for (size_t i = 0; i < vocab_size; i++) {
        scaled[i] /= temperature;
    }

    // Find top-k threshold using nth_element (O(n) vs O(n + k log k) for partial_sort)
    size_t k = std::min(top_k, vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    auto* scaled_ptr = scaled.data();
    std::nth_element(indices.begin(), indices.begin() + k - 1, indices.end(), [scaled_ptr](size_t a, size_t b) {
        return scaled_ptr[a] > scaled_ptr[b];
    });
    float threshold = scaled[indices[k - 1]];

    // Collect top-k indices (values >= threshold) and compute softmax only over them
    static thread_local std::vector<size_t> topk_indices;
    static thread_local std::vector<float> topk_probs;
    topk_indices.clear();
    for (size_t i = 0; i < vocab_size; i++) {
        if (scaled[i] >= threshold) {
            topk_indices.push_back(i);
        }
    }

    float max_val = scaled[topk_indices[0]];
    for (size_t i = 1; i < topk_indices.size(); i++) {
        max_val = std::max(max_val, scaled[topk_indices[i]]);
    }

    topk_probs.resize(topk_indices.size());
    float sum = 0.0f;
    for (size_t i = 0; i < topk_indices.size(); i++) {
        topk_probs[i] = std::exp(scaled[topk_indices[i]] - max_val);
        sum += topk_probs[i];
    }
    OPENVINO_ASSERT(sum > 0.0f, "sample_top_k: sum of probabilities is zero");
    for (size_t i = 0; i < topk_probs.size(); i++) {
        topk_probs[i] /= sum;
    }

    // Multinomial sampling over top-k entries only
    static thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<size_t> dist(topk_probs.begin(), topk_probs.end());
    return static_cast<int64_t>(topk_indices[dist(gen)]);
}

std::pair<ov::Tensor, ov::Tensor> Qwen3OmniSpeechPipeline::build_talker_input(
    const std::vector<int64_t>& full_token_ids,
    const std::vector<ov::Tensor>& all_hidden_states,
    const std::vector<ov::Tensor>& all_intermediate_hidden_states,
    int64_t speaker_codec_id) {
    // Find <|im_start|> positions to identify segments
    std::vector<size_t> im_start_positions;
    for (size_t i = 0; i < full_token_ids.size(); i++) {
        if (full_token_ids[i] == m_config.im_start_token_id) {
            im_start_positions.push_back(i);
        }
    }

    GENAI_DEBUG("Speech: found %zu im_start segments, total_tokens=%zu, intermediate_hs=%zu",
                im_start_positions.size(),
                full_token_ids.size(),
                all_intermediate_hidden_states.size());

    auto hidden_size = m_config.talker_hidden_size;

    // Pre-compute TTS special embeddings in talker space
    auto tts_bos_embed = project_text(embed_thinker_token(m_config.tts_bos_token_id));
    auto tts_eos_embed = project_text(embed_thinker_token(m_config.tts_eos_token_id));
    auto tts_pad_embed = project_text(embed_thinker_token(m_config.tts_pad_token_id));

    // Reuse pre-allocated member buffer (avoids heap allocation per call)
    m_talker_buf.clear();
    m_talker_buf.reserve(full_token_ids.size() * hidden_size);

    // Append a projected tensor's data to the flat buffer
    auto append_tensor = [this, hidden_size](const ov::Tensor& t) {
        const auto* data = t.data<const float>();
        m_talker_buf.insert(m_talker_buf.end(), data, data + hidden_size);
    };

    // Element-wise addition of two embedding tensors, appended to flat buffer
    auto add_embeddings = [this, hidden_size](const ov::Tensor& a, const ov::Tensor& b) {
        size_t offset = m_talker_buf.size();
        m_talker_buf.resize(offset + hidden_size);
        const auto* a_data = a.data<const float>();
        const auto* b_data = b.data<const float>();
        for (size_t i = 0; i < hidden_size; i++) {
            m_talker_buf[offset + i] = a_data[i] + b_data[i];
        }
    };

    // Process each segment
    int last_assistant_seg_idx = -1;
    for (int seg = static_cast<int>(im_start_positions.size()) - 1; seg >= 0; seg--) {
        auto pos = im_start_positions[seg];
        if (pos + 1 < full_token_ids.size() && full_token_ids[pos + 1] == m_config.assistant_token_id) {
            last_assistant_seg_idx = seg;
            break;
        }
    }

    // For simplicity in initial implementation: process user segments and last assistant segment
    for (size_t seg = 0; seg < im_start_positions.size(); seg++) {
        auto seg_start = im_start_positions[seg];
        auto seg_end = (seg + 1 < im_start_positions.size()) ? im_start_positions[seg + 1] : full_token_ids.size();

        if (seg_start + 1 >= full_token_ids.size())
            continue;
        auto role_token = full_token_ids[seg_start + 1];

        // Skip system segments
        if (role_token == m_config.system_token_id) {
            continue;
        }

        // Skip non-last assistant segments
        if (role_token == m_config.assistant_token_id && static_cast<int>(seg) != last_assistant_seg_idx) {
            continue;
        }

        if (role_token == m_config.user_token_id) {
            // User segment: text tokens use word_embedding -> text_projection,
            // multimodal tokens use LLM intermediate hidden states -> hidden_projection
            size_t mm_count = 0, mm_with_hs = 0, mm_without_hs = 0;
            for (size_t pos = seg_start; pos < seg_end; pos++) {
                bool is_multimodal =
                    (full_token_ids[pos] == m_config.audio_token_id || full_token_ids[pos] == m_config.image_token_id ||
                     full_token_ids[pos] == m_config.video_token_id);

                ov::Tensor projected;
                if (is_multimodal && pos < all_intermediate_hidden_states.size()) {
                    projected = project_hidden(all_intermediate_hidden_states[pos]);
                    mm_with_hs++;
                } else if (is_multimodal) {
                    // Multimodal token but no hidden state — fall back to word embedding
                    auto word_embed = embed_thinker_token(full_token_ids[pos]);
                    projected = project_text(word_embed);
                    mm_without_hs++;
                } else {
                    auto word_embed = embed_thinker_token(full_token_ids[pos]);
                    projected = project_text(word_embed);
                }
                if (is_multimodal)
                    mm_count++;

                append_tensor(projected);
            }
            GENAI_DEBUG("Speech: user seg [%zu, %zu): %zu tokens, %zu multimodal (%zu with HS, %zu without)",
                        seg_start,
                        seg_end,
                        seg_end - seg_start,
                        mm_count,
                        mm_with_hs,
                        mm_without_hs);
        } else if (role_token == m_config.assistant_token_id) {
            // Last assistant segment: build structured input with codec tokens
            // Positions in assistant segment (relative to seg_start):
            // 0: <|im_start|>, 1: assistant, 2: \n, 3+: text tokens

            // Project all assistant positions: word_embedding -> text_projection
            std::vector<ov::Tensor> assistant_projected;
            for (size_t pos = seg_start; pos < seg_end; pos++) {
                auto word_embed = embed_thinker_token(full_token_ids[pos]);
                assistant_projected.push_back(project_text(word_embed));
            }

            if (assistant_projected.size() < 3)
                continue;

            // Build text_hidden + codec_hidden (element-wise addition)
            // Positions 0-2: projected header tokens + zeros (no codec)
            for (size_t i = 0; i < 3 && i < assistant_projected.size(); i++) {
                append_tensor(assistant_projected[i]);
            }

            // Positions 3-6: tts_pad + codec token embeddings
            add_embeddings(tts_pad_embed, embed_talker_token(m_config.codec_nothink_id));
            add_embeddings(tts_pad_embed, embed_talker_token(m_config.codec_think_bos_id));
            add_embeddings(tts_pad_embed, embed_talker_token(m_config.codec_think_eos_id));
            add_embeddings(tts_pad_embed, embed_talker_token(speaker_codec_id));

            // Position 7: tts_bos + codec_pad
            add_embeddings(tts_bos_embed, embed_talker_token(m_config.codec_pad_id));

            // Position 8: first text token + codec_bos
            if (assistant_projected.size() > 3) {
                add_embeddings(assistant_projected[3], embed_talker_token(m_config.codec_bos_id));
            }

            // Build trailing_text_hidden: remaining projected text tokens (positions 4+)
            // plus tts_eos at the end — write directly into flat tensor
            size_t trailing_start = 4;  // Skip im_start, assistant, \n, first_text
            size_t trailing_len =
                (assistant_projected.size() > trailing_start ? assistant_projected.size() - trailing_start : 0) +
                1;  // +1 for tts_eos
            ov::Tensor trailing_hidden(ov::element::f32, {1, trailing_len, hidden_size});
            auto* trailing_ptr = trailing_hidden.data<float>();
            size_t t_idx = 0;
            for (size_t i = trailing_start; i < assistant_projected.size(); i++) {
                std::memcpy(trailing_ptr + t_idx * hidden_size,
                            assistant_projected[i].data<float>(),
                            hidden_size * sizeof(float));
                t_idx++;
            }
            std::memcpy(trailing_ptr + t_idx * hidden_size, tts_eos_embed.data<float>(), hidden_size * sizeof(float));

            // Build the talker input tensor from flat buffer (already contiguous)
            size_t total_len = m_talker_buf.size() / hidden_size;
            ov::Tensor talker_input(ov::element::f32, {1, total_len, hidden_size});
            std::memcpy(talker_input.data<float>(), m_talker_buf.data(), m_talker_buf.size() * sizeof(float));

            return {talker_input, trailing_hidden};
        }
    }

    // Fallback: empty input (should not happen with valid ChatML)
    ov::Tensor empty_input(ov::element::f32, {1, 0, hidden_size});
    ov::Tensor empty_trailing(ov::element::f32, {1, 0, hidden_size});
    return {empty_input, empty_trailing};
}

std::pair<std::vector<int64_t>, ov::Tensor> Qwen3OmniSpeechPipeline::predict_codes(
    const ov::Tensor& talker_hidden_state,
    int64_t first_code) {
    std::vector<int64_t> codes;
    codes.reserve(m_config.num_code_groups - 1);

    auto hidden_size = m_config.talker_hidden_size;

    // Zero-fill the pre-allocated accumulator instead of allocating a new tensor
    auto* sum_data = m_cp_embed_sum.data<float>();
    std::fill_n(sum_data, hidden_size, 0.0f);

    reset_code_predictor();

    auto hs_size = talker_hidden_state.get_shape().back();
    // Reusable decode input tensor (written each step)
    ov::Tensor cp_input(ov::element::f32, {1, 1, hs_size});
    size_t cp_history_len = 0;

    // Pre-allocate attention mask at max size (2 prefill + 14 decode = 16)
    auto cp_max_len = m_config.num_code_groups + 1;
    ov::Tensor cp_attn_mask(ov::element::i64, {1, cp_max_len});
    std::fill_n(cp_attn_mask.data<int64_t>(), cp_max_len, 1);

    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;

    // Helper: accumulate embedding into sum and write to cp_input
    auto accumulate_cp_embed = [&](size_t step, int64_t code, bool write_input) {
        if (m_has_cp_embeds) {
            const auto* weights = cp_token_weights(step, code);
            for (size_t i = 0; i < hidden_size; i++)
                sum_data[i] += weights[i];
            if (write_input) {
                std::memcpy(cp_input.data<float>(), weights, hidden_size * sizeof(float));
            }
        } else {
            auto embed = embed_talker_token(code);
            auto* emb_data = embed.data<float>();
            for (size_t i = 0; i < hidden_size; i++)
                sum_data[i] += emb_data[i];
            if (write_input) {
                std::memcpy(cp_input.data<float>(), emb_data, hidden_size * sizeof(float));
            }
        }
    };

    // Prefill: 2 tokens [talker_hidden, talker_embed(first_code)]
    {
        auto first_code_embed = embed_talker_token(first_code);
        ov::Tensor prefill_input(ov::element::f32, {1, 2, hs_size});
        auto* prefill_data = prefill_input.data<float>();
        std::memcpy(prefill_data, talker_hidden_state.data<float>(), hs_size * sizeof(float));
        std::memcpy(prefill_data + hs_size, first_code_embed.data<float>(), hs_size * sizeof(float));

        m_code_predictor.set_tensor("inputs_embeds", prefill_input);

        cp_attn_mask.set_shape({1, 2});
        m_code_predictor.set_tensor("attention_mask", cp_attn_mask);

        ov::Tensor pos_ids(ov::element::i64, {1, 2});
        pos_ids.data<int64_t>()[0] = 0;
        pos_ids.data<int64_t>()[1] = 1;
        m_code_predictor.set_tensor("position_ids", pos_ids);

        ov::Tensor gen_steps(ov::element::i64, {});
        gen_steps.data<int64_t>()[0] = 0;
        m_code_predictor.set_tensor("generation_steps", gen_steps);
        m_code_predictor.set_tensor("beam_idx", beam_idx);
        m_code_predictor.infer();

        auto logits = m_code_predictor.get_tensor("logits");
        auto hidden = m_code_predictor.get_tensor("hidden_states");

        auto vocab_size = logits.get_shape().back();
        const auto* logits_data = logits.data<float>() + (logits.get_size() - vocab_size);
        auto code = sample_top_k(logits_data, vocab_size, 1.0f, 50, 1.0f, {});
        codes.push_back(code);

        accumulate_cp_embed(0, code, true);

        if (!m_has_cp_embeds) {
            const auto* hs_data = hidden.data<float>() + (hidden.get_size() - hs_size);
            std::memcpy(cp_input.data<float>(), hs_data, hs_size * sizeof(float));
        }
        cp_history_len = 2;
    }

    // Reusable tensors for decode steps
    ov::Tensor cp_pos_ids(ov::element::i64, {1, 1});
    ov::Tensor cp_gen_steps(ov::element::i64, {});

    // Generate remaining codes (steps 1..14)
    for (size_t step = 1; step < m_config.num_code_groups - 1; step++) {
        cp_history_len++;
        m_code_predictor.set_tensor("inputs_embeds", cp_input);

        cp_attn_mask.set_shape({1, cp_history_len});
        m_code_predictor.set_tensor("attention_mask", cp_attn_mask);

        cp_pos_ids.data<int64_t>()[0] = static_cast<int64_t>(cp_history_len - 1);
        m_code_predictor.set_tensor("position_ids", cp_pos_ids);

        cp_gen_steps.data<int64_t>()[0] = static_cast<int64_t>(step);
        m_code_predictor.set_tensor("generation_steps", cp_gen_steps);
        m_code_predictor.set_tensor("beam_idx", beam_idx);
        m_code_predictor.infer();

        auto logits = m_code_predictor.get_tensor("logits");
        auto hidden = m_code_predictor.get_tensor("hidden_states");

        auto vocab_size = logits.get_shape().back();
        const auto* logits_data = logits.data<float>() + (logits.get_size() - vocab_size);
        auto code = sample_top_k(logits_data, vocab_size, 1.0f, 50, 1.0f, {});
        codes.push_back(code);

        bool need_next_input = step < m_config.num_code_groups - 2;
        accumulate_cp_embed(step, code, need_next_input);

        if (need_next_input && !m_has_cp_embeds) {
            const auto* hs_data = hidden.data<float>() + (hidden.get_size() - hs_size);
            std::memcpy(cp_input.data<float>(), hs_data, hs_size * sizeof(float));
        }
    }

    return {codes, m_cp_embed_sum};
}

ov::Tensor Qwen3OmniSpeechPipeline::codes_to_wav(const ov::Tensor& codes) {
    m_code2wav.set_tensor("codes", codes);
    m_code2wav.infer();
    auto waveform = m_code2wav.get_tensor("waveform");
    ov::Tensor result(waveform.get_element_type(), waveform.get_shape());
    waveform.copy_to(result);
    return result;
}

ov::Tensor Qwen3OmniSpeechPipeline::generate_speech(const std::vector<int64_t>& full_token_ids,
                                                    const std::vector<ov::Tensor>& all_hidden_states,
                                                    const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                                                    const AudioStreamerVariant& audio_streamer,
                                                    size_t chunk_frames,
                                                    const std::string& speaker,
                                                    size_t max_new_tokens) {
    bool streaming = is_audio_streamer_active(audio_streamer) && chunk_frames > 0;

    if (!m_talker_available) {
        GENAI_WARN("Speech: talker not available");
        if (streaming)
            end_audio_streamer(audio_streamer);
        return ov::Tensor();
    }

    GENAI_DEBUG("Speech: tokens=%zu, hidden_states=%zu, intermediate=%zu",
                full_token_ids.size(),
                all_hidden_states.size(),
                all_intermediate_hidden_states.size());

    int64_t speaker_codec_id = resolve_speaker_id(speaker);

    auto [talker_input, trailing_text_hidden] =
        build_talker_input(full_token_ids, all_hidden_states, all_intermediate_hidden_states, speaker_codec_id);

    if (talker_input.get_shape()[1] == 0) {
        GENAI_WARN("Speech: build_talker_input returned empty, cannot generate speech");
        if (streaming)
            end_audio_streamer(audio_streamer);
        return ov::Tensor();
    }

    GENAI_DEBUG("Speech: talker_input=[1, %zu, %zu], trailing=[1, %zu, ...], streaming=%s, chunk_frames=%zu",
                talker_input.get_shape()[1],
                talker_input.get_shape()[2],
                trailing_text_hidden.get_shape()[1],
                streaming ? "true" : "false",
                chunk_frames);

    reset_talker();

    // Prefill: run talker on constructed input
    auto input_len = talker_input.get_shape()[1];
    auto hidden_size = talker_input.get_shape()[2];

    m_talker.set_tensor("inputs_embeds", talker_input);

    // Use talker generation parameters from config
    auto talker_max_tokens = std::min(max_new_tokens, m_config.talker_max_new_tokens);

    // F2: Pre-allocate attention mask at max size (input_len + talker_max_tokens)
    auto talker_max_len = input_len + talker_max_tokens;
    ov::Tensor talker_attn_mask(ov::element::i64, {1, talker_max_len});
    std::fill_n(talker_attn_mask.data<int64_t>(), talker_max_len, 1);

    talker_attn_mask.set_shape({1, input_len});
    m_talker.set_tensor("attention_mask", talker_attn_mask);

    ov::Tensor pos_ids(ov::element::i64, {1, input_len});
    auto* pos_data = pos_ids.data<int64_t>();
    for (size_t i = 0; i < input_len; i++) {
        pos_data[i] = static_cast<int64_t>(i);
    }
    m_talker.set_tensor("position_ids", pos_ids);

    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;
    m_talker.set_tensor("beam_idx", beam_idx);

    m_talker.infer();

    // Collect all codec codes: [num_steps, num_code_groups]
    std::vector<std::vector<int64_t>> all_codes;
    all_codes.reserve(talker_max_tokens);

    // Streaming cursor: index into all_codes for next chunk start
    size_t chunk_cursor = 0;

    auto trailing_len = trailing_text_hidden.get_shape()[1];
    size_t trailing_idx = 0;
    auto history_len = input_len;
    auto num_quantizers = m_config.num_code_groups;
    bool early_stop = false;

    // Pre-compute tts_pad embedding in talker space for padding
    auto tts_pad_proj = project_text(embed_thinker_token(m_config.tts_pad_token_id));

    // Pre-allocate codes stacking buffer at max possible size, then use set_shape per call
    m_stack_codes_buf = ov::Tensor(ov::element::i64, {1, num_quantizers, talker_max_tokens});

    auto stack_codes_range = [&](size_t begin, size_t end) -> ov::Tensor {
        size_t n_steps = end - begin;
        m_stack_codes_buf.set_shape({1, num_quantizers, n_steps});
        auto* data = m_stack_codes_buf.data<int64_t>();
        for (size_t s = 0; s < n_steps; s++) {
            const auto& frame = all_codes[begin + s];
            for (size_t q = 0; q < num_quantizers && q < frame.size(); q++) {
                data[q * n_steps + s] = frame[q];
            }
        }
        return m_stack_codes_buf;
    };

    auto talker_temp = m_config.talker_temperature;
    auto talker_top_k = m_config.talker_top_k;
    auto talker_rep_penalty = m_config.talker_repetition_penalty;
    const auto& suppress_tokens = m_config.talker_suppress_tokens;
    std::vector<int64_t> generated_first_codes;

    // Reusable tensors for decode steps
    ov::Tensor next_input(ov::element::f32, {1, 1, hidden_size});
    ov::Tensor next_pos(ov::element::i64, {1, 1});

    GENAI_INFO("Speech: generating codec tokens (max %zu steps, temp=%.2f, top_k=%zu, rep_penalty=%.2f)...",
               talker_max_tokens,
               talker_temp,
               talker_top_k,
               talker_rep_penalty);

    for (size_t step = 0; step < talker_max_tokens; step++) {
        auto logits = m_talker.get_tensor("logits");
        auto talker_hidden = m_talker.get_tensor("hidden_states");

        // Sample first codec token from talker with repetition penalty
        auto vocab_size = logits.get_shape().back();
        const auto* logits_data = logits.data<float>() + (logits.get_size() - vocab_size);
        auto first_code = sample_top_k(logits_data,
                                       vocab_size,
                                       talker_temp,
                                       talker_top_k,
                                       talker_rep_penalty,
                                       generated_first_codes,
                                       suppress_tokens);

        if (step < 3 || step % 100 == 0) {
            GENAI_DEBUG("Speech: step %zu, code=%lld", step, (long long)first_code);
        }

        if (first_code == m_config.codec_eos_token_id) {
            GENAI_INFO("Speech: completed at step %zu", step);
            break;
        }

        generated_first_codes.push_back(first_code);

        // Get talker's last hidden state for CodePredictor
        auto hs_size = talker_hidden.get_shape().back();
        ov::Tensor last_hidden(ov::element::f32, {1, 1, hs_size});
        const auto* hs_data = talker_hidden.data<float>() + (talker_hidden.get_size() - hs_size);
        std::memcpy(last_hidden.data<float>(), hs_data, hs_size * sizeof(float));

        // Run CodePredictor for additional code groups
        auto [additional_codes, cp_embed_sum] = predict_codes(last_hidden, first_code);

        // F8: Emplace directly into all_codes (no intermediate step_codes copy)
        all_codes.emplace_back();
        auto& current_codes = all_codes.back();
        current_codes.reserve(1 + additional_codes.size());
        current_codes.push_back(first_code);
        current_codes.insert(current_codes.end(), additional_codes.begin(), additional_codes.end());

        // F9: Streaming uses cursor into all_codes instead of separate chunk_codes
        if (streaming && all_codes.size() - chunk_cursor >= chunk_frames) {
            auto chunk_tensor = stack_codes_range(chunk_cursor, all_codes.size());
            auto chunk_wav = codes_to_wav(chunk_tensor);
            chunk_cursor = all_codes.size();

            auto status = invoke_audio_streamer(audio_streamer, chunk_wav);
            if (status == StreamingStatus::STOP || status == StreamingStatus::CANCEL) {
                GENAI_INFO("Speech: streaming %s at step %zu",
                           status == StreamingStatus::STOP ? "stopped" : "cancelled",
                           step);
                early_stop = true;
                break;
            }
        }

        // Build next talker input: sum of all codec representations + trailing text
        auto first_code_embed = embed_talker_token(first_code);
        auto* next_data = next_input.data<float>();
        auto* fc_data = first_code_embed.data<float>();
        auto* cp_data = cp_embed_sum.data<float>();
        for (size_t i = 0; i < hidden_size; i++) {
            next_data[i] = fc_data[i] + cp_data[i];
        }

        if (trailing_idx < trailing_len) {
            const auto* trail_data = trailing_text_hidden.data<float>() + trailing_idx * hidden_size;
            for (size_t i = 0; i < hidden_size; i++) {
                next_data[i] += trail_data[i];
            }
            trailing_idx++;
        } else {
            auto* pad_data = tts_pad_proj.data<float>();
            for (size_t i = 0; i < hidden_size; i++) {
                next_data[i] += pad_data[i];
            }
        }

        // F2: Reuse pre-allocated attention mask with set_shape
        history_len++;
        talker_attn_mask.set_shape({1, history_len});
        m_talker.set_tensor("attention_mask", talker_attn_mask);

        next_pos.data<int64_t>()[0] = static_cast<int64_t>(history_len - 1);
        m_talker.set_tensor("position_ids", next_pos);

        m_talker.set_tensor("inputs_embeds", next_input);
        m_talker.set_tensor("beam_idx", beam_idx);
        m_talker.infer();
    }

    // Flush remaining frames if streaming
    if (streaming && chunk_cursor < all_codes.size() && !early_stop) {
        auto chunk_tensor = stack_codes_range(chunk_cursor, all_codes.size());
        auto chunk_wav = codes_to_wav(chunk_tensor);
        invoke_audio_streamer(audio_streamer, chunk_wav);
    }

    // Always call end() on active streamer
    if (streaming) {
        end_audio_streamer(audio_streamer);
    }

    if (all_codes.size() == talker_max_tokens) {
        GENAI_WARN("Speech: reached max tokens (%zu) without EOS", talker_max_tokens);
    }

    if (all_codes.empty()) {
        GENAI_WARN("Speech: no codes generated");
        return ov::Tensor();
    }

    GENAI_INFO("Speech: %zu codec steps generated, converting to waveform...", all_codes.size());

    // Always return full waveform for backward compatibility
    auto full_codes = stack_codes_range(0, all_codes.size());
    return codes_to_wav(full_codes);
}

}  // namespace ov::genai
