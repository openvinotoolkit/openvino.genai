// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_omni/speech_pipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>

#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace {

ov::genai::StreamingStatus invoke_speech_streamer(const ov::genai::OmniSpeechStreamerVariant& streamer,
                                                  const ov::Tensor& chunk) {
    return std::visit(
        [&chunk](auto&& arg) -> ov::genai::StreamingStatus {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::function<ov::genai::StreamingStatus(const ov::Tensor&)>>) {
                return arg(chunk);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::OmniSpeechStreamerBase>>) {
                return arg->write(chunk);
            } else {
                return ov::genai::StreamingStatus::RUNNING;
            }
        },
        streamer);
}

void end_speech_streamer(const ov::genai::OmniSpeechStreamerVariant& streamer) {
    std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<ov::genai::OmniSpeechStreamerBase>>) {
                arg->end();
            }
        },
        streamer);
}

bool is_speech_streamer_active(const ov::genai::OmniSpeechStreamerVariant& streamer) {
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
    sc.system_token_id = config.system_token_id;
    sc.user_token_id = config.user_token_id;
    sc.assistant_token_id = config.assistant_token_id;
    sc.audio_token_id = config.audio_token_id;
    sc.image_token_id = config.image_token_id;
    sc.video_token_id = config.video_token_id;
    sc.num_code_groups = config.talker_num_code_groups;
    sc.speaker_ids = config.speaker_ids;

    OPENVINO_ASSERT(sc.codec_bos_id >= 0 && sc.codec_eos_token_id >= 0 && sc.codec_pad_id >= 0 &&
                        sc.codec_nothink_id >= 0 && sc.codec_think_bos_id >= 0 && sc.codec_think_eos_id >= 0,
                    "Qwen3-Omni speech: talker_config codec IDs missing from config.json "
                    "(codec_bos_id / codec_eos_token_id / codec_pad_id / codec_nothink_id / "
                    "codec_think_bos_id / codec_think_eos_id are required for speech generation).");
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

        // Force FP32 inference precision on GPU for talker models to match CPU behavior
        // GPU FP16 causes numerical differences in logits → different sampled tokens → wrong speech length
        ov::AnyMap compilation_props = properties;
        if (device == "GPU" || device.find("GPU") == 0) {
            compilation_props[ov::hint::inference_precision.name()] = ov::element::f32;
            GENAI_DEBUG("Speech: forcing FP32 precision for %s on GPU", filename.c_str());
        }

        auto compiled = utils::singleton_core().compile_model(model, device, compilation_props);
        return compiled.create_infer_request();
    };

    // Load thinker text embeddings for TTS special token embedding
    m_thinker_text_embeddings = load_model("openvino_text_embeddings_model");

    m_talker = load_model("openvino_talker_model");
    m_talker_text_embeddings = load_model("openvino_talker_text_embeddings_model");
    m_talker_projections = load_model("openvino_talker_projections_model");
    m_code_predictor = load_model("openvino_code_predictor_model");
    m_code2wav = load_model("openvino_code2wav_model");

    initialize(model_dir);
}

Qwen3OmniSpeechPipeline::Qwen3OmniSpeechPipeline(const ModelsMap& models_map,
                                                 const VLMConfig& config,
                                                 const std::filesystem::path& config_dir_path,
                                                 const std::map<std::string, std::string>& device_mapping,
                                                 const std::string& default_device,
                                                 const ov::AnyMap& properties)
    : m_config(Qwen3OmniSpeechConfig::from_vlm_config(config)) {
    // Compile each submodel from its in-memory IR + weights, placing it on the device named in
    // device_mapping (falling back to default_device). Submodels absent from models_map stay empty
    // and trip the availability check in initialize().
    auto load_model = [&](const std::string& name) -> ov::InferRequest {
        auto it = models_map.find(name);
        if (it == models_map.end()) {
            return {};
        }
        const auto& [ir, weights] = it->second;
        auto model = utils::singleton_core().read_model(ir, weights);

        auto dev_it = device_mapping.find(name);
        const std::string& device = dev_it != device_mapping.end() ? dev_it->second : default_device;

        // Force FP32 inference precision on GPU (see disk ctor): GPU FP16 shifts talker logits,
        // changing sampled tokens and corrupting speech length.
        ov::AnyMap compilation_props = properties;
        if (device == "GPU" || device.find("GPU") == 0) {
            compilation_props[ov::hint::inference_precision.name()] = ov::element::f32;
            GENAI_DEBUG("Speech: forcing FP32 precision for %s on GPU", name.c_str());
        }

        auto compiled = utils::singleton_core().compile_model(model, device, compilation_props);
        return compiled.create_infer_request();
    };

    m_thinker_text_embeddings = load_model("text_embeddings");
    m_talker = load_model("talker");
    m_talker_text_embeddings = load_model("talker_text_embeddings");
    m_talker_projections = load_model("talker_projections");
    m_code_predictor = load_model("code_predictor");
    m_code2wav = load_model("code2wav");

    initialize(config_dir_path);
}

void Qwen3OmniSpeechPipeline::initialize(const std::filesystem::path& config_dir) {
    // All speech models must be present for speech generation
    m_talker_available = m_talker && m_talker_text_embeddings && m_talker_projections && m_code_predictor &&
                         m_code2wav && m_thinker_text_embeddings;

    if (!m_talker_available) {
        return;
    }
    auto output_pshape = m_talker_projections.get_compiled_model().output("text_projection").get_partial_shape();
    OPENVINO_ASSERT(output_pshape.rank().is_static() && output_pshape.rank().get_length() >= 2,
                    "Talker text projection output must have at least 2 dimensions");
    auto last_dim = output_pshape[output_pshape.rank().get_length() - 1];
    OPENVINO_ASSERT(last_dim.is_static(), "Talker text projection output last dimension must be static");
    m_config.talker_hidden_size = static_cast<size_t>(last_dim.get_length());
    OPENVINO_ASSERT(m_config.talker_hidden_size > 0,
                    "Failed to detect talker hidden size from text projection model");

    {
        auto cp_inputs = m_code_predictor.get_compiled_model().inputs();
        auto has_cp_input = [&](const std::string& name) {
            return std::any_of(cp_inputs.begin(), cp_inputs.end(), [&](const ov::Output<const ov::Node>& port) {
                return port.get_any_name() == name;
            });
        };
        auto cp_outputs = m_code_predictor.get_compiled_model().outputs();
        auto has_cp_output = [&](const std::string& name) {
            return std::any_of(cp_outputs.begin(), cp_outputs.end(), [&](const ov::Output<const ov::Node>& port) {
                return port.get_any_name() == name;
            });
        };

        // Single-step stateful CodePredictor API: the graph runs one inner step per infer() and
        // hides its KV cache as state; this pipeline drives the num_code_groups-1 step loop.
        // Sampling and codec embedding are in-graph, so the graph emits the sampled `code` and its
        // `token_embed` directly (no separate codec_embedding weights file needed).
        OPENVINO_ASSERT(has_cp_input("inputs_embeds"), "CodePredictor: missing 'inputs_embeds' input");
        OPENVINO_ASSERT(has_cp_input("attention_mask"), "CodePredictor: missing 'attention_mask' input");
        OPENVINO_ASSERT(has_cp_input("position_ids"), "CodePredictor: missing 'position_ids' input");
        OPENVINO_ASSERT(has_cp_input("step"), "CodePredictor: missing 'step' input (expected single-step API)");
        OPENVINO_ASSERT(has_cp_input("seed"), "CodePredictor: missing 'seed' input");
        OPENVINO_ASSERT(has_cp_input("temperature"), "CodePredictor: missing 'temperature' input");
        OPENVINO_ASSERT(has_cp_input("top_k"), "CodePredictor: missing 'top_k' input");
        OPENVINO_ASSERT(has_cp_output("code"), "CodePredictor: missing 'code' output");
        OPENVINO_ASSERT(has_cp_output("token_embed"), "CodePredictor: missing 'token_embed' output");
    }

    // Pre-allocate scratch buffers that are reused across generate_speech() calls
    m_cp_embed_sum = ov::Tensor(ov::element::f32, {1, 1, m_config.talker_hidden_size});

    if (m_config.speaker_ids.empty()) {
        m_talker_available = false;
    }

    // Load talker and CodePredictor generation parameters from generation_config.json.
    // CP defaults (1.0 / 50 / 1.0) match the reference Qwen3-Omni implementation; json keys
    // cp_temperature / cp_top_k / cp_repetition_penalty may override them if present.
    auto gen_config_path = config_dir / "generation_config.json";
    if (std::filesystem::exists(gen_config_path)) {
        std::ifstream f(gen_config_path);
        auto gen_data = nlohmann::json::parse(f);
        utils::read_json_param(gen_data, "talker_temperature", m_config.talker_temperature);
        utils::read_json_param(gen_data, "talker_top_k", m_config.talker_top_k);
        utils::read_json_param(gen_data, "talker_repetition_penalty", m_config.talker_repetition_penalty);
        utils::read_json_param(gen_data, "talker_max_new_tokens", m_config.talker_max_new_tokens);
        utils::read_json_param(gen_data, "cp_temperature", m_config.cp_temperature);
        utils::read_json_param(gen_data, "cp_top_k", m_config.cp_top_k);
        utils::read_json_param(gen_data, "cp_repetition_penalty", m_config.cp_repetition_penalty);
        GENAI_INFO("Speech: talker params: temp=%.2f, top_k=%zu, rep_penalty=%.2f, max_tokens=%zu",
                    m_config.talker_temperature,
                    m_config.talker_top_k,
                    m_config.talker_repetition_penalty,
                    m_config.talker_max_new_tokens);
        GENAI_INFO("Speech: code_predictor params: temp=%.2f, top_k=%zu, rep_penalty=%.2f",
                    m_config.cp_temperature,
                    m_config.cp_top_k,
                    m_config.cp_repetition_penalty);
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

    // Pre-compute constant embeddings. These depend only on model weights and
    // static config, so hoisting them out of generate_speech() saves ~12 inference
    // calls per call at the cost of a one-shot warm-up during pipeline construction.
    if (m_talker_available) {
        m_tts_bos_embed = project_text(embed_thinker_token(m_config.tts_bos_token_id));
        m_tts_eos_embed = project_text(embed_thinker_token(m_config.tts_eos_token_id));
        m_tts_pad_embed = project_text(embed_thinker_token(m_config.tts_pad_token_id));

        const std::array<int64_t, 5> codec_specials{m_config.codec_nothink_id,
                                                    m_config.codec_think_bos_id,
                                                    m_config.codec_think_eos_id,
                                                    m_config.codec_pad_id,
                                                    m_config.codec_bos_id};
        for (auto id : codec_specials) {
            m_codec_special_embed.emplace(id, embed_talker_token(id));
        }

        for (const auto& [name, id] : m_config.speaker_ids) {
            m_speaker_embed.emplace(id, embed_talker_token(id));
            std::string lower_name = name;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
            m_lower_speaker_ids.emplace(std::move(lower_name), id);
        }
    }
}

ov::Tensor Qwen3OmniSpeechPipeline::get_speaker_embedding(const std::string& name) const {
    OPENVINO_ASSERT(!m_lower_speaker_ids.empty(),
                    "Talker::get_speaker_embedding: model has no named speakers; "
                    "ensure 'talker_config.speaker_id' is defined in config.json");
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    auto it = m_lower_speaker_ids.find(lower);
    OPENVINO_ASSERT(it != m_lower_speaker_ids.end(),
                    "Talker::get_speaker_embedding: unknown speaker '",
                    name,
                    "'. Use list_speakers() to enumerate available names.");
    // Copy so caller can blend without mutating our internal cache.
    const auto& cached = m_speaker_embed.at(it->second);
    ov::Tensor copy(cached.get_element_type(), cached.get_shape());
    cached.copy_to(copy);
    return copy;
}

std::vector<std::string> Qwen3OmniSpeechPipeline::list_speakers() const {
    std::vector<std::string> names;
    names.reserve(m_config.speaker_ids.size());
    for (const auto& [name, id] : m_config.speaker_ids) {
        names.push_back(name);
    }
    return names;
}

int64_t Qwen3OmniSpeechPipeline::resolve_speaker_id(const std::string& speaker) const {
    if (!speaker.empty() && !m_lower_speaker_ids.empty()) {
        std::string lower_speaker = speaker;
        std::transform(lower_speaker.begin(), lower_speaker.end(), lower_speaker.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        auto it = m_lower_speaker_ids.find(lower_speaker);
        if (it != m_lower_speaker_ids.end()) {
            return it->second;
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

    // Per-instance scratch buffers — not thread-safe, matching the contract of the rest of the
    // pipeline (single-threaded per Qwen3OmniSpeechPipeline instance).
    m_sample_scaled.assign(logits, logits + vocab_size);
    m_sample_indices.resize(vocab_size);

    // Suppress forbidden tokens (set to -inf before any other processing)
    for (auto token_id : suppress_tokens) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            m_sample_scaled[token_id] = -std::numeric_limits<float>::infinity();
        }
    }

    // Apply repetition penalty to previously generated tokens
    if (repetition_penalty != 1.0f && !generated_tokens.empty()) {
        for (auto token_id : generated_tokens) {
            if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size)
                continue;
            if (m_sample_scaled[token_id] >= 0.0f) {
                m_sample_scaled[token_id] /= repetition_penalty;
            } else {
                m_sample_scaled[token_id] *= repetition_penalty;
            }
        }
    }

    // Apply temperature
    for (size_t i = 0; i < vocab_size; i++) {
        m_sample_scaled[i] /= temperature;
    }

    // Find top-k threshold using nth_element (O(n) vs O(n + k log k) for partial_sort)
    size_t k = std::min(top_k, vocab_size);
    std::iota(m_sample_indices.begin(), m_sample_indices.end(), 0);
    auto* scaled_ptr = m_sample_scaled.data();
    std::nth_element(m_sample_indices.begin(),
                     m_sample_indices.begin() + k - 1,
                     m_sample_indices.end(),
                     [scaled_ptr](size_t a, size_t b) {
                         return scaled_ptr[a] > scaled_ptr[b];
                     });
    float threshold = m_sample_scaled[m_sample_indices[k - 1]];

    // Collect top-k indices (values >= threshold) and compute softmax only over them
    m_sample_topk_indices.clear();
    for (size_t i = 0; i < vocab_size; i++) {
        if (m_sample_scaled[i] >= threshold) {
            m_sample_topk_indices.push_back(i);
        }
    }

    float max_val = m_sample_scaled[m_sample_topk_indices[0]];
    for (size_t i = 1; i < m_sample_topk_indices.size(); i++) {
        max_val = std::max(max_val, m_sample_scaled[m_sample_topk_indices[i]]);
    }

    m_sample_topk_probs.resize(m_sample_topk_indices.size());
    float sum = 0.0f;
    for (size_t i = 0; i < m_sample_topk_indices.size(); i++) {
        m_sample_topk_probs[i] = std::exp(m_sample_scaled[m_sample_topk_indices[i]] - max_val);
        sum += m_sample_topk_probs[i];
    }
    OPENVINO_ASSERT(sum > 0.0f, "sample_top_k: sum of probabilities is zero");
    for (size_t i = 0; i < m_sample_topk_probs.size(); i++) {
        m_sample_topk_probs[i] /= sum;
    }

    // Multinomial sampling over top-k entries using the pipeline's seeded RNG
    std::discrete_distribution<size_t> dist(m_sample_topk_probs.begin(), m_sample_topk_probs.end());
    return static_cast<int64_t>(m_sample_topk_indices[dist(m_rng)]);
}

std::pair<ov::Tensor, ov::Tensor> Qwen3OmniSpeechPipeline::build_talker_input(
    const std::vector<int64_t>& full_token_ids,
    const std::vector<ov::Tensor>& all_intermediate_hidden_states,
    const ov::Tensor& speaker_embed) {
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

    // TTS special embeddings were pre-computed at pipeline construction (see ctor).
    const auto& tts_bos_embed = m_tts_bos_embed;
    const auto& tts_eos_embed = m_tts_eos_embed;
    const auto& tts_pad_embed = m_tts_pad_embed;

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
            // Last assistant segment: build input with codec tokens
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
            add_embeddings(tts_pad_embed, m_codec_special_embed.at(m_config.codec_nothink_id));
            add_embeddings(tts_pad_embed, m_codec_special_embed.at(m_config.codec_think_bos_id));
            add_embeddings(tts_pad_embed, m_codec_special_embed.at(m_config.codec_think_eos_id));
            add_embeddings(tts_pad_embed, speaker_embed);

            // Position 7: tts_bos + codec_pad
            add_embeddings(tts_bos_embed, m_codec_special_embed.at(m_config.codec_pad_id));

            // Position 8: first text token + codec_bos
            if (assistant_projected.size() > 3) {
                add_embeddings(assistant_projected[3], m_codec_special_embed.at(m_config.codec_bos_id));
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
    int64_t first_code,
    float cp_temperature,
    size_t cp_top_k) {
    const size_t num_cp_steps = m_config.num_code_groups - 1;
    std::vector<int64_t> codes;
    codes.reserve(num_cp_steps);

    auto hs_size = talker_hidden_state.get_shape().back();

    // codec_hiddens_sum starts from the first-code embedding; each step's codec embedding is added.
    // Mirrors the reference accumulation (first_code_embed + sum(step codec embeds)).
    auto first_code_embed = embed_talker_token(first_code);
    auto* sum_data = m_cp_embed_sum.data<float>();
    std::memcpy(sum_data, first_code_embed.data<float>(), hs_size * sizeof(float));

    // Fresh KV state per talker step: the single-step graph grows its cache across the inner loop.
    for (auto& state : m_code_predictor.query_state()) {
        state.reset();
    }

    // Prefill input: [1, 2, hidden] = [talker_hidden_state, embed(first_code)].
    // Decode inputs (steps > 0): [1, 1, hidden] = previous step's codec embedding (from the graph).
    ov::Tensor prefill_embeds(ov::element::f32, {1, 2, hs_size});
    auto* pf_data = prefill_embeds.data<float>();
    std::memcpy(pf_data, talker_hidden_state.data<float>(), hs_size * sizeof(float));
    std::memcpy(pf_data + hs_size, first_code_embed.data<float>(), hs_size * sizeof(float));

    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;

    // Sampling is in-graph (Gumbel-max); these scalars parameterize it per step.
    ov::Tensor step_tensor(ov::element::i64, {});
    ov::Tensor seed_tensor(ov::element::i64, {});
    ov::Tensor temp_tensor(ov::element::f32, {});
    temp_tensor.data<float>()[0] = cp_temperature;
    ov::Tensor topk_tensor(ov::element::i64, {});
    topk_tensor.data<int64_t>()[0] = static_cast<int64_t>(cp_top_k);

    // step_embeds is filled from the graph's token_embed output and fed back as the next input.
    ov::Tensor step_embeds(ov::element::f32, {1, 1, hs_size});

    size_t history_len = 0;  // KV positions already cached
    for (size_t step = 0; step < num_cp_steps; step++) {
        const auto& input_embeds = (step == 0) ? prefill_embeds : step_embeds;
        const size_t seq_len = input_embeds.get_shape()[1];

        ov::Tensor attn_mask(ov::element::i64, {1, history_len + seq_len});
        std::fill_n(attn_mask.data<int64_t>(), history_len + seq_len, 1);

        ov::Tensor pos_ids(ov::element::i64, {1, seq_len});
        auto* pos_data = pos_ids.data<int64_t>();
        for (size_t i = 0; i < seq_len; i++) {
            pos_data[i] = static_cast<int64_t>(history_len + i);
        }

        step_tensor.data<int64_t>()[0] = static_cast<int64_t>(step);
        // Per-step seed drawn from the shared RNG so one rng_seed reproduces the whole audio.
        seed_tensor.data<int64_t>()[0] = static_cast<int64_t>(m_rng());

        m_code_predictor.set_tensor("inputs_embeds", input_embeds);
        m_code_predictor.set_tensor("attention_mask", attn_mask);
        m_code_predictor.set_tensor("position_ids", pos_ids);
        m_code_predictor.set_tensor("step", step_tensor);
        m_code_predictor.set_tensor("seed", seed_tensor);
        m_code_predictor.set_tensor("temperature", temp_tensor);
        m_code_predictor.set_tensor("top_k", topk_tensor);
        m_code_predictor.set_tensor("beam_idx", beam_idx);
        m_code_predictor.infer();
        history_len += seq_len;

        // Graph emits the sampled code and its codec embedding (baked-in codec_embedding weights).
        codes.push_back(m_code_predictor.get_tensor("code").data<int64_t>()[0]);

        auto token_embed = m_code_predictor.get_tensor("token_embed");
        const auto* te_data = token_embed.data<float>();
        auto* se_data = step_embeds.data<float>();
        for (size_t i = 0; i < hs_size; i++) {
            se_data[i] = te_data[i];
            sum_data[i] += te_data[i];
        }
    }

    return {codes, m_cp_embed_sum};
}

ov::Tensor Qwen3OmniSpeechPipeline::codes_to_wav(const ov::Tensor& codes) {
    auto codes_shape = codes.get_shape();
    GENAI_DEBUG("Speech: code2wav input codes shape=[%zu, %zu, %zu]",
                codes_shape[0], codes_shape[1], codes_shape[2]);

    m_code2wav.set_tensor("codes", codes);
    m_code2wav.infer();
    auto waveform = m_code2wav.get_tensor("waveform");
    auto wav_shape = waveform.get_shape();

    GENAI_INFO("Speech: code2wav output waveform shape=[%zu, %zu, %zu], total_samples=%zu",
               wav_shape[0], wav_shape[1], wav_shape[2], waveform.get_size());

    ov::Tensor result(waveform.get_element_type(), waveform.get_shape());
    waveform.copy_to(result);
    return result;
}

TalkerResults Qwen3OmniSpeechPipeline::generate_speech(const std::vector<int64_t>& full_token_ids,
                                                       const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                                                       const OmniSpeechStreamerVariant& audio_streamer,
                                                       const OmniTalkerSpeechConfig& talker_speech_config) {
    // Stamp start_time for the speech-side perf record.
    const auto speech_start_time = std::chrono::steady_clock::now();
    auto build_result = [&](ov::Tensor waveform) -> TalkerResults {
        TalkerResults result;
        if (waveform && waveform.get_size() > 0) {
            result.perf_metrics.num_generated_samples = waveform.get_size();
            result.waveforms.push_back(std::move(waveform));
        }
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - speech_start_time).count();
        result.perf_metrics.generation_time_ms = static_cast<float>(duration_us) / 1000.0f;
        return result;
    };

    // Resolve sampling overrides up front: caller-supplied std::optional<...> takes precedence
    // over the JSON-loaded checkpoint defaults at m_config.{talker,cp}_*.
    const float talker_temp = talker_speech_config.talker_temperature.value_or(m_config.talker_temperature);
    const size_t talker_top_k_resolved = talker_speech_config.talker_top_k.value_or(m_config.talker_top_k);
    const float talker_rep_penalty =
        talker_speech_config.talker_repetition_penalty.value_or(m_config.talker_repetition_penalty);
    const float cp_temp = talker_speech_config.cp_temperature.value_or(m_config.cp_temperature);
    const size_t cp_top_k_resolved = talker_speech_config.cp_top_k.value_or(m_config.cp_top_k);
    if (talker_speech_config.cp_repetition_penalty) {
        // The single-step CodePredictor graph samples in-graph (Gumbel-max) and exposes only
        // step / seed / temperature / top_k, so a per-call repetition-penalty override has no
        // effect. Warn once so users don't silently assume it took.
        GENAI_WARN("Speech: cp_repetition_penalty override is ignored — the CodePredictor model "
                   "samples in-graph and has no repetition-penalty input.");
    }

    const size_t chunk_frames = talker_speech_config.audio_chunk_frames;
    OPENVINO_ASSERT(chunk_frames >= 1, "audio_chunk_frames must be >= 1 (got ", chunk_frames, ")");
    bool streaming = is_speech_streamer_active(audio_streamer);

    if (!m_talker_available) {
        GENAI_WARN("Speech: talker not available");
        if (streaming)
            end_speech_streamer(audio_streamer);
        return build_result(ov::Tensor{});
    }

    // Reseed at every entry so output depends only on inputs + seed, not prior call history.
    // Single shared stream across talker first-code sampling and all CodePredictor steps — one
    // seed fully reproduces the generated audio. Matches the reference torch.Generator contract.
    m_rng.seed(static_cast<std::mt19937::result_type>(talker_speech_config.rng_seed));

    GENAI_DEBUG("Speech: tokens=%zu, intermediate=%zu",
                full_token_ids.size(),
                all_intermediate_hidden_states.size());

    // Resolve speaker embedding from the variant: Tensor path takes the embedding directly;
    // string path looks up the named speaker in precomputed embeddings.
    ov::Tensor speaker_embed_to_use;
    if (std::holds_alternative<ov::Tensor>(talker_speech_config.speaker)) {
        const auto& tensor = std::get<ov::Tensor>(talker_speech_config.speaker);
        const auto& shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] == 1 &&
                            shape[2] == m_config.talker_hidden_size,
                        "speaker embedding must have shape [1, 1, ",
                        m_config.talker_hidden_size,
                        "], got ",
                        shape);
        speaker_embed_to_use = tensor;
    } else {
        const auto& speaker_name = std::get<std::string>(talker_speech_config.speaker);
        int64_t speaker_codec_id = resolve_speaker_id(speaker_name);
        speaker_embed_to_use = m_speaker_embed.at(speaker_codec_id);
    }

    auto [talker_input, trailing_text_hidden] =
        build_talker_input(full_token_ids, all_intermediate_hidden_states, speaker_embed_to_use);

    if (talker_input.get_shape()[1] == 0) {
        GENAI_WARN("Speech: build_talker_input returned empty, cannot generate speech");
        if (streaming)
            end_speech_streamer(audio_streamer);
        return build_result(ov::Tensor{});
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
    auto talker_max_tokens = talker_speech_config.max_new_tokens;
    if (talker_max_tokens == std::numeric_limits<std::size_t>::max()) {
        talker_max_tokens = m_config.talker_max_new_tokens;
    } else if (talker_max_tokens > m_config.talker_max_new_tokens) {
        GENAI_WARN("Speech: max_new_tokens (%zu) exceeds model maximum (%zu) — may degrade output quality or produce excess tokens",
                   talker_max_tokens, m_config.talker_max_new_tokens);
    }

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

    // Reuse the pre-computed tts_pad embedding (talker space) for per-step padding below.
    const auto& tts_pad_proj = m_tts_pad_embed;

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

    // talker_temp / talker_top_k_resolved / talker_rep_penalty resolved at function entry
    // from the OmniTalkerSpeechConfig overrides (or m_config defaults if unset).
    const auto& suppress_tokens = m_config.talker_suppress_tokens;
    std::vector<int64_t> generated_first_codes;

    // Reusable tensors for decode steps
    ov::Tensor next_input(ov::element::f32, {1, 1, hidden_size});
    ov::Tensor next_pos(ov::element::i64, {1, 1});

    GENAI_INFO("Speech: generating codec tokens (max %zu steps, temp=%.2f, top_k=%zu, rep_penalty=%.2f)...",
               talker_max_tokens,
               talker_temp,
               talker_top_k_resolved,
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
                                       talker_top_k_resolved,
                                       talker_rep_penalty,
                                       generated_first_codes,
                                       suppress_tokens);

        if (step < 3 || step % 100 == 0) {
            GENAI_DEBUG("Speech: step %zu, code=%lld (EOS=%lld)", step, (long long)first_code, (long long)m_config.codec_eos_token_id);
        }

        if (first_code == m_config.codec_eos_token_id) {
            GENAI_INFO("Speech: completed at step %zu (code=%lld matched EOS=%lld)", step, (long long)first_code, (long long)m_config.codec_eos_token_id);
            break;
        }

        generated_first_codes.push_back(first_code);

        // Get talker's last hidden state for CodePredictor
        auto hs_size = talker_hidden.get_shape().back();
        ov::Tensor last_hidden(ov::element::f32, {1, 1, hs_size});
        const auto* hs_data = talker_hidden.data<float>() + (talker_hidden.get_size() - hs_size);
        std::memcpy(last_hidden.data<float>(), hs_data, hs_size * sizeof(float));

        // Run CodePredictor for additional code groups
        auto [additional_codes, cp_embed_sum] =
            predict_codes(last_hidden, first_code, cp_temp, cp_top_k_resolved);

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

            auto status = invoke_speech_streamer(audio_streamer, chunk_wav);
            if (status == StreamingStatus::STOP || status == StreamingStatus::CANCEL) {
                GENAI_INFO("Speech: streaming %s at step %zu",
                           status == StreamingStatus::STOP ? "stopped" : "cancelled",
                           step);
                early_stop = true;
                break;
            }
        }

        auto* next_data = next_input.data<float>();
        auto* cp_data = cp_embed_sum.data<float>();
        for (size_t i = 0; i < hidden_size; i++) {
            next_data[i] = cp_data[i];
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
        invoke_speech_streamer(audio_streamer, chunk_wav);
    }

    // Always call end() on active streamer
    if (streaming) {
        end_speech_streamer(audio_streamer);
    }

    if (all_codes.size() == talker_max_tokens) {
        GENAI_WARN("Speech: reached max tokens (%zu) without EOS", talker_max_tokens);
    }

    if (all_codes.empty()) {
        GENAI_WARN("Speech: no codes generated");
        return build_result(ov::Tensor{});
    }

    GENAI_INFO("Speech: %zu codec steps generated, converting to waveform...", all_codes.size());

    auto full_codes = stack_codes_range(0, all_codes.size());
    return build_result(codes_to_wav(full_codes));
}

}  // namespace ov::genai
