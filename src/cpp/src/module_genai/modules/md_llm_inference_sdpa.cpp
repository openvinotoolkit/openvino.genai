// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_OPENVINO_NEW_ARCH

#include "md_llm_inference_sdpa.hpp"

#include "module_genai/module_factory.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/chat_history.hpp"
#include "module_genai/utils/com_utils.hpp"

namespace ov {
namespace genai {
namespace module {

namespace {
bool dump_performance_enabled() {
    static const bool enabled = utils::check_env_variable("DUMP_PERFORMANCE");
    return enabled;
}

double elapsed_ms(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

GENAI_REGISTER_MODULE_SAME(LLMInferenceSDPAModule);

// ============================================================================
// Static YAML config description (for --print-config)
// ============================================================================

void LLMInferenceSDPAModule::print_static_config() {
    std::cout << R"(
global_context:
  model_type: "qwen3_5"
pipeline_modules:
  llm_inference_sdpa:
    type: "LLMInferenceSDPAModule"
    description: "LLM module using SDPA (stateful) backend — supports text & VL modes"
    device: "CPU"
    inputs:
      # ---- Text mode inputs (required) ----
      - name: "input_ids"            # Tokenized input ids
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      # ---- VL mode inputs (additional, optional) ----
      - name: "visual_embeds"        # [Optional] visual embeddings from vision encoder
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "visual_pos_mask"      # [Optional] boolean mask marking visual token positions
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "grid_thw"             # [Optional] grid dimensions [N,3] for 3D MRoPE position ids
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "generated_text"
        type: "String"
    params:
      model_path: "model_path"              # Directory containing config.json + IR files
      model_cfg_path: "model_config.json"   # Optional fallback when model_path is not provided
      max_new_tokens: "256"
      text_device: ""                       # Override device for text model (e.g. CPU for VL TDR avoidance)
    )" << std::endl;
}

// ============================================================================
// Construction / Destruction
// ============================================================================

LLMInferenceSDPAModule::LLMInferenceSDPAModule(const IBaseModuleDesc::PTR& desc,
                                                 const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize LLMInferenceSDPAModule");
    }
}

LLMInferenceSDPAModule::~LLMInferenceSDPAModule() {}

// ============================================================================
// Helpers (ported from md_qwen3_5_modeling.cpp)
// ============================================================================

std::string LLMInferenceSDPAModule::quant_suffix() const {
    using namespace ov::genai::modeling::weights;
    auto cfg = parse_quantization_config_from_env();
    if (!cfg.enabled()) {
        cfg.mode        = QuantizationConfig::Mode::INT4_ASYM;
        cfg.backup_mode = QuantizationConfig::Mode::INT4_ASYM;
        cfg.group_size  = 128;
    }
    auto tok = [](QuantizationConfig::Mode m) -> std::string {
        switch (m) {
            case QuantizationConfig::Mode::INT4_SYM:  return "4s";
            case QuantizationConfig::Mode::INT4_ASYM: return "4a";
            case QuantizationConfig::Mode::INT8_SYM:  return "8s";
            case QuantizationConfig::Mode::INT8_ASYM: return "8a";
            default:                                  return "n";
        }
    };
    return "_q" + tok(cfg.mode) + "_b" + tok(cfg.backup_mode) + "_g" + std::to_string(cfg.group_size);
}

bool LLMInferenceSDPAModule::has_ir_pair(const std::filesystem::path& xml,
                                          const std::filesystem::path& bin) {
    return std::filesystem::is_regular_file(xml) && std::filesystem::is_regular_file(bin);
}

bool LLMInferenceSDPAModule::has_model_input(const std::shared_ptr<ov::Model>& m,
                                              const std::string& name) {
    for (const auto& inp : m->inputs())
        if (inp.get_names().count(name))
            return true;
    return false;
}

ov::Tensor LLMInferenceSDPAModule::make_zeros(ov::element::Type t, ov::Shape shape) {
    ov::Tensor out(t, shape);
    std::memset(out.data(), 0, out.get_byte_size());
    return out;
}

ov::Tensor LLMInferenceSDPAModule::make_beam_idx(size_t batch) {
    ov::Tensor t(ov::element::i32, {batch});
    for (size_t i = 0; i < batch; ++i)
        t.data<int32_t>()[i] = static_cast<int32_t>(i);
    return t;
}

int64_t LLMInferenceSDPAModule::argmax_last(const ov::Tensor& logits) {
    const auto s = logits.get_shape();
    if (s.size() != 3 || s[0] != 1)
        throw std::runtime_error("logits must be [1,S,V]");
    const size_t offset = (s[1] - 1) * s[2];

    auto argmax = [&](auto* data, size_t n) -> int64_t {
        auto best = data[0];
        size_t idx = 0;
        for (size_t i = 1; i < n; ++i)
            if (data[i] > best) { best = data[i]; idx = i; }
        return static_cast<int64_t>(idx);
    };
    if (logits.get_element_type() == ov::element::f16)
        return argmax(logits.data<const ov::float16>()  + offset, s[2]);
    if (logits.get_element_type() == ov::element::bf16)
        return argmax(logits.data<const ov::bfloat16>() + offset, s[2]);
    if (logits.get_element_type() == ov::element::f32)
        return argmax(logits.data<const float>()        + offset, s[2]);
    throw std::runtime_error("Unsupported logits dtype");
}

// ============================================================================
// Initialization
// ============================================================================

bool LLMInferenceSDPAModule::initialize() {
    const auto& params = module_desc->params;

    // Resolve model directory
    std::filesystem::path models_path = get_optional_param("model_path");
    if (models_path.empty()) {
        models_path = get_param("model_cfg_path");
    }
    if (models_path.empty() || !std::filesystem::is_directory(models_path)) {
        GENAI_ERR("LLMInferenceSDPAModule: model_path is required and must be an existing directory");
        return false;
    }

    // Override max_new_tokens from params
    {
        auto it = params.find("max_new_tokens");
        if (it != params.end() && !it->second.empty()) {
            try { m_max_new_tokens = std::stoull(it->second); }
            catch (...) { GENAI_ERR("Failed to parse max_new_tokens"); }
        }
    }

    // Load model config
    try {
        m_model_config = ov::genai::modeling::models::Qwen3_5Config::from_json_file(models_path);
    } catch (const std::exception& e) {
        GENAI_ERR("Failed to load Qwen3.5 config from " + models_path.string() + ": " + e.what());
        return false;
    }

    // Resolve IR paths
    const std::string qs = quant_suffix();
    GENAI_INFO("LLMInferenceSDPAModule: quant suffix = " + qs);

    const auto text_xml    = models_path / ("qwen3_5_text"    + qs + ".xml");
    const auto text_bin    = models_path / ("qwen3_5_text"    + qs + ".bin");
    const auto text_vl_xml = models_path / ("qwen3_5_text_vl" + qs + ".xml");
    const auto text_vl_bin = models_path / ("qwen3_5_text_vl" + qs + ".bin");

    // Prefer VL IR (supports both text and VL modes at runtime).
    // Fall back to text-only IR when VL IR is not available.
    std::filesystem::path chosen_text_xml, chosen_text_bin;
    if (has_ir_pair(text_vl_xml, text_vl_bin)) {
        chosen_text_xml = text_vl_xml;
        chosen_text_bin = text_vl_bin;
        m_text_uses_vl_ir = true;
    } else if (has_ir_pair(text_xml, text_bin)) {
        chosen_text_xml = text_xml;
        chosen_text_bin = text_bin;
        m_text_uses_vl_ir = false;
        GENAI_INFO("VL text IR not found; using text-only IR (VL mode will not be available)");
    } else {
        GENAI_ERR("No text IR found. Expected: " + text_vl_xml.string() + " or " + text_xml.string());
        return false;
    }

    // Compile the text model
    GENAI_INFO("LLMInferenceSDPAModule: loading text IR: " + chosen_text_xml.string());
    auto text_ir = m_core.read_model(chosen_text_xml.string(), chosen_text_bin.string());

    // Verify VL inputs when using VL IR
    if (m_text_uses_vl_ir) {
        using IO = ov::genai::modeling::models::Qwen3_5TextIO;
        if (!has_model_input(text_ir, IO::kVisualEmbeds) ||
            !has_model_input(text_ir, IO::kVisualPosMask)) {
            GENAI_ERR("Text IR missing VL inputs (visual_embeds / visual_pos_mask)");
            return false;
        }
    }

    // Resolve text device — allow override via params
    std::string text_device = module_desc->device;
    {
        auto it = params.find("text_device");
        if (it != params.end() && !it->second.empty())
            text_device = it->second;
    }

    const ov::AnyMap latency_props = {
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::num_requests(1)
    };

    m_device = text_device;
    GENAI_INFO("LLMInferenceSDPAModule: compiling text -> " + text_device);
    m_compiled_text = m_core.compile_model(text_ir, text_device, latency_props);

    // Build tokenizer
    try {
        m_tokenizer = std::make_unique<ov::genai::Tokenizer>(models_path);
    } catch (const std::exception& e) {
        GENAI_ERR("LLMInferenceSDPAModule: tokenizer init failed: " + std::string(e.what()));
        // Non-fatal — upstream provides input_ids directly
    }

    // Collect stop token ids from all available sources:
    //   1) Tokenizer EOS token
    //   2) config.json text_config.eos_token_id (Qwen3.5 = 248044)
    if (m_tokenizer) {
        try {
            auto eid = m_tokenizer->get_eos_token_id();
            if (eid >= 0) m_stop_ids.insert(eid);
        } catch (...) {}
    }
    if (m_model_config.text.eos_token_id > 0) {
        m_stop_ids.insert(m_model_config.text.eos_token_id);
    }
    if (m_stop_ids.empty()) {
        GENAI_INFO("LLMInferenceSDPAModule: no stop token ids found — "
                   "decoding will run for the full max_new_tokens budget");
    }

    GENAI_INFO("LLMInferenceSDPAModule initialised (vl_ir=" +
               std::string(m_text_uses_vl_ir ? "true" : "false") +
               ", device=" + text_device + ")");
    return true;
}

// ============================================================================
// Text decode (no vision inputs) — stateful prefill + greedy decode
// ============================================================================

std::string LLMInferenceSDPAModule::run_text_decode(const ov::Tensor& input_ids,
                                                     const ov::Tensor& attention_mask,
                                                     const ov::Tensor& position_ids,
                                                     const ov::Tensor& rope_deltas) {
    using TIO = ov::genai::modeling::models::Qwen3_5TextIO;

    const size_t  batch      = input_ids.get_shape()[0];
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape()[1]);

    auto beam_idx = make_beam_idx(batch);
    auto text_req = m_compiled_text->create_infer_request();
    text_req.reset_state();

    // --- Prefill ---
    text_req.set_tensor(TIO::kInputIds,      input_ids);
    text_req.set_tensor(TIO::kAttentionMask, attention_mask);
    text_req.set_tensor(TIO::kPositionIds,   position_ids);
    text_req.set_tensor(TIO::kBeamIdx,       beam_idx);

    if (m_text_uses_vl_ir) {
        // Feed zero visual inputs for text-only usage of VL IR
        text_req.set_tensor(TIO::kVisualEmbeds,
            make_zeros(ov::element::f32, {batch, static_cast<size_t>(prompt_len),
                       static_cast<size_t>(m_model_config.text.hidden_size)}));
        text_req.set_tensor(TIO::kVisualPosMask,
            make_zeros(ov::element::boolean, {batch, static_cast<size_t>(prompt_len)}));
    }

    const auto t_prefill0 = std::chrono::steady_clock::now();
    text_req.infer();
    const auto t_prefill1 = std::chrono::steady_clock::now();
    int64_t next_id = argmax_last(text_req.get_tensor(TIO::kLogits));

    // --- Decode loop ---
    std::vector<int64_t> generated{next_id};
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask = make_zeros(ov::element::i64, {batch, 1});
    for (size_t b = 0; b < batch; ++b) step_mask.data<int64_t>()[b] = 1;

    ov::Tensor dec_vis, dec_vis_mask;
    if (m_text_uses_vl_ir) {
        dec_vis      = make_zeros(ov::element::f32,     {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)});
        dec_vis_mask = make_zeros(ov::element::boolean, {batch, 1});
    }

    int64_t past_len     = prompt_len;

    size_t decode_steps = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();

    for (size_t step = 1; step < m_max_new_tokens; ++step) {
        if (!m_stop_ids.empty() && m_stop_ids.count(next_id)) break;

        for (size_t b = 0; b < batch; ++b)
            step_ids.data<int64_t>()[b] = next_id;

        auto pos = ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(
            rope_deltas, past_len, 1);

        text_req.set_tensor(TIO::kInputIds,      step_ids);
        text_req.set_tensor(TIO::kAttentionMask, step_mask);
        text_req.set_tensor(TIO::kPositionIds,   pos);
        text_req.set_tensor(TIO::kBeamIdx,       beam_idx);
        if (m_text_uses_vl_ir) {
            text_req.set_tensor(TIO::kVisualEmbeds,  dec_vis);
            text_req.set_tensor(TIO::kVisualPosMask, dec_vis_mask);
        }

        text_req.infer();
        next_id = argmax_last(text_req.get_tensor(TIO::kLogits));
        generated.push_back(next_id);
        ++decode_steps;
        ++past_len;
    }

    const auto t_dec1 = std::chrono::steady_clock::now();

    if (dump_performance_enabled()) {
        const double ttft_ms    = elapsed_ms(t_prefill0, t_prefill1);
        const double decode_ms  = elapsed_ms(t_dec0, t_dec1);
        const double tpot_ms    = decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0;
        const double throughput = decode_steps > 0 && decode_ms > 0.0
                                   ? static_cast<double>(decode_steps) * 1000.0 / decode_ms
                                   : 0.0;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Mode: sdpa / text\n"
                  << "Device: " << m_device << "\n"
                  << "Prompt token size: " << prompt_len << "\n"
                  << "Output token size: " << generated.size() << "\n"
                  << "TTFT: " << ttft_ms << " ms\n"
                  << "Decode time: " << decode_ms << " ms\n";
        if (decode_steps > 0) {
            std::cout << "TPOT: " << tpot_ms << " ms/token\n"
                      << "Throughput: " << throughput << " tokens/s\n";
        } else {
            std::cout << "TPOT: N/A\nThroughput: N/A\n";
        }
    }

    // Decode tokens to text
    if (m_tokenizer) {
        return m_tokenizer->decode(generated, ov::genai::skip_special_tokens(true));
    } else {
        // Fallback: return token ids as text
        std::ostringstream oss;
        for (auto id : generated) oss << id << ' ';
        return oss.str();
    }
}

// ============================================================================
// VL decode (with visual embeddings) — stateful prefill + greedy decode
// ============================================================================

std::string LLMInferenceSDPAModule::run_vl_decode(const ov::Tensor& input_ids,
                                                    const ov::Tensor& attention_mask,
                                                    const ov::Tensor& position_ids,
                                                    const ov::Tensor& rope_deltas,
                                                    const ov::Tensor& visual_embeds,
                                                    const ov::Tensor& visual_pos_mask) {
    using TIO = ov::genai::modeling::models::Qwen3_5TextIO;

    const size_t  batch      = input_ids.get_shape()[0];
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape()[1]);

    auto beam_idx = make_beam_idx(batch);
    auto text_req = m_compiled_text->create_infer_request();
    text_req.reset_state();

    // --- Prefill ---
    text_req.set_tensor(TIO::kInputIds,      input_ids);
    text_req.set_tensor(TIO::kAttentionMask, attention_mask);
    text_req.set_tensor(TIO::kPositionIds,   position_ids);
    text_req.set_tensor(TIO::kBeamIdx,       beam_idx);
    text_req.set_tensor(TIO::kVisualEmbeds,  visual_embeds);
    text_req.set_tensor(TIO::kVisualPosMask, visual_pos_mask);

    const auto t_prefill0 = std::chrono::steady_clock::now();
    text_req.infer();
    const auto t_prefill1 = std::chrono::steady_clock::now();
    int64_t next_id = argmax_last(text_req.get_tensor(TIO::kLogits));

    // --- Decode loop ---
    std::vector<int64_t> generated{next_id};
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask = make_zeros(ov::element::i64, {batch, 1});
    for (size_t b = 0; b < batch; ++b) step_mask.data<int64_t>()[b] = 1;

    ov::Tensor dec_vis      = make_zeros(ov::element::f32,     {batch, 1, static_cast<size_t>(m_model_config.text.hidden_size)});
    ov::Tensor dec_vis_mask = make_zeros(ov::element::boolean, {batch, 1});

    int64_t past_len     = prompt_len;

    size_t decode_steps = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();

    for (size_t step = 1; step < m_max_new_tokens; ++step) {
        if (!m_stop_ids.empty() && m_stop_ids.count(next_id)) break;

        for (size_t b = 0; b < batch; ++b)
            step_ids.data<int64_t>()[b] = next_id;

        auto pos = ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(
            rope_deltas, past_len, 1);

        text_req.set_tensor(TIO::kInputIds,      step_ids);
        text_req.set_tensor(TIO::kAttentionMask, step_mask);
        text_req.set_tensor(TIO::kPositionIds,   pos);
        text_req.set_tensor(TIO::kBeamIdx,       beam_idx);
        text_req.set_tensor(TIO::kVisualEmbeds,  dec_vis);
        text_req.set_tensor(TIO::kVisualPosMask, dec_vis_mask);

        text_req.infer();
        next_id = argmax_last(text_req.get_tensor(TIO::kLogits));
        generated.push_back(next_id);
        ++decode_steps;
        ++past_len;
    }

    const auto t_dec1 = std::chrono::steady_clock::now();

    if (dump_performance_enabled()) {
        const double ttft_ms    = elapsed_ms(t_prefill0, t_prefill1);
        const double decode_ms  = elapsed_ms(t_dec0, t_dec1);
        const double tpot_ms    = decode_steps > 0 ? decode_ms / static_cast<double>(decode_steps) : 0.0;
        const double throughput = decode_steps > 0 && decode_ms > 0.0
                                   ? static_cast<double>(decode_steps) * 1000.0 / decode_ms
                                   : 0.0;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Mode: sdpa / vl\n"
                  << "Device: " << m_device << "\n"
                  << "Prompt token size: " << prompt_len << "\n"
                  << "Output token size: " << generated.size() << "\n"
                  << "TTFT: " << ttft_ms << " ms\n"
                  << "Decode time: " << decode_ms << " ms\n";
        if (decode_steps > 0) {
            std::cout << "TPOT: " << tpot_ms << " ms/token\n"
                      << "Throughput: " << throughput << " tokens/s\n";
        } else {
            std::cout << "TPOT: N/A\nThroughput: N/A\n";
        }
    }

    // Decode tokens to text
    if (m_tokenizer) {
        return m_tokenizer->decode(generated, ov::genai::skip_special_tokens(true));
    } else {
        std::ostringstream oss;
        for (auto id : generated) oss << id << ' ';
        return oss.str();
    }
}

// ============================================================================
// run() — module entry point
// ============================================================================

void LLMInferenceSDPAModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    // input_ids is required
    if (this->inputs.find("input_ids") == this->inputs.end()) {
        GENAI_ERR("LLMInferenceSDPAModule[" + module_desc->name +
                  "]: input_ids is required");
        return;
    }

    ov::Tensor input_ids = inputs["input_ids"].data.as<ov::Tensor>();

    // Build attention_mask internally — all 1s (no padding in single-request SDPA)
    const size_t batch   = input_ids.get_shape()[0];
    const size_t seq_len = input_ids.get_shape()[1];
    ov::Tensor attention_mask(ov::element::i64, {batch, seq_len});
    std::fill_n(attention_mask.data<int64_t>(), batch * seq_len, int64_t{1});

    // Determine VL mode: all three additional inputs must be present
    const bool is_vl = (this->inputs.find("visual_embeds") != this->inputs.end() &&
                        this->inputs.find("visual_pos_mask") != this->inputs.end() &&
                        this->inputs.find("grid_thw") != this->inputs.end() &&
                        this->inputs.find("position_ids") != this->inputs.end() &&
                        this->inputs.find("rope_delta") != this->inputs.end());

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(m_model_config);

    if (is_vl) {
        // ---- VL mode ----
        ov::Tensor visual_embeds   = inputs["visual_embeds"].data.as<ov::Tensor>();
        ov::Tensor visual_pos_mask = inputs["visual_pos_mask"].data.as<ov::Tensor>();
        ov::Tensor grid_thw        = inputs["grid_thw"].data.as<ov::Tensor>();
        ov::Tensor position_ids    = inputs["position_ids"].data.as<ov::Tensor>();
        ov::Tensor rope_delta      = inputs["rope_delta"].data.as<ov::Tensor>();


        std::string generated_text = run_vl_decode(input_ids, attention_mask,
                                                    position_ids, rope_delta,
                                                    visual_embeds, visual_pos_mask);
        GENAI_INFO("LLM output: " + generated_text);
        this->outputs["generated_text"].data = generated_text;
    } else {
        // ---- Text mode ----
        auto plan = planner.build_plan(input_ids, &attention_mask, nullptr);

        std::string generated_text = run_text_decode(input_ids, attention_mask,
                                                      plan.position_ids, plan.rope_deltas);
        GENAI_INFO("LLM output: " + generated_text);
        this->outputs["generated_text"].data = generated_text;
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov

#endif  // ENABLE_OPENVINO_NEW_ARCH
