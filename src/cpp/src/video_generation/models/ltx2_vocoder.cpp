// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "video_generation/models/ltx2_vocoder.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"

using namespace ov::genai;

LTX2Vocoder::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    utils::read_json_param(data, "output_sampling_rate", output_sampling_rate);
}

LTX2Vocoder::LTX2Vocoder(const std::filesystem::path& root_dir) : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2Vocoder::LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties)
    : LTX2Vocoder(root_dir) {
    compile(device, properties);
}

LTX2Vocoder::LTX2Vocoder(const LTX2Vocoder&) = default;

std::shared_ptr<LTX2Vocoder> LTX2Vocoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2Vocoder must have exactly one of m_model or m_request initialized");

    std::shared_ptr<LTX2Vocoder> cloned = std::make_shared<LTX2Vocoder>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const LTX2Vocoder::Config& LTX2Vocoder::get_config() const {
    return m_config;
}

LTX2Vocoder& LTX2Vocoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 vocoder model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

LTX2Vocoder& LTX2Vocoder::reshape(int64_t batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    m_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor LTX2Vocoder::infer(const ov::Tensor& mel_spectrogram) {
    OPENVINO_ASSERT(m_request, "Vocoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_input_tensor(mel_spectrogram);
    m_request.infer();

    // Copy to an owned tensor — the waveform is returned to the user and must outlive the request buffer
    const ov::Tensor output = m_request.get_output_tensor();
    ov::Tensor waveform(output.get_element_type(), output.get_shape());
    output.copy_to(waveform);
    return waveform;
}
