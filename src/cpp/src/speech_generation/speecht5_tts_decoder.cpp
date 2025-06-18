// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speecht5_tts_decoder.hpp"

#include <algorithm>

#include "debug_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace ov;
using namespace std;

namespace {
void patch_model_with_spectrogram(std::shared_ptr<ov::Model>& speecht5_tts_decoder) {
    auto spectrogram_param = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    speecht5_tts_decoder->add_parameters(ParameterVector{spectrogram_param});
    spectrogram_param->output(0).set_names({"spectrogram"});
    auto spectrum =
        speecht5_tts_decoder->output("spectrum").get_node_shared_ptr()->input_value(0).get_node_shared_ptr();

    auto unsqueeze_dim = op::v0::Constant::create(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto unsqueeze_spectrum = make_shared<op::v0::Unsqueeze>(spectrum, unsqueeze_dim);
    auto spectrogram_out = make_shared<op::v0::Concat>(OutputVector{spectrogram_param, unsqueeze_spectrum}, 0);
    spectrogram_out->output(0).set_names({"spectrogram_out"});
    auto spectrogram_res = make_shared<op::v0::Result>(spectrogram_out);
    speecht5_tts_decoder->add_results(ResultVector{spectrogram_res});
}
}  // namespace

namespace ov::genai {
ov::Tensor SpeechT5TTSDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    try {
        return m_request.get_compiled_model().get_context().create_host_tensor(element_type, shape);
    } catch (std::exception& ex) {
        return ov::Tensor(element_type, shape);
    }
}

SpeechT5TTSDecoder::SpeechT5TTSDecoder(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);
    patch_model_with_spectrogram(model);
    auto compiled_model = core.compile_model(model, device, properties);

    utils::print_compiled_model_properties(compiled_model, "speecht5_tts decoder model");
    m_request = compiled_model.create_infer_request();
    m_beam_idx_tensor = create_host_tensor(ov::element::i32, {1});
    m_beam_idx_tensor.data<int32_t>()[0] = 0;
}

std::shared_ptr<SpeechT5TTSDecoder> SpeechT5TTSDecoder::from_path(const std::filesystem::path& models_path,
                                                                  const std::string& device,
                                                                  const ov::AnyMap& properties) {
    return std::make_shared<SpeechT5TTSDecoder>(models_path, device, properties);
}

void SpeechT5TTSDecoder::start_async(const Tensor& inputs_embeds,
                                     const Tensor& speaker_embeddings,
                                     const Tensor& encoder_hidden_states,
                                     const Tensor& encoder_attention_mask,
                                     const Tensor& spectrogram) {
    m_request.set_tensor("inputs_embeds", inputs_embeds);
    m_request.set_tensor("speaker_embeddings", speaker_embeddings);
    m_request.set_tensor("encoder_hidden_states", encoder_hidden_states);
    m_request.set_tensor("encoder_attention_mask", encoder_attention_mask);
    m_request.set_tensor("beam_idx", m_beam_idx_tensor);
    m_request.set_tensor("spectrogram", spectrogram);
    m_request.start_async();
};

std::tuple<Tensor, Tensor, Tensor, Tensor> SpeechT5TTSDecoder::wait() {
    m_request.wait();
    auto out_seq = m_request.get_tensor("output_sequence_out");
    auto spectrum = m_request.get_tensor("spectrum");
    auto prob = m_request.get_tensor("prob");
    auto spectrogram_out = m_request.get_tensor("spectrogram_out");
    return std::make_tuple(out_seq, spectrum, prob, spectrogram_out);
}

void SpeechT5TTSDecoder::reset_state() {
    m_request.reset_state();
};

}  // namespace ov::genai
