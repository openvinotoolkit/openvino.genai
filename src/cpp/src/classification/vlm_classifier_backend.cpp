// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "classification/vlm_classifier_backend.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "utils.hpp"

namespace {
bool is_sequence_classification(const std::vector<std::string>& architectures) {
    const std::string suffix = "ForSequenceClassification";
    return std::any_of(architectures.begin(), architectures.end(), [&suffix](const std::string& architecture) {
        return architecture.size() >= suffix.size() &&
               architecture.compare(architecture.size() - suffix.size(), suffix.size(), suffix) == 0;
    });
}
}  // namespace

namespace ov::genai {

VLMClassifierBackend::VLMClassifierBackend(const std::filesystem::path& models_path,
                                           const std::string& device,
                                           const ov::AnyMap& properties,
                                           const ClassifierConfig& config) {
    OPENVINO_ASSERT(std::filesystem::exists(models_path / "openvino_language_model.xml"),
                    "ClassificationPipeline supports visual language classifiers only: openvino_language_model.xml "
                    "is not found in ",
                    models_path);
    OPENVINO_ASSERT(
        is_sequence_classification(config.architectures),
        "ClassificationPipeline expects a model exported as *ForSequenceClassification; config.json architectures of ",
        models_path,
        " do not contain such an entry");

    ov::AnyMap plugin_properties = properties;
    utils::extract_extensions_to_core(plugin_properties);

    m_inputs_embedder = std::make_shared<InputsEmbedder>(models_path, device, plugin_properties);
    // the text is the exact model input: no chat template and no extra special tokens are added
    m_inputs_embedder->set_prompt_formatting(PromptFormatting::AlreadyTemplated);

    std::shared_ptr<ov::Model> model = utils::singleton_core().read_model(models_path / "openvino_language_model.xml");
    const ov::PartialShape logits_shape = model->output("logits").get_partial_shape();
    OPENVINO_ASSERT(logits_shape.rank().is_static() && logits_shape.size() == 3 && logits_shape[2].is_static(),
                    "The classifier language model must output logits [batch, seq_len, num_labels], got ",
                    logits_shape);
    m_num_labels = logits_shape[2].get_length();

    ov::preprocess::PrePostProcessor processor(model);
    processor.output("logits").postprocess().custom([&model](const ov::Output<ov::Node>& logits) {
        return utils::last_token_pooling(logits, model->input("attention_mask"), false);
    });
    ov::CompiledModel compiled_model =
        utils::singleton_core().compile_model(processor.build(),
                                              device,
                                              utils::get_model_properties(plugin_properties, "language_model", device));
    utils::print_compiled_model_properties(compiled_model, "VLM classifier language model");
    m_request = compiled_model.create_infer_request();
    for (const auto& input : compiled_model.inputs()) {
        const auto& names = input.get_names();
        m_input_names.insert(names.begin(), names.end());
    }
}

size_t VLMClassifierBackend::num_labels() const {
    return m_num_labels;
}

bool VLMClassifierBackend::has_input(const std::string& name) const {
    return m_input_names.count(name) > 0;
}

ov::Tensor VLMClassifierBackend::infer(const ClassificationInputs& inputs) {
    m_inputs_embedder->get_cache_state().reset_state();

    const std::vector<EncodedImage> images = m_inputs_embedder->encode_images(inputs.images);
    const NormalizedPrompt prompt = m_inputs_embedder->normalize_prompt(inputs.text, 0, images);
    VLMPerfMetrics metrics;
    const ov::Tensor inputs_embeds =
        m_inputs_embedder->get_inputs_embeds(prompt.unified_prompt, images, metrics, true, prompt.images_sequence);

    const size_t seq_length = inputs_embeds.get_shape().at(1);
    ov::Tensor attention_mask(ov::element::i64, {1, seq_length});
    std::fill_n(attention_mask.data<int64_t>(), seq_length, int64_t{1});

    m_request.reset_state();
    m_request.set_tensor("inputs_embeds", inputs_embeds);
    m_request.set_tensor("attention_mask", attention_mask);

    if (has_input("position_ids")) {
        m_request.set_tensor("position_ids", m_inputs_embedder->get_position_ids(seq_length, 0).first);
    }

    if (has_input("beam_idx")) {
        ov::Tensor beam_idx(ov::element::i32, {1});
        beam_idx.data<int32_t>()[0] = 0;
        m_request.set_tensor("beam_idx", beam_idx);
    }

    for (const auto& [name, tensor] : m_inputs_embedder->get_lm_extra_inputs()) {
        if (tensor && tensor.get_size() > 0 && has_input(name)) {
            m_request.set_tensor(name, tensor);
        }
    }

    m_request.infer();

    const ov::Tensor logits = m_request.get_tensor("logits");
    OPENVINO_ASSERT(logits.get_shape() == ov::Shape({1, m_num_labels}),
                    "Expected pooled logits of shape [1, ",
                    m_num_labels,
                    "], got ",
                    logits.get_shape());

    // the request output is overwritten by the next infer()
    ov::Tensor result(logits.get_element_type(), logits.get_shape());
    logits.copy_to(result);
    return result;
}

}  // namespace ov::genai
