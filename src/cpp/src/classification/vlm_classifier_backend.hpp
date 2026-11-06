// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "classification/classifier_backend.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// Visual language classifier: InputsEmbedder merges the text and media embeddings, the language model ends in the
/// classification head and outputs [batch, seq_len, num_labels], reduced to the last token in the compiled graph.
class VLMClassifierBackend final : public ClassifierBackend {
public:
    VLMClassifierBackend(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties,
                         const ClassifierConfig& config);

    size_t num_labels() const override;

    ov::Tensor infer(const ClassificationInputs& inputs) override;

private:
    bool has_input(const std::string& name) const;

    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    ov::InferRequest m_request;
    std::unordered_set<std::string> m_input_names;
    size_t m_num_labels = 0;
};

}  // namespace ov::genai
