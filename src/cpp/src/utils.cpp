// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

namespace ov {
namespace generate_utils {

Tensor init_attention_mask(Tensor& position_ids) {
    auto shape = position_ids.get_shape();
    auto attention_mask = ov::Tensor{position_ids.get_element_type(), shape};
    std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);
    return attention_mask;
}

void print_tensor(const ov::Tensor& tensor) {
    std::vector<int64_t> res;

    auto t_shape = tensor.get_shape();
    std::cout << "[";
    for (size_t i = 0; i < t_shape[1]; ++i) {
        if (tensor.get_element_type() == ov::element::i64) {
            res.emplace_back(tensor.data<int64_t>()[i]);
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
    }
    std::cout << "]" << std::endl;
}

bool is_xml(const std::string& path) { return path.compare(path.length() - 4, 4, ".xml") == 0;}

std::pair<int64_t, float> softmax(const ov::Tensor& logits, const size_t batch_idx) {
    if (logits.get_shape()[0] <= batch_idx) {
        OPENVINO_THROW("logits batch size doesn't match the number of beams");
    }

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    const float* logits_data = logits.data<const float>() + batch_offset + sequence_offset;
    
    int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    float max_logit = logits_data[out_token];

    float log_sum = std::log(
        std::accumulate(logits_data, logits_data + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
        }));
    return {out_token, log_sum};
}

}  // namespace generate_utils
}  // namespace ov