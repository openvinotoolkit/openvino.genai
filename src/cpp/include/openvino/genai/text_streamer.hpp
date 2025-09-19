// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

using CallbackTypeVariant = std::variant<bool, StreamingStatus>;

/**
 * @brief TextStreamer is used to decode tokens into text and call a user-defined callback function.
 *
 * @param tokenizer Tokenizer object to decode tokens into text.
 * @param callback User-defined callback function to process the decoded text, callback should return
 * either boolean flag or StreamingStatus.
 * @param detokenization_params AnyMap with detokenization parameters, e.g. ov::genai::skip_special_tokens(...)
 */
class OPENVINO_GENAI_EXPORTS TextStreamer : public StreamerBase {
public:
    StreamingStatus write(int64_t token) override;
    StreamingStatus write(const std::vector<int64_t>& tokens) override;

    void end() override;

    TextStreamer(const Tokenizer& tokenizer, std::function<CallbackTypeVariant(std::string)> callback, const ov::AnyMap& detokenization_params = {});

private:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    std::vector<int64_t> m_decoded_lengths;
    size_t m_printed_len = 0;
    ov::AnyMap m_additional_detokenization_params;

    StreamingStatus set_streaming_status(CallbackTypeVariant callback_status);

    std::function<CallbackTypeVariant(std::string)> m_subword_callback = [](std::string words) -> bool {
        return false;
    };

    StreamingStatus run_callback_if_needed(const std::string& text);

    void compute_decoded_length_for_position(size_t cache_position);
};

}  // namespace genai
}  // namespace ov
