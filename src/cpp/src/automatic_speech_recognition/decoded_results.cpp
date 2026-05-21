// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {
ASRDecodedResults::operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

ASRDecodedResults::operator std::vector<std::string>() const {
    return texts;
}

std::ostream& operator<<(std::ostream& os, const ASRDecodedResults& dr) {
    OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(),
                    "The number of scores and texts doesn't match in ASRDecodedResults.");
    if (dr.texts.empty()) {
        return os;
    }
    if (dr.texts.size() == 1) {
        os << dr.texts[0];
        return os;
    }
    for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
        os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
    }
    return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
}
}  // namespace ov::genai
