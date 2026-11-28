// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <variant>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/text_streamer.hpp"

namespace ov::genai::utils {

// Wraps a text generation call so that it fills Results::parsed: from a TextParserStreamer passed as
// the streamer (reset before generation), then by applying GenerationConfig::parsers to each text.
// Results is DecodedResults or a type derived from it (e.g. VLMDecodedResults).
template <typename Results>
Results run_generate_with_parsers(const OptionalGenerationConfig& generation_config,
                                  const StreamerVariant& streamer,
                                  const std::function<Results(void)>& generate_callable) {
    std::shared_ptr<TextParserStreamer> parser_streamer;
    // If streamer is of StreamerBase type, and it is TextParserStreamer, get parsed message
    // Streaming is available only for batch size 1 therefore only parsed[0]
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
        parser_streamer = std::dynamic_pointer_cast<TextParserStreamer>(*streamer_obj);
    }

    // TODO: Determine 'need_to_reset_parser' from generation_config when available.
    bool need_to_reset_parser = true;
    if (parser_streamer && need_to_reset_parser) {
        parser_streamer->reset();
    }

    Results res = generate_callable();

    if (parser_streamer) {
        res.parsed.resize(1);
        res.parsed[0] = parser_streamer->get_parsed_message();
    }

    // If no parsers are defined, return
    if (!generation_config.has_value() || generation_config->parsers.empty()) {
        return res;
    }

    const std::vector<std::shared_ptr<Parser>>& parsers = generation_config->parsers;
    res.parsed.resize(res.texts.size());

    // Apply Base parsers sequentially even if IncrementalParser has run.
    for (size_t i = 0; i < res.texts.size(); ++i) {
        auto& msg = res.parsed[i];
        if (!msg.contains("content")) {
            // Initialize msg with content
            msg["content"] = res.texts[i];
        }

        for (auto& parser : parsers) {
            parser->parse(msg);
        }
        res.parsed[i] = msg;
    }
    return res;
}

}  // namespace ov::genai::utils
