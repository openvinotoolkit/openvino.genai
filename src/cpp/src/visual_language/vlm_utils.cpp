// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vlm_utils.hpp"
#include "openvino/core/except.hpp"
#include "utils.hpp"

#include <algorithm>

namespace ov::genai::vlm_utils {

std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text,
                                                             ov::genai::Tokenizer& tokenizer,
                                                             const std::regex& native_pattern) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{
        prefix_begin,
        text.end(),
        native_pattern,
        {0, 1}  // Every match emits two values: whole match and submatch
    }; iter != std::sregex_token_iterator{}; ++iter) {
        if (is_submatch) {
            size_t idx = std::stoul(iter->str());
            OPENVINO_ASSERT(idx != 0);
            tokenized.push_back(idx - 1);
        } else {
            std::string regular_text{prefix_begin, iter->first};
            if (!regular_text.empty()) {
                tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
            }
            prefix_begin = iter->second;
        }
        is_submatch = !is_submatch;
    }
    std::string regular_text{prefix_begin, text.end()};
    if (!regular_text.empty()) {
        tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
    }
    return tokenized;
}

ov::Tensor insert_image_placeholders(const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
                                     const std::vector<size_t>& tokens_per_images) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(ov::genai::utils::overloaded{[](const ov::Tensor& token_chunk) {
                                                                      return token_chunk.get_shape().at(1);
                                                                  },
                                                                  [&](size_t image_id) {
                                                                      return tokens_per_images.at(image_id);
                                                                  }},
                                   chunk);
    }

    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        const size_t written = std::visit(ov::genai::utils::overloaded{[&](const ov::Tensor& token_chunk) {
                                                                            const size_t length = token_chunk.get_shape().at(1);
                                                                            std::copy_n(token_chunk.data<int64_t>(), length, merged.data<int64_t>() + offset);
                                                                            return length;
                                                                        },
                                                                        [&](size_t image_id) {
                                                                            const int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                                                                            std::fill_n(merged.data<int64_t>() + offset,
                                                                                        tokens_per_images.at(image_id),
                                                                                        fill_value);  // -1 to distinguish 0 token and 0 image id.
                                                                            return tokens_per_images.at(image_id);
                                                                        }},
                                         chunk);
        offset += written;
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    OPENVINO_ASSERT(tokens.get_shape().size() == 2 && tokens.get_shape()[0] == 1,
                    "Expected token tensor with shape [1, N], got ",
                    tokens.get_shape());
    OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64,
                    "Expected token tensor element type i64, got ",
                    tokens.get_element_type());

    const size_t full_length = tokens.get_shape().at(1);
    if (full_length == 0) {
        return chunks;
    }

    const int64_t* tokens_ptr = tokens.data<const int64_t>();
    int64_t last_token = tokens_ptr[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < full_length; ++offset) {
            // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
            // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
            // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
            // If last_token and next_token are negative, it's continuation of an image placeholder - skip
            // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
        const int64_t next_token = tokens_ptr[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            chunks.emplace_back(std::in_place_type<ov::Tensor>,
                                ov::element::i64,
                                ov::Shape{1, offset - text_start},
                                const_cast<int64_t*>(tokens_ptr + text_start));
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }

    if (last_token >= 0) {
        chunks.emplace_back(std::in_place_type<ov::Tensor>,
                            ov::element::i64,
                            ov::Shape{1, full_length - text_start},
                            const_cast<int64_t*>(tokens_ptr + text_start));
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

void merge_audio_embeddings(ov::Tensor& input_embeds,
                            const std::vector<int64_t>& input_ids,
                            const std::vector<EncodedAudio>& audios,
                            const std::vector<size_t>& audios_sequence,
                            int64_t audio_token_id) {
    if (audios_sequence.empty()) {
        return;
    }
    OPENVINO_ASSERT(audio_token_id >= 0,
                    "Audio was provided but config.json defines no audio_token_id, so the audio placeholders "
                    "cannot be located. Re-export the model.");

    // Checked up front because the zero-token skip loops below index audios before the
    // per-run checks would catch a bad index.
    for (size_t audio_id : audios_sequence) {
        OPENVINO_ASSERT(audio_id < audios.size(),
                        "Audio index ",
                        audio_id,
                        " is out of range for ",
                        audios.size(),
                        " provided audios. The sequence must be rebased to this turn's audios.");
    }

    const auto& shape = input_embeds.get_shape();
    const auto seq_len = shape[1];
    const auto hidden_size = shape[2];

    OPENVINO_ASSERT(input_ids.size() >= seq_len,
                    "input_ids size (",
                    input_ids.size(),
                    ") must be >= embedding seq_len (",
                    seq_len,
                    "). Ensure input_ids are not from a stale or re-tokenized source.");

    auto* embed_data = input_embeds.data<float>();
    const size_t bytes_per_token = hidden_size * sizeof(float);

    size_t run_index = 0;
    for (size_t i = 0; i < seq_len;) {
        if (input_ids[i] != audio_token_id) {
            i++;
            continue;
        }

        size_t run_end = i;
        while (run_end < seq_len && input_ids[run_end] == audio_token_id) {
            run_end++;
        }
        const size_t run_length = run_end - i;

        // A zero-token audio expands to start+end with no pads, so it contributes no run. Skip
        // over those entries or run k stops lining up with audios_sequence[k].
        while (run_index < audios_sequence.size() && audios[audios_sequence[run_index]].num_audio_tokens == 0) {
            run_index++;
        }

        OPENVINO_ASSERT(run_index < audios_sequence.size(),
                        "Found more audio placeholder runs in the prompt than provided audios (",
                        audios_sequence.size(),
                        "). A literal audio marker in user text would do this.");

        const auto& audio = audios[audios_sequence[run_index]];

        OPENVINO_ASSERT(run_length == audio.num_audio_tokens,
                        "Audio placeholder run ",
                        run_index,
                        " holds ",
                        run_length,
                        " tokens but audio ",
                        audios_sequence[run_index],
                        " encodes to ",
                        audio.num_audio_tokens,
                        ". The prompt expansion and the encoder disagree - this is an internal "
                        "inconsistency, please report it with the prompt that triggered it.");

        const auto audio_hidden_size = audio.audio_features.get_shape()[1];
        OPENVINO_ASSERT(audio_hidden_size == hidden_size,
                        "Audio embedding hidden_size (",
                        audio_hidden_size,
                        ") must match input embedding hidden_size (",
                        hidden_size,
                        "). Check that audio encoder output dimension matches the language model.");

        const auto* audio_data = audio.audio_features.data<const float>();
        for (size_t t = 0; t < run_length; t++) {
            std::memcpy(embed_data + (i + t) * hidden_size, audio_data + t * hidden_size, bytes_per_token);
        }

        run_index++;
        i = run_end;
    }

    // Trailing zero-token audios also have no run, so consume them before the final check.
    while (run_index < audios_sequence.size() && audios[audios_sequence[run_index]].num_audio_tokens == 0) {
        run_index++;
    }

    OPENVINO_ASSERT(run_index == audios_sequence.size(),
                    "Placed ",
                    run_index,
                    " of ",
                    audios_sequence.size(),
                    " audios: the prompt has fewer audio placeholder runs than tags resolved. This is an "
                    "internal inconsistency, please report it with the prompt that triggered it.");
}

}  // namespace ov::genai::vlm_utils
