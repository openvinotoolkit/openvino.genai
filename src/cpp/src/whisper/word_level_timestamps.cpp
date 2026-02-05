// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "word_level_timestamps.hpp"

#include <fstream>
#include <sstream>

#include "debug_utils.hpp"
#include "openvino/openvino.hpp"
#include "whisper/alignment_heads.hpp"
#include "whisper/transformations/scaled_dot_product_attention_decomposition.hpp"

namespace {

std::vector<ov::Tensor> median_filter_last_axis(const std::vector<ov::Tensor>& alignment_qks,
                                                const size_t filter_width = 7) {
    std::vector<ov::Tensor> filtered_tensors;
    const size_t pad_width = filter_width / 2;

    for (const auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        ov::Tensor filtered_tensor{ov::element::f32, shape};
        const auto* input_data = tensor.data<float>();
        auto* output_data = filtered_tensor.data<float>();

        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t seq = 0; seq < seq_len; ++seq) {
                for (size_t frame = 0; frame < frame_len; ++frame) {
                    std::vector<float> window;
                    for (int offset = -static_cast<int>(pad_width); offset <= static_cast<int>(pad_width); ++offset) {
                        int neighbor_frame = static_cast<int>(frame) + offset;
                        if (neighbor_frame >= 0 && neighbor_frame < static_cast<int>(frame_len)) {
                            size_t index = batch * seq_len * frame_len + seq * frame_len + neighbor_frame;
                            window.push_back(input_data[index]);
                        }
                    }

                    float median;
                    size_t mid = window.size() / 2;
                    if (window.size() % 2 == 0) {
                        // Even size: average of two middle elements
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                        float upper = window[mid];
                        // Find the max element in the lower partition (the other middle element)
                        float lower = *std::max_element(window.begin(), window.begin() + mid);
                        median = (lower + upper) / 2.0f;
                    } else {
                        std::nth_element(window.begin(), window.begin() + mid, window.end());
                        median = window[mid];
                    }

                    size_t output_index = batch * seq_len * frame_len + seq * frame_len + frame;
                    output_data[output_index] = median;
                }
            }
        }

        filtered_tensors.push_back(filtered_tensor);
    }

    return filtered_tensors;
}

// [head_size] * [batch,seq_len,frame_len]
// Apply softmax along frame axis (last dimension), matching: weights.softmax(dim=-1)
void softmax_frame_axis(std::vector<ov::Tensor>& alignment_qks) {
    for (auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        auto* data = tensor.data<float>();

        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t seq = 0; seq < seq_len; ++seq) {
                // Find max
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t frame = 0; frame < frame_len; ++frame) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    max_val = std::max(max_val, data[index]);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t frame = 0; frame < frame_len; ++frame) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    data[index] = std::exp(data[index] - max_val);
                    sum_exp += data[index];
                }

                // Normalize
                for (size_t frame = 0; frame < frame_len; ++frame) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    data[index] /= sum_exp;
                }
            }
        }
    }
}

// [head_size] * [batch,seq_len,frame_len]
// Standardize along token axis (seq_len), matching: (weights - mean) / std
// Python: std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
//         weights = (weights - mean) / std
void mean_normalize_token_axis(std::vector<ov::Tensor>& alignment_qks) {
    for (auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        auto* data = tensor.data<float>();

        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t frame = 0; frame < frame_len; ++frame) {
                // Compute mean along seq_len axis
                float sum = 0.0f;
                for (size_t seq = 0; seq < seq_len; ++seq) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    sum += data[index];
                }
                float mean = sum / seq_len;

                // Compute standard deviation along seq_len axis (unbiased=False, so divide by N)
                float sum_sq_diff = 0.0f;
                for (size_t seq = 0; seq < seq_len; ++seq) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    float diff = data[index] - mean;
                    sum_sq_diff += diff * diff;
                }
                float std = std::sqrt(sum_sq_diff / seq_len);

                // Avoid division by zero
                if (std < 1e-6f) {
                    std = 1e-6f;
                }

                // Standardize: (x - mean) / std
                for (size_t seq = 0; seq < seq_len; ++seq) {
                    size_t index = batch * seq_len * frame_len + seq * frame_len + frame;
                    data[index] = (data[index] - mean) / std;
                }
            }
        }
    }
}

// [head_size] * [batch,seq_len,frame_len] -> [head_size] * [seq_len,frame_len]
std::vector<ov::Tensor> reduce_batch_dim(const std::vector<ov::Tensor>& alignment_qks) {
    std::vector<ov::Tensor> result;
    for (const auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        ov::Tensor reduced_batch_tensor{ov::element::f32, {seq_len, frame_len}};
        const auto* input_data = tensor.data<float>();
        auto* output_data = reduced_batch_tensor.data<float>();

        std::memcpy(output_data, input_data, seq_len * frame_len * sizeof(float));

        result.push_back(reduced_batch_tensor);
    }
    return result;
}

// [head_size] * [seq_len,frame_len] -> [seq_len,frame_len]
// Averages across heads and negates for DTW cost minimization
// weights / weights.norm(dim=-2, keepdim=True)
std::vector<std::vector<float>> mean_across_heads(const std::vector<ov::Tensor>& alignment_qks) {
    if (alignment_qks.empty()) {
        return {};
    }

    const ov::Shape& shape = alignment_qks[0].get_shape();
    const size_t seq_len = shape[0];
    const size_t frame_len = shape[1];
    const size_t head_size = alignment_qks.size();

    std::vector<std::vector<float>> matrix(seq_len, std::vector<float>(frame_len, 0.0f));

    for (const auto& tensor : alignment_qks) {
        const auto* data = tensor.data<float>();
        for (size_t seq = 0; seq < seq_len; ++seq) {
            for (size_t frame = 0; frame < frame_len; ++frame) {
                size_t index = seq * frame_len + frame;
                matrix[seq][frame] += data[index];
            }
        }
    }

    // Average and negate
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t frame = 0; frame < frame_len; ++frame) {
            matrix[seq][frame] = -matrix[seq][frame] / static_cast<float>(head_size);
        }
    }

    return matrix;
}

// DTW implementation matching Python: alignment = dtw(-matrix.double().numpy())
// Input: negated attention matrix [seq_len, frame_len]
// Output: alignment path (token_indices, frame_indices)
std::vector<std::pair<size_t, size_t>> dtw_and_backtrace(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        return {};
    }

    const size_t N = matrix.size();     // seq_len (tokens)
    const size_t M = matrix[0].size();  // frame_len (audio frames)

    // Initialize cost and trace matrices
    std::vector<std::vector<float>> cost(N + 1, std::vector<float>(M + 1, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> trace(N + 1, std::vector<int>(M + 1, -1));

    cost[0][0] = 0.0f;

    // Set boundary conditions for backtrace (matching Python: trace[0, :] = 2 and trace[:, 0] = 1)
    for (size_t j = 0; j <= M; ++j) {
        trace[0][j] = 2;  // Move left along bottom edge
    }
    for (size_t i = 0; i <= N; ++i) {
        trace[i][0] = 1;  // Move up along left edge
    }

    // Forward pass: compute DTW cost
    // Python: for j in range(1, M + 1): for i in range(1, N + 1):
    for (size_t j = 1; j <= M; ++j) {
        for (size_t i = 1; i <= N; ++i) {
            float c0 = cost[i - 1][j - 1];  // diagonal
            float c1 = cost[i - 1][j];      // from top
            float c2 = cost[i][j - 1];      // from left

            // Python uses strict inequality: if c0 < c1 and c0 < c2
            float c;
            int t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }

            cost[i][j] = matrix[i - 1][j - 1] + c;
            trace[i][j] = t;
        }
    }

    // Backtracking: reconstruct optimal path
    std::vector<std::pair<size_t, size_t>> path;
    size_t i = N, j = M;

    while (i > 0 || j > 0) {
        // Python appends (i-1, j-1) before following trace direction
        // We need to handle underflow safely since i or j might be 0
        size_t path_i = (i > 0) ? i - 1 : 0;
        size_t path_j = (j > 0) ? j - 1 : 0;
        path.push_back({path_i, path_j});
        int t = trace[i][j];
        if (t == 0) {
            i--;
            j--;
        } else if (t == 1) {
            i--;
        } else if (t == 2) {
            j--;
        }
    }

    // Path is in reverse order (end to start), reverse it
    std::reverse(path.begin(), path.end());

    return path;
}

// [head_size] * [batch,seq_len,frame_len] -> [head_size] * [batch,seq_len,n_frames]
std::vector<ov::Tensor> extract_n_frames(const std::vector<ov::Tensor>& alignment_qks, const size_t n_frames) {
    std::vector<ov::Tensor> extracted_tensors;

    for (const auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        OPENVINO_ASSERT(n_frames <= frame_len, "Requested n_frames exceeds tensor frame length: ", frame_len);

        ov::Tensor extracted_tensor{ov::element::f32, {batch_size, seq_len, n_frames}};
        auto* input_data = tensor.data<float>();
        auto* output_data = extracted_tensor.data<float>();

        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t seq = 0; seq < seq_len; ++seq) {
                size_t input_offset = batch * seq_len * frame_len + seq * frame_len;
                size_t output_offset = batch * seq_len * n_frames + seq * n_frames;

                std::memcpy(output_data + output_offset, input_data + input_offset, n_frames * sizeof(float));
            }
        }

        extracted_tensors.push_back(extracted_tensor);
    }

    return extracted_tensors;
}

std::pair<std::vector<std::string>, std::vector<std::vector<int64_t>>> split_tokens_on_unicode(
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer) {
    const std::string decoded_full = tokenizer.decode(tokens, ov::genai::skip_special_tokens(false));
    const std::string replacement_char = u8"\uFFFD";

    std::vector<std::string> words;
    std::vector<std::vector<int64_t>> word_tokens;
    std::vector<int64_t> current_tokens;
    size_t unicode_offset = 0;

    for (const auto token : tokens) {
        current_tokens.push_back(token);
        const std::string decoded = tokenizer.decode(current_tokens, ov::genai::skip_special_tokens(false));

        const bool has_replacement_char = decoded.find(replacement_char) != std::string::npos;

        const bool decoded_full_has_replacement_char = decoded.find(replacement_char) != std::string::npos;

        const auto char_at_position = decoded_full[unicode_offset + decoded.find(replacement_char)];

        if (!has_replacement_char ||
            (decoded_full_has_replacement_char && std::string(1, char_at_position) == replacement_char)) {
            // Finalize current word
            words.push_back(decoded);
            word_tokens.push_back(current_tokens);
            current_tokens.clear();
            unicode_offset += decoded.size();
        }
    }

    return {words, word_tokens};
}

std::vector<ov::genai::WhisperWordTiming> match_words_to_alignment_path(
    const std::vector<std::string>& words,
    const std::vector<std::vector<int64_t>>& word_tokens,
    const std::vector<std::pair<size_t, size_t>>& alignment_path,
    const float chunk_time_offset) {
    // jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    std::vector<size_t> jumps_indicies;
    jumps_indicies.push_back(0);  // First element is always a jump
    for (size_t i = 1; i < alignment_path.size(); ++i) {
        auto prev_token_idx = alignment_path[i - 1].first;
        auto curr_token_idx = alignment_path[i].first;
        if (curr_token_idx != prev_token_idx) {
            jumps_indicies.push_back(i);
        }
    }

    std::vector<float> jump_times;
    for (const auto& jump_idx : jumps_indicies) {
        const auto frame_idx = alignment_path[jump_idx].second;
        jump_times.push_back(frame_idx * 0.02f);  // 20ms per frame
    }

    // word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    std::vector<size_t> word_boundaries;
    word_boundaries.push_back(0);
    for (size_t i = 0; i < word_tokens.size() - 1; ++i) {
        word_boundaries.push_back(word_boundaries.back() + word_tokens[i].size());
    }

    // begin_times = jump_times[word_boundaries[:-1]]
    // end_times = jump_times[word_boundaries[1:]]
    std::vector<ov::genai::WhisperWordTiming> word_timestamps;
    for (size_t i = 0; i < words.size() - 1; ++i) {
        const size_t begin_idx = word_boundaries[i];
        const size_t end_idx = word_boundaries[i + 1];
        const float start_time = jump_times[begin_idx] + chunk_time_offset;
        const float end_time = jump_times[end_idx] + chunk_time_offset;
        word_timestamps.push_back({words[i], word_tokens[i], start_time, end_time});
    }

    return word_timestamps;
};

std::vector<std::pair<size_t, size_t>> find_alignment_path(const std::vector<ov::Tensor>& alignment_heads_qks,
                                                           const size_t n_active_frames,
                                                           const std::vector<int64_t>& sot_tokens) {
    // Extract only up to n_frames to match input audio length
    auto n_frames_alignment_qks = extract_n_frames(alignment_heads_qks, size_t(n_active_frames / 2));

    softmax_frame_axis(n_frames_alignment_qks);

    // Apply L2 normalization along token axis (matching Python: weights / weights.norm(dim=-2, keepdim=True))
    mean_normalize_token_axis(n_frames_alignment_qks);

    auto filtered_alignment_qks = median_filter_last_axis(n_frames_alignment_qks, 7);

    const auto reduced_batch_tensor = reduce_batch_dim(filtered_alignment_qks);
    const auto matrix = mean_across_heads(reduced_batch_tensor);

    // matix shape: [sot_tokens.size():-1]
    auto matrix_text_tokens_slice =
        std::vector<std::vector<float>>(matrix.begin() + sot_tokens.size(), matrix.end() - 1);

    const auto alignment_path = dtw_and_backtrace(matrix_text_tokens_slice);

    return alignment_path;
}

// https://github.com/openai/whisper/blob/v20250625/whisper/timing.py#L307
void truncate_long_words_at_sentence_boundaries(std::vector<ov::genai::WhisperWordTiming>& words) {
    std::vector<float> word_durations;
    for (const auto& word : words) {
        float duration = word.end_ts - word.start_ts;
        if (duration > 0.0f) {
            word_durations.push_back(duration);
        }
    }

    if (word_durations.empty()) {
        return;
    }

    float median_duration;
    {
        std::vector<float> sorted_durations = word_durations;
        std::sort(sorted_durations.begin(), sorted_durations.end());
        size_t mid = sorted_durations.size() / 2;
        if (sorted_durations.size() % 2 == 0) {
            median_duration = (sorted_durations[mid - 1] + sorted_durations[mid]) / 2.0f;
        } else {
            median_duration = sorted_durations[mid];
        }
    }
    median_duration = std::min(0.7f, median_duration);
    float max_duration = median_duration * 2.0f;

    const std::string sentence_end_marks = ".。!！?？";
    for (size_t i = 1; i < words.size(); ++i) {
        float duration = words[i].end_ts - words[i].start_ts;
        if (duration > max_duration) {
            if (sentence_end_marks.find(words[i].word) != std::string::npos) {
                words[i].end_ts = words[i].start_ts + max_duration;
            } else if (sentence_end_marks.find(words[i - 1].word) != std::string::npos) {
                words[i].start_ts = words[i].end_ts - max_duration;
            }
        }
    }
}

// https://github.com/openai/whisper/blob/v20250625/whisper/timing.py#L245
std::vector<ov::genai::WhisperWordTiming> merge_punctuations(std::vector<ov::genai::WhisperWordTiming>& words) {
    if (words.size() < 2) {
        return words;
    }

    const std::string prepend_punctuations = "\"'“¿([{-";
    const std::string append_punctuations = "\"'.。,，!！?？:：”)]}、";

    // merge prepended punctuations
    size_t i = words.size() - 2;
    size_t j = words.size() - 1;
    while (i < words.size()) {
        auto& previous = words[i];
        auto& following = words[j];
        if (previous.word.size() > 1 && previous.word[0] == ' ' &&
            prepend_punctuations.find(previous.word.substr(1)) != std::string::npos) {
            // prepend it to the following word
            following.word = previous.word + following.word;
            following.token_ids.insert(following.token_ids.begin(),
                                       previous.token_ids.begin(),
                                       previous.token_ids.end());
            previous.word = "";
            previous.token_ids.clear();
        } else {
            j = i;
        }
        if (i == 0) {
            break;
        }
        i--;
    }

    // merge appended punctuations
    i = 0;
    j = 1;
    while (j < words.size()) {
        auto& previous = words[i];
        auto& following = words[j];
        if (!previous.word.empty() && previous.word.back() != ' ' && following.word.size() == 1 &&
            append_punctuations.find(following.word) != std::string::npos) {
            // append it to the previous word
            previous.word = previous.word + following.word;
            previous.token_ids.insert(previous.token_ids.end(), following.token_ids.begin(), following.token_ids.end());
            following.word = "";
            following.token_ids.clear();
        } else {
            i = j;
        }
        j++;
    }

    // Remove empty words
    std::vector<ov::genai::WhisperWordTiming> filtered_words;
    for (const auto& word : words) {
        if (!word.word.empty()) {
            filtered_words.push_back(word);
        }
    }

    return filtered_words;
}

std::string trim(const std::string& text) {
    std::string result = text;
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
                     return !std::isspace(ch);
                 }));

    result.erase(std::find_if(result.rbegin(),
                              result.rend(),
                              [](unsigned char ch) {
                                  return !std::isspace(ch);
                              })
                     .base(),
                 result.end());
    return result;
}

// https://github.com/openai/whisper/blob/v20250625/whisper/tokenizer.py#L311
std::pair<std::vector<std::string>, std::vector<std::vector<int64_t>>> split_tokens_on_spaces(
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer) {
    const auto [subwords, subword_tokens_list] = split_tokens_on_unicode(tokens, tokenizer);

    const int64_t eot = tokenizer.get_eos_token_id();

    std::vector<std::string> words;
    std::vector<std::vector<int64_t>> word_tokens;

    for (size_t i = 0; i < subwords.size(); ++i) {
        const std::string& subword = subwords[i];
        const std::vector<int64_t>& subword_tokens = subword_tokens_list[i];

        const bool is_special = subword_tokens.size() && subword_tokens[0] >= eot;
        const bool with_space = !subword.empty() && std::isspace(static_cast<unsigned char>(subword[0]));

        const std::string trimmed_subword = trim(subword);
        const bool is_punctuation = !trimmed_subword.empty() && trimmed_subword.size() == 1 &&
                                    std::ispunct(static_cast<unsigned char>(trimmed_subword[0]));

        if (words.empty() || is_special || with_space || is_punctuation) {
            words.push_back(subword);
            word_tokens.push_back(subword_tokens);
        } else {
            words.back() += subword;
            word_tokens.back().insert(word_tokens.back().end(), subword_tokens.begin(), subword_tokens.end());
        }
    }

    return {words, word_tokens};
}

std::vector<ov::Tensor> infer_alignments_heads_qks(const std::vector<int64_t>& tokens,
                                                   std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                   const ov::Tensor& hidden_state_tensor,
                                                   const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    const size_t batch_size = 1;

    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);

    const ov::Tensor input_ids_tensor{ov::element::i64,
                                      {batch_size, tokens.size()},
                                      const_cast<int64_t*>(tokens.data())};

    decoder->start_async(hidden_state_tensor, input_ids_tensor, beam_idx);
    decoder->wait();

    const auto& alignment_heads_qks = decoder->get_alignments_heads_qks(alignment_heads);
    decoder->reset_state();

    return alignment_heads_qks;
}

std::vector<ov::Tensor> infer_alignments_heads_qks(const std::vector<int64_t>& tokens,
                                                   ov::InferRequest& decoder,
                                                   const ov::Tensor& hidden_state_tensor,
                                                   const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    hidden_state_tensor.copy_to(decoder.get_tensor("encoder_hidden_states"));
    // NB: input_ids format: [token1, token2, pad, pad]
    auto padded_input_ids = decoder.get_tensor("input_ids");
    OPENVINO_ASSERT(padded_input_ids.get_size() >= tokens.size());
    OPENVINO_ASSERT(padded_input_ids.get_element_type() == ov::element::i64);
    std::fill_n(padded_input_ids.data<int64_t>(), padded_input_ids.get_size(), 0u);
    std::copy_n(&tokens[0], tokens.size(), padded_input_ids.data<int64_t>());

    // NB: attention_mask format: [1, 1, 0, 0]
    auto padded_attention_mask = decoder.get_tensor("attention_mask");
    OPENVINO_ASSERT(padded_attention_mask.get_size() >= tokens.size());
    auto* padded_mask_data = padded_attention_mask.data<int64_t>();
    std::fill_n(padded_mask_data, padded_attention_mask.get_size(), 0u);
    std::fill_n(padded_mask_data, tokens.size(), 1u);

    decoder.infer();

    return ov::genai::get_whisper_alignments_heads_qks(decoder, alignment_heads);
}

}  // namespace

namespace ov::genai {

std::vector<ov::genai::WhisperWordTiming> add_word_level_timestamps(const std::vector<int64_t>& sot_tokens,
                                                                    const std::vector<int64_t>& input_tokens,
                                                                    ov::genai::Tokenizer& tokenizer,
                                                                    std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                                    const ov::Tensor& hidden_state_tensor,
                                                                    const ov::genai::WhisperGenerationConfig& config,
                                                                    const size_t n_active_frames,
                                                                    const float chunk_time_offset) {
    // [text_tokens] + [eos_token]
    std::vector<int64_t> text_tokens;
    for (const auto& token : input_tokens) {
        if (token < config.eos_token_id) {
            text_tokens.push_back(token);
        }
    }
    text_tokens.push_back(config.eos_token_id);

    // [sot_tokens] + [no_timestamps_token] + [text_tokens] + [eos_token]
    std::vector<int64_t> infer_tokens = sot_tokens;
    infer_tokens.push_back(config.no_timestamps_token_id);
    infer_tokens.insert(infer_tokens.end(), text_tokens.begin(), text_tokens.end());

    auto alignment_heads_qks =
        infer_alignments_heads_qks(infer_tokens, decoder, hidden_state_tensor, config.alignment_heads);

    const auto alignment_path = find_alignment_path(alignment_heads_qks, n_active_frames, sot_tokens);

    const auto [words, word_tokens] = split_tokens_on_spaces(text_tokens, tokenizer);

    auto words_timestamps = match_words_to_alignment_path(words, word_tokens, alignment_path, chunk_time_offset);

    truncate_long_words_at_sentence_boundaries(words_timestamps);

    auto merged_timestamps = merge_punctuations(words_timestamps);

    return merged_timestamps;
}

std::vector<ov::genai::WhisperWordTiming> add_word_level_timestamps(const std::vector<int64_t>& sot_tokens,
                                                                    const std::vector<int64_t>& input_tokens,
                                                                    ov::genai::Tokenizer& tokenizer,
                                                                    ov::InferRequest& decoder,
                                                                    const ov::Tensor& hidden_state_tensor,
                                                                    const ov::genai::WhisperGenerationConfig& config,
                                                                    const size_t n_active_frames,
                                                                    const float chunk_time_offset) {
    // [text_tokens] + [eos_token]
    std::vector<int64_t> text_tokens;
    for (const auto& token : input_tokens) {
        if (token < config.eos_token_id) {
            text_tokens.push_back(token);
        }
    }
    text_tokens.push_back(config.eos_token_id);

    // [sot_tokens] + [no_timestamps_token] + [text_tokens] + [eos_token]
    std::vector<int64_t> infer_tokens = sot_tokens;
    infer_tokens.push_back(config.no_timestamps_token_id);
    infer_tokens.insert(infer_tokens.end(), text_tokens.begin(), text_tokens.end());

    auto alignment_heads_qks =
        infer_alignments_heads_qks(infer_tokens, decoder, hidden_state_tensor, config.alignment_heads);

    const auto alignment_path = find_alignment_path(alignment_heads_qks, n_active_frames, sot_tokens);

    const auto [words, word_tokens] = split_tokens_on_spaces(text_tokens, tokenizer);

    auto words_timestamps = match_words_to_alignment_path(words, word_tokens, alignment_path, chunk_time_offset);

    truncate_long_words_at_sentence_boundaries(words_timestamps);

    auto merged_timestamps = merge_punctuations(words_timestamps);

    return merged_timestamps;
}

void decompose_scaled_dot_product_attention_for_whisper(std::shared_ptr<ov::Model> model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::genai::WhisperScaledDotProductAttentionDecomposition>();
    auto result = manager.run_passes(model);
}

void add_cross_attention_qk_scaled_scores_outputs_for_whisper(std::shared_ptr<ov::Model> model) {
    size_t idx = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (op->get_type_info().name != std::string("Add")) {
            continue;
        }

        bool should_skip_op = true;

        for (const auto& output : op->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find("cross_attention_qk_scaled_scores") != std::string::npos) {
                    should_skip_op = false;
                    break;
                }
            }

            // output found, exit outputs loop
            if (!should_skip_op) {
                break;
            }
        }

        if (should_skip_op) {
            continue;
        }

        model->add_output(op->output(0)).add_names({"cross_attention_qk_scaled_scores_" + std::to_string(idx)});
        idx++;
    }
}
}  // namespace ov::genai
