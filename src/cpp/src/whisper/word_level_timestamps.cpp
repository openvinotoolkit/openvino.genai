// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "word_level_timestamps.hpp"

#include <fstream>
#include <sstream>

namespace {

std::vector<ov::Tensor> extract_qks_alignment_heads(const std::vector<ov::Tensor>& encoder_attention_qks,
                                                    const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    std::vector<ov::Tensor> alignment_qks;
    for (size_t i = 0; i < alignment_heads.size(); ++i) {
        const auto& [layer_idx, head_idx] = alignment_heads[i];

        ov::Tensor alignemnt_tensor = encoder_attention_qks.at(layer_idx);

        // [batch, head_num, seq_len, frame_len]
        const ov::Shape& alignment_shape = alignemnt_tensor.get_shape();

        // [batch, seq_len, frame_len]
        ov::Tensor head_tensor{ov::element::f32, {alignment_shape[0], alignment_shape[2], alignment_shape[3]}};
        auto* alignment_data = alignemnt_tensor.data<float>();
        auto* head_data = head_tensor.data<float>();
        const size_t batch_size = alignment_shape[0];
        const size_t head_num = alignment_shape[1];
        const size_t seq_len = alignment_shape[2];
        const size_t frame_len = alignment_shape[3];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            const size_t batch_offset = batch * head_num * seq_len * frame_len;
            const size_t head_offset = head_idx * seq_len * frame_len;
            const size_t head_batch_offset = batch * seq_len * frame_len;

            std::memcpy(head_data + head_batch_offset,
                        alignment_data + batch_offset + head_offset,
                        seq_len * frame_len * sizeof(float));
        }

        alignment_qks.push_back(head_tensor);
    }

    return alignment_qks;
}

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
                    std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                    size_t output_index = batch * seq_len * frame_len + seq * frame_len + frame;
                    output_data[output_index] = window[window.size() / 2];
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
                // Find max for numerical stability
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
// Takes first batch only
std::vector<ov::Tensor> shrink_batch_dim(const std::vector<ov::Tensor>& alignment_qks) {
    std::vector<ov::Tensor> shrunk_tensors;
    for (const auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        ov::Tensor shrunk_tensor{ov::element::f32, {seq_len, frame_len}};
        auto* input_data = tensor.data<float>();
        auto* output_data = shrunk_tensor.data<float>();

        // Copy first batch only
        std::memcpy(output_data, input_data, seq_len * frame_len * sizeof(float));

        shrunk_tensors.push_back(shrunk_tensor);
    }
    return shrunk_tensors;
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
        path.push_back({i - 1, j - 1});  // Store zero-indexed coordinates
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

void save_matrix_as_numpy(const std::vector<std::vector<float>>& matrix, const std::string& filename) {
    if (matrix.empty() || matrix[0].empty()) {
        return;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // NumPy header format
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t major_version = 1;
    const uint8_t minor_version = 0;

    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&major_version), 1);
    file.write(reinterpret_cast<const char*>(&minor_version), 1);

    // Create header dict string
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";

    std::string header_str = header.str();
    // Pad to make total header size (including length field) a multiple of 64 bytes
    size_t header_len = header_str.size();
    size_t total_header_size = 10 + 2 + header_len;  // 6 (magic) + 2 (version) + 2 (header_len) + header
    size_t padding = (64 - (total_header_size % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');
    header_len = header_str.size();

    // Write header length (little-endian uint16)
    uint16_t header_len_le = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<const char*>(&header_len_le), 2);

    // Write header
    file.write(header_str.c_str(), header_len);

    // Write data in row-major order (C order)
    for (const auto& row : matrix) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    file.close();
    std::cout << "Saved matrix [" << rows << ", " << cols << "] to " << filename << std::endl;
}

void save_vector_of_tensors_as_np(std::vector<ov::Tensor> tensors, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // NumPy header format
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t major_version = 1;
    const uint8_t minor_version = 0;

    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&major_version), 1);
    file.write(reinterpret_cast<const char*>(&minor_version), 1);

    // Create header dict string
    const size_t head_size = tensors.size();
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << head_size << ", ";
    const ov::Shape& first_shape = tensors[0].get_shape();
    for (size_t i = 0; i < first_shape.size(); ++i) {
        header << first_shape[i];
        if (i < first_shape.size() - 1) {
            header << ", ";
        }
    }
    header << "), }";
    std::string header_str = header.str();
    // Pad to make total header size (including length field) a multiple of 64 bytes
    size_t header_len = header_str.size();
    size_t total_header_size = 10 + 2 + header_len;  // 6 (magic) + 2 (version) + 2 (header_len) + header
    size_t padding = (64 - (total_header_size % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');
    header_len = header_str.size();

    // Write header length (little-endian uint16)
    uint16_t header_len_le = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<const char*>(&header_len_le), 2);
    // Write header
    file.write(header_str.c_str(), header_len);
    // Write data in row-major order (C order)
    for (const auto& tensor : tensors) {
        const ov::Shape& shape = tensor.get_shape();
        size_t total_size = 1;
        for (const auto& dim : shape) {
            total_size *= dim;
        }
        const float* data = tensor.data<float>();
        file.write(reinterpret_cast<const char*>(data), total_size * sizeof(float));
    }

    file.close();
    std::cout << "Saved vector of tensors to " << filename << std::endl;
}

std::vector<ov::Tensor> extract_n_frames(const std::vector<ov::Tensor>& alignment_qks, const size_t n_frames) {
    std::vector<ov::Tensor> extracted_tensors;

    for (const auto& tensor : alignment_qks) {
        const ov::Shape& shape = tensor.get_shape();
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        if (n_frames > frame_len) {
            throw std::runtime_error("Requested n_frames exceeds tensor frame length.");
        }

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

std::vector<std::pair<float, float>> to_timestamps(std::vector<std::pair<size_t, size_t>> path,
                                                   const float time_per_frame) {
    std::vector<std::pair<float, float>> timestamps;
    if (path.empty()) {
        return timestamps;
    }

    size_t current_token = path[0].first;
    size_t start_frame = path[0].second;
    size_t end_frame = path[0].second;

    for (size_t i = 1; i < path.size(); ++i) {
        const auto& [token_idx, frame_idx] = path[i];
        if (token_idx == current_token) {
            end_frame = frame_idx;
        } else {
            // Save timestamp for the completed token
            timestamps.emplace_back(start_frame * time_per_frame, (end_frame + 1) * time_per_frame);
            // Start new token
            current_token = token_idx;
            start_frame = frame_idx;
            end_frame = frame_idx;
        }
    }
    // Save timestamp for the last token
    timestamps.emplace_back(start_frame * time_per_frame, (end_frame + 1) * time_per_frame);

    return timestamps;
}

std::pair<std::vector<std::string>, std::vector<std::vector<int64_t>>> split_tokens_on_unicode(
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer) {
    const std::string decoded_full = tokenizer.decode(tokens, ov::genai::skip_special_tokens(false));
    const std::string replacement_char = u8"\uFFFD";  // todo: check replacement char correctness

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
    // std::cout << "words size: " << words.size() << std::endl;
    // std::cout << "word_tokens size: " << word_tokens.size() << std::endl;

    // std::cout << "path: ";
    // for (auto& ts : token_timestamps) {
    //     std::cout << "(" << ts.first << ", " << ts.second << ")\n";
    // }
    // std::cout << std::endl;

    // std::cout << "word, tokens [\n";
    // for (size_t i = 0; i < words.size(); ++i) {
    //     std::cout << "  \"" << words[i] << "\", [";
    //     for (const auto& token_id : word_tokens[i]) {
    //         std::cout << token_id << ", ";
    //     }
    //     std::cout << "]\n";
    // }
    // std::cout << "]" << std::endl;

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

    // std::cout << "jumps indices (" << jumps_indicies.size() << "): ";
    // for (const auto& idx : jumps_indicies) {
    //     std::cout << idx << ", ";
    // }
    // std::cout << std::endl;

    std::vector<float> jump_times;
    for (const auto& jump_idx : jumps_indicies) {
        const auto frame_idx = alignment_path[jump_idx].second;
        jump_times.push_back(frame_idx * 0.02f);  // assuming 20ms per frame
    }

    // std::cout << "jump times (" << jump_times.size() << "): ";
    // for (const auto& time : jump_times) {
    //     std::cout << time << "\n";
    // }
    // std::cout << std::endl;

    // word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    std::vector<size_t> word_boundaries;
    word_boundaries.push_back(0);
    for (size_t i = 0; i < word_tokens.size() - 1; ++i) {
        word_boundaries.push_back(word_boundaries.back() + word_tokens[i].size());
    }

    // std::cout << "word boundaries (" << word_boundaries.size() << "): ";
    // for (const auto& boundary : word_boundaries) {
    //     std::cout << boundary << ", ";
    // }
    // std::cout << std::endl;

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

    // size_t token_index = 0;
    // for (size_t i = 0; i < words.size(); ++i) {
    //     const auto& word = words[i];
    //     const auto& tokens = word_tokens[i];
    //     const size_t num_tokens = tokens.size();

    //     if (word.find("<|") != std::string::npos) {
    //         token_index += num_tokens;
    //         // std::cout << "Skipping special token word: " << word << std::endl;
    //         continue;
    //     }

    //     if (token_index + num_tokens > token_timestamps.size()) {
    //         std::cout << "Not enough timestamps for word: " << word << std::endl;
    //         break;
    //     }

    //     const float start_time = token_timestamps[token_index].first;
    //     const float end_time = token_timestamps[token_index + num_tokens - 1].second;
    //     std::cout << "Word: \"" << word << "\" | Tokens: [";
    //     for (const auto& token_id : tokens) {
    //         std::cout << token_id << ", ";
    //     }
    //     std::cout << "] | Start time: " << start_time << "s | End time: " << end_time << "s" << std::endl;
    //     token_index += num_tokens;
    // }
    // std::cout << "Token index after processing: " << token_index << std::endl;

    return word_timestamps;
};

// [heads][batch, seq_len, frame_len] -> [heads][batch, text_tokens_seq_len, frame_len]
std::vector<ov::Tensor> extract_text_tokens(const std::vector<ov::Tensor>& encoder_attention_qks,
                                            ov::genai::Tokenizer& tokenizer,
                                            const std::vector<int64_t>& tokens) {
    // text token id = token id < tokenizer.get_eos_token_id()

    std::vector<size_t> text_token_indices;
    const int64_t eot = tokenizer.get_eos_token_id();

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] <= eot) {
            text_token_indices.push_back(i);
        }
    }

    std::vector<ov::Tensor> text_token_tensors;
    for (const auto& tensor : encoder_attention_qks) {
        const ov::Shape& shape = tensor.get_shape();
        const size_t batch_size = shape[0];
        const size_t seq_len = shape[1];
        const size_t frame_len = shape[2];

        ov::Tensor text_tensor{ov::element::f32, {batch_size, text_token_indices.size(), frame_len}};
        auto* input_data = tensor.data<float>();
        auto* output_data = text_tensor.data<float>();

        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t i = 0; i < text_token_indices.size(); ++i) {
                size_t token_idx = text_token_indices[i];
                size_t input_offset = batch * seq_len * frame_len + token_idx * frame_len;
                size_t output_offset = batch * text_token_indices.size() * frame_len + i * frame_len;

                std::memcpy(output_data + output_offset, input_data + input_offset, frame_len * sizeof(float));
            }
        }

        text_token_tensors.push_back(text_tensor);
    }

    return text_token_tensors;
}

std::vector<std::pair<size_t, size_t>> find_alignment_path(
    const std::vector<ov::Tensor>& encoder_attention_qks,
    const std::vector<std::pair<size_t, size_t>>& alignment_heads,
    const size_t n_frames,
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer) {
    // for (size_t layer = 0; layer < encoder_attention_qks.size(); ++layer) {
    //     std::cout << "Layer " << layer
    //               << " accumulated encoder QK shape: " << encoder_attention_qks.at(layer).get_shape().to_string()
    //               << std::endl;
    // }

    // std::cout << "Raw tokens: ";
    // for (const auto& token : tokens) {
    //     std::cout << token << ", ";
    // }
    // std::cout << std::endl;

    const auto alignment_qks = extract_qks_alignment_heads(encoder_attention_qks, alignment_heads);

    const auto shrunk_alignment = shrink_batch_dim(encoder_attention_qks);

    // for (size_t i = 0; i < alignment_qks.size(); ++i) {
    //     const ov::Tensor& qk_tensor = alignment_qks.at(i);
    //     std::cout << "Alignment head " << i << " QK shape: " << qk_tensor.get_shape().to_string() << std::endl;
    // }

    // Extract only up to n_frames to match input feature length
    auto n_frame_alignment_qks = extract_n_frames(alignment_qks, size_t(n_frames / 2));
    save_vector_of_tensors_as_np(n_frame_alignment_qks,
                                 "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                                 "genai_attention_weights.npy");
    // const auto text_token_alignment_qks = extract_text_tokens(n_frame_alignment_qks, tokenizer, tokens);

    // for (size_t i = 0; i < text_token_alignment_qks.size(); ++i) {
    //     const ov::Tensor& qk_tensor = text_token_alignment_qks.at(i);
    //     std::cout << "Text token alignment head " << i << " QK shape: " << qk_tensor.get_shape().to_string()
    //               << std::endl;
    // }

    softmax_frame_axis(n_frame_alignment_qks);
    save_vector_of_tensors_as_np(n_frame_alignment_qks,
                                 "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                                 "genai_attention_weights_softmax.npy");

    // Apply L2 normalization along token axis (matching Python: weights / weights.norm(dim=-2, keepdim=True))
    mean_normalize_token_axis(n_frame_alignment_qks);
    save_vector_of_tensors_as_np(n_frame_alignment_qks,
                                 "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                                 "genai_attention_weights_normalized.npy");

    auto filtered_alignment_qks = median_filter_last_axis(n_frame_alignment_qks, 7);
    save_vector_of_tensors_as_np(filtered_alignment_qks,
                                 "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                                 "genai_attention_weights_median_filter.npy");

    // for (size_t i = 0; i < filtered_alignment_qks.size(); ++i) {
    //     const ov::Tensor& qk_tensor = filtered_alignment_qks.at(i);
    //     std::cout << "Filtered alignment head " << i << " QK shape: " << qk_tensor.get_shape().to_string() <<
    //     std::endl;
    // }

    // Apply softmax along frame axis (matching Python: weights.softmax(dim=-1))

    // for (size_t i = 0; i < shrunk_tensors.size(); ++i) {
    //     const ov::Tensor& qk_tensor = shrunk_tensors.at(i);
    //     std::cout << "Shrunk alignment head " << i << " QK shape: " << qk_tensor.get_shape().to_string() <<
    //     std::endl;
    // }
    const auto shrunk_tensors = shrink_batch_dim(filtered_alignment_qks);
    const auto matrix = mean_across_heads(shrunk_tensors);
    save_matrix_as_numpy(matrix,
                         "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                         "genai_matrix.npy");

    // save matrix for debugging as numpy file
    // save_matrix_as_numpy(matrix,
    //                      "/home/asuvorov/projects/openvino.genai/src/cpp/src/whisper/alignment_cost_matrix.npy");

    // std::cout << "DTW cost matrix shape: [" << matrix.size() << ", " << (matrix.empty() ? 0 : matrix[0].size()) <<
    // "]"
    //           << std::endl;

    // matix shape: [text_tokens_seq_len, frame_len]
    // need to slice text tokens -> [4:-1, :]
    auto matrix_text_tokens_slice = std::vector<std::vector<float>>{};
    if (matrix.size() >= 4) {
        matrix_text_tokens_slice = std::vector<std::vector<float>>(matrix.begin() + 3, matrix.end() - 1);
    } else {
        matrix_text_tokens_slice = matrix;
    }

    save_matrix_as_numpy(matrix_text_tokens_slice,
                         "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/"
                         "genai_sliced_matrix.npy");

    const auto alignment_path = dtw_and_backtrace(matrix_text_tokens_slice);

    // std::cout << "Alignment path (" << alignment_path.size() << "): \n[";
    // for (const auto& [token_idx, frame_idx] : alignment_path) {
    //     std::cout << "(" << token_idx << ", " << frame_idx << "),\n";
    // }
    // std::cout << "]" << std::endl;

    // const auto time_per_frame = 0.02f;  // 20 ms per frame
    // const auto timestamps = to_timestamps(alignment_path, time_per_frame);

    return alignment_path;
}

void truncate_long_words_at_sentence_boundaries(std::vector<ov::genai::WhisperWordTiming>& words) {
    // word_durations = np.array([t.end - t.start for t in alignment])
    // word_durations = word_durations[word_durations.nonzero()]
    // median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    // median_duration = min(0.7, float(median_duration))
    // max_duration = median_duration * 2

    // # hack: truncate long words at sentence boundaries.
    // # a better segmentation algorithm based on VAD should be able to replace this.
    // if len(word_durations) > 0:
    //     sentence_end_marks = ".。!！?？"
    //     # ensure words at sentence boundaries are not longer than twice the median word duration.
    //     for i in range(1, len(alignment)):
    //         if alignment[i].end - alignment[i].start > max_duration:
    //             if alignment[i].word in sentence_end_marks:
    //                 alignment[i].end = alignment[i].start + max_duration
    //             elif alignment[i - 1].word in sentence_end_marks:
    //                 alignment[i].start = alignment[i].end - max_duration

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

std::vector<ov::genai::WhisperWordTiming> merge_punctuations(std::vector<ov::genai::WhisperWordTiming>& words) {
    // prepend_punctuations: str = "\"'“¿([{-",
    // append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    // # merge prepended punctuations
    // i = len(alignment) - 2
    // j = len(alignment) - 1
    // while i >= 0:
    //     previous = alignment[i]
    //     following = alignment[j]
    //     if previous.word.startswith(" ") and previous.word.strip() in prepended:
    //         # prepend it to the following word
    //         following.word = previous.word + following.word
    //         following.tokens = previous.tokens + following.tokens
    //         previous.word = ""
    //         previous.tokens = []
    //     else:
    //         j = i
    //     i -= 1

    // # merge appended punctuations
    // i = 0
    // j = 1
    // while j < len(alignment):
    //     previous = alignment[i]
    //     following = alignment[j]
    //     if not previous.word.endswith(" ") and following.word in appended:
    //         # append it to the previous word
    //         previous.word = previous.word + following.word
    //         previous.tokens = previous.tokens + following.tokens
    //         following.word = ""
    //         following.tokens = []
    //     else:
    //         i = j
    //     j += 1
    const std::string prepend_punctuations = "\"'“¿([{-";
    const std::string append_punctuations = "\"'.。,，!！?？:：”)]}、";

    // merge prepended punctuations
    size_t i = words.size() - 2;
    size_t j = words.size() - 1;
    while (i < words.size()) {
        auto& previous = words[i];
        auto& following = words[j];
        if (!previous.word.empty() && previous.word[0] == ' ' &&
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
        if (!previous.word.empty() && previous.word.back() != ' ' &&
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
        const bool with_space = !subword.empty() && std::isspace(subword[0]);

        // punctuation = subword.strip() in string.punctuation
        // python is_punct: r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        // const bool is_punctuation = !subword.empty() && std::ispunct(subword[0]);
        const std::string trimmed_subword = trim(subword);
        const bool is_punctuation =
            !trimmed_subword.empty() && trimmed_subword.size() == 1 && std::ispunct(trimmed_subword[0]);

        if (words.empty() || is_special || with_space || is_punctuation) {
            words.push_back(subword);
            word_tokens.push_back(subword_tokens);
        } else {
            words.back() += subword;
            word_tokens.back().insert(word_tokens.back().end(), subword_tokens.begin(), subword_tokens.end());
        }
    }

    // std::cout << "Words:\n";
    // for (size_t i = 0; i < words.size(); ++i) {
    //     std::cout << "  \"" << words[i] << "\", Tokens: [";
    //     for (const auto& token_id : word_tokens[i]) {
    //         std::cout << token_id << ", ";
    //     }
    //     std::cout << "]\n";
    // }
    // std::cout << std::endl;

    return {words, word_tokens};
}

std::vector<ov::genai::WhisperWordTiming> get_word_level_timestamps(
    const std::vector<ov::Tensor>& encoder_attention_qks,
    const size_t n_frames,
    const std::vector<int64_t>& tokens,
    ov::genai::Tokenizer& tokenizer,
    const ov::genai::WhisperGenerationConfig& generation_config,
    const float chunk_time_offset) {
    auto tokens_copy = tokens;  // to avoid modifying input tokens
    // tokens_copy.push_back(tokenizer.get_eos_token_id());

    if (generation_config.save_attention_weights) {
        save_vector_of_tensors_as_np(
            encoder_attention_qks,
            "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/current/"
            "encoder_attention_qks.npy");
        // save_vector_of_tensors_as_np(
        //     encoder_attention_qks,
        //     "/home/asuvorov/projects/openvino.genai/.vscode/tasks/word_level_timestamps/data/reference/"
        //     "encoder_attention_qks.npy");
    }

    const auto alignment_path =
        find_alignment_path(encoder_attention_qks, generation_config.alignment_heads, n_frames, tokens_copy, tokenizer);

    std::vector<int64_t> text_tokens;
    const int64_t eot = tokenizer.get_eos_token_id();
    for (const auto& token : tokens_copy) {
        if (token <= eot) {
            text_tokens.push_back(token);
        }
    }

    // std::cout << "text_tokens (" << text_tokens.size() << "):\n[";
    // for (const auto& token : text_tokens) {
    //     std::cout << token << ", ";
    // }
    // std::cout << "]" << std::endl;

    const auto [words, word_tokens] = split_tokens_on_spaces(text_tokens, tokenizer);

    // for (size_t i = 0; i < words.size(); ++i) {
    //     std::cout << "Word " << i << ": \"" << words[i] << "\", Tokens: [";
    //     for (const auto& token_id : word_tokens[i]) {
    //         std::cout << token_id << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    // }

    auto words_timestamps = match_words_to_alignment_path(words, word_tokens, alignment_path, chunk_time_offset);

    // for (auto& word_timing : words_timestamps) {
    //     std::cout << word_timing.word << " " << word_timing.start_ts << " - " << word_timing.end_ts << "s" <<
    //     std::endl;
    // }
    // std::cout << std::endl;

    truncate_long_words_at_sentence_boundaries(words_timestamps);

    auto merged_timestamps = merge_punctuations(words_timestamps);

    // the "hack" part of OpenAI code is missing: https://github.com/openai/whisper/blob/main/whisper/timing.py#L346
    // It is intended to adjust word timings at sentence boundaries based on median word duration.
    // # GenAI:     That's     0.00 - 1.04 | OpenAI:  That's     0.60 - 1.04
    // # GenAI:     funny,     1.04 - 1.34 | OpenAI:  funny,     1.04 - 1.34
    // # GenAI:     remarked   1.72 - 1.96 | OpenAI:  remarked   1.72 - 1.96
    // # GenAI:     a          1.96 - 2.04 | OpenAI:  a          1.96 - 2.04
    // # GenAI:     bit,       2.04 - 2.20 | OpenAI:  bit,       2.04 - 2.20
    // # GenAI:     see        2.38 - 2.38 | OpenAI:  see        2.38 - 2.38
    // # GenAI:     you        2.38 - 2.50 | OpenAI:  you        2.38 - 2.50
    // # GenAI:     thought    2.50 - 2.64 | OpenAI:  thought    2.50 - 2.64
    // # GenAI:     funny.     2.64 - 2.86 | OpenAI:  funny.     2.64 - 2.86

    return merged_timestamps;
};

}  // namespace

namespace ov::genai {

std::vector<ov::genai::WhisperWordTiming> add_word_level_timestamps(const std::vector<int64_t>& sot_tokens,
                                                                    const std::vector<int64_t>& text_tokens,
                                                                    ov::genai::Tokenizer& tokenizer,
                                                                    std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                                    const ov::Tensor& hidden_state_tensor,
                                                                    const ov::genai::WhisperGenerationConfig& config,
                                                                    const size_t n_frames,
                                                                    const float chunk_time_offset
                                                                    // todo: add raw_perf_metrics
) {
    const size_t batch_size = 1;

    std::vector<int64_t> tokens = sot_tokens;
    tokens.push_back(config.no_timestamps_token_id);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(config.eos_token_id);

    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);

    const ov::Tensor input_ids_tensor{ov::element::i64, {1, tokens.size()}, const_cast<int64_t*>(tokens.data())};

    decoder->start_async(hidden_state_tensor, input_ids_tensor, beam_idx);
    decoder->wait();

    const auto& accumulated_qks = decoder->get_encoder_qks();
    decoder->reset_state();

    auto word_timestamps =
        get_word_level_timestamps(accumulated_qks, n_frames, tokens, tokenizer, config, chunk_time_offset);

    return word_timestamps;
}
}  // namespace ov::genai
