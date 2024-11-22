// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>
#include <string_view>

namespace {

// only batch_size = 1 currently supported
constexpr size_t BATCH_SIZE = 1;

size_t get_seq_len_axis(std::shared_ptr<ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // therefore usually seq_length_axis = 2
    size_t seq_length_axis = 2;

    // "ReadValue" node is KV cache representation in stateful model
    std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

    for (const auto op : model->get_ops()) {
        if (op->get_type_name() != kv_node_type_name) {
            continue;
        }

        // Shape example: [-1,4,0,64]
        auto shape = op->get_input_partial_shape(0);

        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            // Find axis = 0. This would be sequence length axis.
            if (shape[i] == 0) {
                seq_length_axis = i;
            }
        }
        break;
    }

    return seq_length_axis;
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back() && text.size() > print_len) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        constexpr char replacement[] = "\xef\xbf\xbd";  // MSVC with /utf-8 fails to compile � directly with newline in string literal error.
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0) {
            // Don't print incomplete text
            return;
        } else if (text.size() > print_len) {
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text length is increaeseds.
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
            print_len = text.size();
        }
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        if (text.size() <= print_len)
            return;
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // Copy elements from the old to a new tensor and return it.
    // Trim kv tensor on sequence length axis
    // key/values tensor shape example: [BATCH_SIZE, num_kv_heads, seq_len, head_size]
    // Sequence length axis position may vary from one model to another

    auto shape = tensor.get_shape();

    OPENVINO_ASSERT(seq_len_axis < shape.size(),
                    "Sequence length axis: ",
                    seq_len_axis,
                    " should be less than shape size: ",
                    shape.size());

    size_t old_seq_len = shape[seq_len_axis];

    OPENVINO_ASSERT(new_seq_len <= old_seq_len);

    // if new_seq_len equal to old one no need to copy tensor, return as is
    if (old_seq_len == new_seq_len)
        return tensor;

    shape[seq_len_axis] = new_seq_len;

    if (seq_len_axis == 0) {
        tensor.set_shape(shape);
        return tensor;
    }

    ov::Coordinate new_shape_begin{0, 0, 0, 0};
    ov::Coordinate new_shape_end{shape};

    auto new_tensor = ov::Tensor(tensor, new_shape_begin, new_shape_end);

    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // trim kv_cache values up to the new_seq_len
    auto states = request.query_state();
    ov::parallel_for(states.size(), [&](size_t i) {
        ov::Tensor old_tensor = states.at(i).get_state();
        states.at(i).set_state(trimm_tensor(old_tensor, seq_len_axis, new_seq_len));
    });
}

class PromptLookupCandidateGenerator {
private:
    const size_t max_ngram_size = 3;
    size_t num_pred_tokens = 5;
    const size_t max_pred_tokens = 20;

public:
    PromptLookupCandidateGenerator(const size_t max_ngram_size, const size_t num_pred_tokens)
        : max_ngram_size{max_ngram_size},
          num_pred_tokens{num_pred_tokens} {};

    std::vector<int64_t> generate_candidates(const std::vector<int64_t>& input_ids) {
        const size_t input_length = input_ids.size();

        for (int32_t ngram_size = max_ngram_size; ngram_size > 0; ngram_size--) {
            // extract last ngram_size tokens as search ngram
            std::vector<int64_t> ngram = std::vector<int64_t>{input_ids.cend() - ngram_size, input_ids.cend()};

            // find ngram match in input_ids
            size_t ngram_i = 0;
            for (size_t input_i = 0; input_i < input_length - ngram_size; input_i++) {
                if (ngram[ngram_i] != input_ids[input_i]) {
                    ngram_i = 0;
                    continue;
                }

                ngram_i++;

                if (ngram_i < ngram_size) {
                    continue;
                }

                // match found with the end at input_i
                size_t avaliable_num_pred = std::min(input_length - (input_i + 1), num_pred_tokens);

                // return candidates with length of avaliable_num_pred
                return std::vector<int64_t>{input_ids.cbegin() + input_i + 1,
                                            input_ids.cbegin() + input_i + 1 + avaliable_num_pred};
            }
        }

        return std::vector<int64_t>{};
    }

    void update_candidate_strategy(const size_t num_matches) {
        // dynamically adjust number of generated candidates based on number of matches
        // we want to balance the benefits of getting assistant tokens correct with the
        // cost of forecasting incorrect assistant tokens.
        if (num_matches == num_pred_tokens) {
            num_pred_tokens = std::min(num_pred_tokens + 2, max_pred_tokens);
        } else {
            num_pred_tokens = std::max(num_pred_tokens - 1, size_t(1));
        }
    }
};

int64_t get_eos_token(const std::shared_ptr<ov::Model> tokenizer) {
    auto rt_info = tokenizer->get_rt_info();  // Get the runtime info for the model

    auto it = rt_info.find("eos_token_id");
    if (it == rt_info.end()) {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }
    return it->second.as<int64_t>();
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    // tokenizer model
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

    const std::string model_dir = std::string{argv[1]};

    auto tokenizer_model = core.read_model(model_dir + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);

    std::vector<int64_t> full_input_ids{input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_size()};

    ov::InferRequest detokenizer =
        core.compile_model(model_dir + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};

    std::shared_ptr<ov::Model> ov_model = core.read_model(model_dir + "/openvino_model.xml");

    size_t seq_len_axis = get_seq_len_axis(ov_model);

    ov::InferRequest model = core.compile_model(ov_model, "CPU").create_infer_request();

    model.set_tensor("input_ids", input_ids);
    model.set_tensor("attention_mask", attention_mask);

    ov::Tensor position_ids = model.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    size_t seq_len = input_ids.get_shape()[1];

    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    model.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    // To collect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    model.infer();

    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto logits = model.get_tensor("logits");
    size_t vocab_size = logits.get_shape().back();
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;

    full_input_ids.push_back(out_token);

    auto first_token = out_token;
    text_streamer.put(out_token);

    const int64_t EOS_TOKEN = get_eos_token(tokenizer_model);

    // Prompt lookup decoding is a speculative decoding technique where the draft model replaced
    // with string matching in the prompt to generate candidate token sequences.
    int max_sequence_length = 100;
    PromptLookupCandidateGenerator candidateGenerator{3, 5};

    while (out_token != EOS_TOKEN && seq_len < max_sequence_length) {
        auto candidates = candidateGenerator.generate_candidates(full_input_ids);

        // cut redundant candidates on last iteration
        size_t tokens_to_generate = max_sequence_length - seq_len;
        candidates.resize(std::min(candidates.size(), tokens_to_generate - 1));
        size_t candidates_size = candidates.size();

        // candidates_size + 1 tokens will be fed at once in a single infer request.
        input_ids.set_shape({BATCH_SIZE, candidates_size + 1});
        input_ids.data<int64_t>()[0] = first_token;
        std::copy_n(candidates.begin(), candidates_size, input_ids.data<int64_t>() + 1);

        attention_mask.set_shape({BATCH_SIZE, seq_len + candidates_size + 1});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        position_ids.set_shape({BATCH_SIZE, candidates_size + 1});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        model.infer();

        data_logits = logits.data<float>();  // [BATCH_SIZE, 1 + candidates_size, vocab_size]

        // 1. accept current out token (if not eos)
        // 2. check if it matches appropriate candidate
        //      2.1 if it's match, continue - accept next token
        //      2.2 it it's mismatch, stop iteration but still accept current token as it was last token generated by
        //      model from a valid sequence.
        size_t accepted_tokens_number = 0;
        for (size_t i = 0; i < candidates_size + 1; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            out_token = std::max_element(start, stop) - start;

            if (out_token == EOS_TOKEN) {
                break;
            }

            text_streamer.put(out_token);
            full_input_ids.push_back(out_token);
            accepted_tokens_number++;

            if (i == candidates_size || out_token != candidates[i]) {
                break;
            }
        }

        if (accepted_tokens_number > 0) {
            candidateGenerator.update_candidate_strategy(accepted_tokens_number - 1);
        }

        // After the inference request, key/values have shape [BATCH_SIZE, seq_len + candidates_size, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += accepted_tokens_number;
        update_kv_cache(model, seq_len_axis, seq_len);

        first_token = out_token;
    }

    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    model.reset_state();
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
