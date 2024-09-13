// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <openvino/core/parallel.hpp>
#include <openvino/openvino.hpp>
#include <random>

namespace {

constexpr size_t BATCH_SIZE = 1;

size_t get_seq_len_axis(std::shared_ptr<ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // threfore usually seq_length_axis = 2
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
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        } else if (text.size() > print_len) {
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text lengh is increaesed.
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
    // Sequense length axis position may vary from one model to another

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

class AssistedCandidateGenerator {
private:
    ov::InferRequest draft_model;
    size_t max_seq_length;
    size_t num_pred_tokens = 5;
    size_t seq_len_axis;
    const size_t max_pred_tokens = 10;
    int64_t out_of_kv_cache_token = -1;
    size_t draft_model_seq_length = 0;

public:
    AssistedCandidateGenerator(ov::InferRequest draft_model,
                               const size_t max_seq_length,
                               const size_t num_pred_tokens,
                               const size_t seq_len_axis)
        : draft_model{draft_model},
          max_seq_length{max_seq_length},
          num_pred_tokens{num_pred_tokens},
          seq_len_axis{seq_len_axis} {};

    int64_t generate_next_token(const std::vector<int64_t> tokens) {
        size_t tokens_size = tokens.size();
        auto input_ids = draft_model.get_tensor("input_ids");
        input_ids.set_shape({BATCH_SIZE, tokens_size});
        std::copy_n(tokens.begin(), tokens_size, input_ids.data<int64_t>());

        auto attention_mask = draft_model.get_tensor("attention_mask");
        attention_mask.set_shape({BATCH_SIZE, draft_model_seq_length + tokens_size});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        auto position_ids = draft_model.get_tensor("position_ids");
        position_ids.set_shape({BATCH_SIZE, tokens_size});
        std::iota(position_ids.data<int64_t>(),
                  position_ids.data<int64_t>() + position_ids.get_size(),
                  draft_model_seq_length);

        draft_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        draft_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        draft_model.infer();

        auto logits = draft_model.get_tensor("logits");
        size_t vocab_size = logits.get_shape().back();
        auto sequence_logits = logits.data<float>() + (tokens_size - 1) * vocab_size;

        draft_model_seq_length += tokens_size;

        return std::max_element(sequence_logits, sequence_logits + vocab_size) - sequence_logits;
    }

    std::vector<int64_t> generate_candidates(int64_t out_token) {
        std::vector<int64_t> candidates;

        // limit candidates size by num_pred_tokens or by max_seq_length
        size_t candidates_to_generate = std::min(num_pred_tokens, max_seq_length - draft_model_seq_length - 1);

        candidates.reserve(candidates_to_generate);

        // generate cadidates
        for (size_t i = 0; i < candidates_to_generate; i++) {
            // if out_of_kv_cache_token is present, prepend it to out_token in order to collect kv cache for it
            if (out_of_kv_cache_token != -1) {
                out_token = generate_next_token(std::vector{out_of_kv_cache_token, out_token});
                out_of_kv_cache_token = -1;
            } else {
                out_token = generate_next_token(std::vector{out_token});
            }

            candidates.push_back(out_token);
        }

        out_of_kv_cache_token = candidates.back();
        return candidates;
    }

    void update_candidate_strategy(const size_t num_matches) {
        // dynamically adjust number of generated candidates based on number of matches
        // we want to balance the benefits of getting candidates tokens correct with the
        // cost of forecasting incorrect candidates tokens.
        if (num_matches == num_pred_tokens) {
            num_pred_tokens = std::min(num_pred_tokens + 2, max_pred_tokens);
        } else {
            num_pred_tokens = std::max(int64_t(num_pred_tokens) - 1, int64_t(1));
        }
    }

    void update_kv_cache(const size_t seq_length) {
        // this is the case when main model accepted all candidates from draft model
        // we need to collect kv cache for out_of_kv_cache_token by infering it
        // on next candidates generation cycle out_of_kv_cache_token will be prefixed
        // to main models's latest out token
        if (draft_model_seq_length < seq_length) {
            return;
        }

        out_of_kv_cache_token = -1;
        ::update_kv_cache(draft_model, seq_len_axis, seq_length);
        draft_model_seq_length = seq_length;
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
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <DRAFT MODEL_DIR> <MAIN MODEL_DIR> '<PROMPT>'");
    }

    // tokenizer model
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[3]);
    ov::InferRequest detokenizer =
        core.compile_model(std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};

    // draft model (which is smaller, less accurate but faster)
    std::shared_ptr<ov::Model> ov_draft_model = core.read_model(std::string{argv[1]} + "/openvino_model.xml");

    size_t draft_model_seq_len_axis = get_seq_len_axis(ov_draft_model);

    ov::InferRequest draft_model = core.compile_model(ov_draft_model, "CPU").create_infer_request();

    size_t seq_len = input_ids.get_shape()[1];

    // main model (which is bigger, more accurate but slower)
    std::shared_ptr<ov::Model> ov_main_model = core.read_model(std::string{argv[2]} + "/openvino_model.xml");

    size_t main_model_seq_len_axis = get_seq_len_axis(ov_main_model);

    ov::InferRequest main_model = core.compile_model(ov_main_model, "CPU").create_infer_request();

    size_t max_sequence_length = 100;

    AssistedCandidateGenerator candidateGenerator{draft_model, max_sequence_length, 5, draft_model_seq_len_axis};

    main_model.set_tensor("input_ids", input_ids);
    main_model.set_tensor("attention_mask", attention_mask);

    auto position_ids = main_model.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);

    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    main_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    main_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    // To coollect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    candidateGenerator.generate_next_token(
        std::vector<int64_t>(input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_size()));

    main_model.infer();

    size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    OPENVINO_ASSERT(vocab_size == main_model.get_tensor("logits").get_shape().back(),
                    "vocab size should be the same for the both models");

    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto logits = main_model.get_tensor("logits");
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;

    text_streamer.put(out_token);

    const int64_t EOS_TOKEN = get_eos_token(tokenizer_model);

    /* Speculative decoding works the following way. The draft model predicts the next K
       tokens one by one in an autoregressive manner, while the main model validates these
       predictions and corrects them if necessary. We go through each predicted token, and
       if a difference is detected between the draft and main model, we stop and keep the
       last token predicted by the main model. Then the draft model gets the latest main
       prediction and again tries to predict the next K tokens, repeating the cycle.

       This approach reduces the need for multiple infer requests to the main model,
       enhancing performance. For instance, in more predictable parts of text generation,
       the draft model can, in best-case scenarios, generate the next K tokens that exactly
       match the target. In that case they are validated in a single inference call to
       the main model instead of running K subsequent requests.
       */

    while (out_token != EOS_TOKEN && seq_len < max_sequence_length) {
        // generate candidates from the draft model
        std::vector<int64_t> candidates = candidateGenerator.generate_candidates(out_token);
        size_t candidates_size = candidates.size();

        // For the main network, candidates_size + 1 tokens will be fed at once in a single infer request.
        input_ids.set_shape({BATCH_SIZE, candidates_size + 1});

        input_ids.data<int64_t>()[0] = out_token;
        if (candidates_size > 0) {
            std::copy_n(candidates.begin(), candidates_size, input_ids.data<int64_t>() + 1);
        }

        attention_mask.set_shape({BATCH_SIZE, seq_len + candidates_size + 1});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        position_ids.set_shape({BATCH_SIZE, candidates_size + 1});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        main_model.infer();

        data_logits = logits.data<float>();  // [BATCH_SIZE, K, vocab_size]

        // match model tokens with candidate tokens
        // 1. accept current out token (if not eos)
        // 2. check if it matches apropriate candidate
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
            accepted_tokens_number++;

            if (i == candidates_size || out_token != candidates[i]) {
                break;
            }
        }

        // After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += accepted_tokens_number;

        if (accepted_tokens_number > 0) {
            candidateGenerator.update_candidate_strategy(accepted_tokens_number - 1);
        }

        candidateGenerator.update_kv_cache(seq_len);
        update_kv_cache(main_model, main_model_seq_len_axis, seq_len);

        candidates.clear();
    }
    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    draft_model.reset_state();
    main_model.reset_state();
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
