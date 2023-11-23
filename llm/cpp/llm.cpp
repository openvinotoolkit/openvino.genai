// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>
#include <valarray>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int64_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int64_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}

// Modifyed Knuth–Morris–Pratt algorithm which returns a set of tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, std::vector<int64_t> needle) {  // TODO: pass iters to haystack to avoid searchng last ngram symbols
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle[pos] == needle[cnd]) {
            partial_match_table[pos] = partial_match_table[cnd];
        } else {
            partial_match_table[pos] = cnd;
            while (cnd >= 0 && needle[pos] != needle[cnd]) {
                cnd = partial_match_table[cnd];
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t j = 0;  // The position of the current character in haystack
    int k = 0;  // The position of the current character in needle
    while (j < haystack.size() - 1) {
        if (needle[k] == haystack[j]) {
            ++j;
            ++k;
            if (k == int(needle.size())) {
                res.push_back(haystack[j]);
                k = partial_match_table[k];
            }
        } else {
            k = partial_match_table[k];
            if (k < 0) {
                ++j;
                ++k;
            }
        }
    }
    return res;
}
constexpr size_t GROUP_SIZE = 1;
enum class StopCriteria {early, heuristic, never};
StopCriteria stop_criteria = StopCriteria::never;
size_t MAX_NEW_TOKENS = 10;
constexpr float LENGTH_PENALTY = 1.0;  // TODO: align defaults with transformers
constexpr int64_t EOS_TOKEN = 1;  // There's no way to extract the value from the tokenizer for now  // TODO: 2 for llama2
constexpr size_t N_GROUPS = 3;
constexpr float DIVERSITY_PENALTY = 9e9f;
constexpr size_t NO_REPEAT_NGRAM_SIZE = 3;
}

    struct Beam {
        float log_prob;
        std::vector<int64_t> tokens;
        ov::InferRequest ireq;  // TODO: move to sep struct
        bool operator<(const Beam& other) {
            return log_prob > other.log_prob;  // greater, not less to build min heap
        }
        Beam(float log_prob, const std::vector<int64_t>& tokens, const ov::InferRequest& ireq) : log_prob{log_prob}, tokens{tokens}, ireq{ireq} {}
        Beam& operator=(Beam&& other) {
            log_prob = other.log_prob;
            tokens = std::move(other.tokens);
            ireq = other.ireq;
            return *this;
        }
        Beam& operator=(const Beam& other) {
            log_prob = other.log_prob;
            tokens = other.tokens;
            ireq = other.ireq;
            return *this;
        }
        Beam(Beam&& other) : log_prob{other.log_prob}, tokens{std::move(other.tokens)}, ireq{other.ireq} {};
        Beam(const Beam& other) : log_prob{other.log_prob}, tokens{other.tokens}, ireq{other.ireq} {};
    };

std::ostream& operator<<(std::ostream& os, const Beam& beam) {
    os << std::setprecision(6) << beam.log_prob << ": ";
    for (size_t token : beam.tokens) {
        os << token << ' ';
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Beam>& beams) {
    for (const Beam& beam : beams) {
        os << beam << '\n';
    }
    return os;
}


struct Hypotheses {
        std::vector<Beam> beams;
        void push(Beam&& beam) {
            beam.log_prob /= std::pow(beam.tokens.size(), LENGTH_PENALTY);
            beams.push_back(std::move(beam));
            std::push_heap(beams.begin(), beams.end());
            if (beams.size() > GROUP_SIZE) {
                std::pop_heap(beams.begin(), beams.end());
                beams.pop_back();
            }
        }
        bool is_done(float best_sum_logprobs, size_t cur_len) const {   // TODO: just done()?
            if (beams.size() < GROUP_SIZE) {
                return false;
            }
            switch (stop_criteria) {
                case StopCriteria::early: return true;
                case StopCriteria::heuristic: {
                    float worst_score = beams.front().log_prob;
                    float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), LENGTH_PENALTY);
                    return worst_score >= highest_attainable_score;
                }
                case StopCriteria::never: {
                    float worst_score = beams.front().log_prob;
                    float highest_attainable_score = LENGTH_PENALTY > 0.0f ? best_sum_logprobs / std::pow(float(MAX_NEW_TOKENS), LENGTH_PENALTY) : best_sum_logprobs / std::pow(float(cur_len), LENGTH_PENALTY);
                    return worst_score >= highest_attainable_score;
                }
                default: throw std::runtime_error("Never reached");
            }
        }
    };
        struct Group {
        std::vector<Beam> beams;  // TODO: one contigous array with all beams?
        Hypotheses hypotheses;
    };

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    ov::Core core;
    // core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    // auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    // ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    ov::Tensor input_ids{ov::element::i64, {1, 3}};
    input_ids.data<int64_t>()[0] = 1;
    input_ids.data<int64_t>()[1] = 408;
    input_ids.data<int64_t>()[2] = 2176;
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::CompiledModel compiled = core.compile_model(model, "CPU");  // , ov::cache_dir("llm-cache"));

    struct Token {float log; int64_t idx;
        bool operator<(Token indexed) {
            return log > indexed.log;  // greater, not less to pick most probable tokens
        }
    };
    std::vector<Group> groups{N_GROUPS};
    {
        ov::InferRequest ireq = compiled.create_infer_request();
        for (size_t idx = 2; idx < inputs.size(); ++idx) {
            ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
        }
        ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
        ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("input_ids").get_size()});
        std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
        std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
        ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
        std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
        ireq.infer();

        ov::Tensor logits_tensor = ireq.get_tensor("logits");
        size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
        std::valarray<float> logits{logits_tensor.data<const float>() + (input_ids.get_size() - 1) * vocab_size, vocab_size};  // TODO: maybe use valarray<Token>
        float max_logit = logits.max();
        float log_sum = std::log((std::exp(logits - max_logit)).sum());  // TODO: log(softmax) only for topk logits
        std::valarray<float> log_prob = logits - max_logit - log_sum;

        std::vector<Token> topk;
        topk.reserve(log_prob.size());
        for (size_t idx = 0; idx < log_prob.size(); ++idx) {
            topk.push_back({log_prob[idx], int64_t(idx)});
        }
        for (size_t group_idx = 0; group_idx < N_GROUPS; ++group_idx) {
            std::partial_sort(topk.begin(), topk.begin() + GROUP_SIZE, topk.end());
            for (size_t idx = 0; idx < GROUP_SIZE; ++idx) {
                groups[group_idx].beams.push_back(Beam{topk[idx].log, {topk[idx].idx}, compiled.create_infer_request()});
                topk[idx].log -= DIVERSITY_PENALTY;
                ov::InferRequest& beam_ireq = groups[group_idx].beams.back().ireq;
                for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
                    beam_ireq.set_input_tensor(tensor_idx, ireq.get_output_tensor(tensor_idx - 2));
                }
                beam_ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
                beam_ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_size() + 1});
                std::fill_n(beam_ireq.get_tensor("attention_mask").data<int64_t>(), beam_ireq.get_tensor("attention_mask").get_size(), 1);
                beam_ireq.get_tensor("input_ids").data<int64_t>()[0] = topk[idx].idx;  // TODO: don't allow EOS as first token?
                beam_ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
                beam_ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
                beam_ireq.infer();
            }
        }
    }
    for (size_t length_count = 0; length_count < MAX_NEW_TOKENS - 1; ++length_count) {
        for (size_t group_idx = 0; group_idx < N_GROUPS; ++group_idx) {
            std::vector<Beam> candidates;
            candidates.reserve(2 * GROUP_SIZE);
            for (size_t beam_idx = 0; beam_idx < GROUP_SIZE; ++beam_idx) {
                ov::InferRequest& beam_ireq = groups[group_idx].beams[beam_idx].ireq;
                // beam_ireq.wait();  TODO: async
                ov::Tensor logits_tensor = beam_ireq.get_tensor("logits");
                std::valarray<float> logits{logits_tensor.data<const float>(), logits_tensor.get_size()};  // TODO: maybe use valarray<Token>
                float max_logit = logits.max();
                float log_sum = std::log((std::exp(logits - max_logit)).sum());  // TODO: log(softmax) only for topk logits
                std::valarray<float> log_prob = logits - max_logit - log_sum;
                std::vector<Token> tokens;
                tokens.reserve(log_prob.size());
                for (size_t idx = 0; idx < log_prob.size(); ++idx) {
                    tokens.push_back({log_prob[idx], int64_t(idx)});
                }
                for (size_t prev_group_idx = 0; prev_group_idx < group_idx; ++prev_group_idx) {  // TODO: range based for
                    for (size_t prev_beam_idx = 0; prev_beam_idx < GROUP_SIZE; ++prev_beam_idx) {
                        tokens[groups[prev_group_idx].beams[prev_beam_idx].tokens.back()].log -= DIVERSITY_PENALTY;
                    }
                }
                std::vector<int64_t>& other_tokens = groups[group_idx].beams[beam_idx].tokens;
                if (other_tokens.size() >= NO_REPEAT_NGRAM_SIZE) {
                    for (int64_t ban_id : kmp_search(other_tokens, {other_tokens.end() - NO_REPEAT_NGRAM_SIZE + 1, other_tokens.end()})) {
                        tokens[ban_id].log = -std::numeric_limits<float>::infinity();
                    }
                }
                // Sample 2 * GROUP_SIZE next tokens to get at least 1 non EOS token per beam
                std::nth_element(tokens.begin(), tokens.begin() + 2 * GROUP_SIZE, tokens.end());
                for (size_t idx = 0; idx < 2 * GROUP_SIZE; ++idx) {
                    candidates.push_back(groups[group_idx].beams[beam_idx]);
                    candidates.back().log_prob += tokens[idx].log;
                    candidates.back().tokens.push_back(tokens[idx].idx);
                }
            }
            std::sort(candidates.begin(), candidates.end());
            size_t cur_len = groups[group_idx].beams.front().tokens.size() + 1;
            groups[group_idx].beams.clear();
            for (size_t cand_id = 0; cand_id < candidates.size(); ++cand_id) {
                if (EOS_TOKEN == candidates[cand_id].tokens.back()) {  // TODO: idx->token_id
                    // if beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_id >= GROUP_SIZE) {
                        continue;
                    }
                    groups[group_idx].hypotheses.push(std::move(candidates[cand_id]));
                } else {
                    groups[group_idx].beams.push_back(std::move(candidates[cand_id]));
                    size_t cur_beam = groups[group_idx].beams.size() - 1;  // TODO: beter loop iteration
                    auto ireq = compiled.create_infer_request();
                    for (size_t tensor_id = 3; tensor_id < inputs.size(); ++tensor_id) {
                        ireq.set_input_tensor(tensor_id, groups[group_idx].beams.back().ireq.get_output_tensor(tensor_id - 2));
                    }
                    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
                    ireq.get_tensor("input_ids").data<int64_t>()[0] = groups[group_idx].beams.back().tokens.back();
                    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_size() + 1});
                    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
                    ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
                    ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
                    groups[group_idx].beams.back().ireq = ireq;
                    if (groups[group_idx].beams.size() == GROUP_SIZE) {
                        break;
                    }
                }
            }
            if (std::all_of(groups.begin(), groups.end(), [cur_len](const Group& gr){return gr.hypotheses.is_done(cur_len, gr.beams.front().log_prob);})) {  // TODO: that requires groups[group_idx].beams to be not empty
                break;
            }

            for (Beam& beam : groups[group_idx].beams) {
                beam.ireq.infer();
            }
        }
    }
    // finalize
    for (Group& group : groups) {
        std::cout << "\nGroup:";
        if (group.hypotheses.is_done(group.beams.front().tokens.size(), group.beams.front().log_prob)) {
            continue;
        }
        for (Beam& beam: group.beams) {  // TODO: &&
            group.hypotheses.push(std::move(beam));
        }
        std::sort_heap(group.hypotheses.beams.begin(), group.hypotheses.beams.end());
        for (const Beam& beam: group.hypotheses.beams) {
            std::cout << "\nscore: " << beam.log_prob << " prediction: ";  // TODO: alight with transformers
            for (int64_t token : beam.tokens) {
                // print_token(detokenizer, token);
                std::cout << token << ", ";
            }
        }
    }


    // while (out_token != EOS_TOKEN) {
    //     for (size_t idx = 2; idx < inputs.size(); ++idx) {
    //          ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 1));
    //     }
    //     ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
    //     ireq.start_async();
    //     print_token(detokenizer, out_token);
    //     ireq.wait();
    //     logits = ireq.get_tensor("logits").data<float>();
    //     out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
    // }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
