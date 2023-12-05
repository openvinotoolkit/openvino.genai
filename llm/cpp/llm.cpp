// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

ov::Tensor detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, tokens.size()});
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        inp.data<int64_t>()[idx] = tokens[idx];
    }
    detokenizer.infer();
    return detokenizer.get_output_tensor();
}

// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, std::vector<int64_t> needle) {
    if (needle.empty()) {  // NO_REPEAT_NGRAM_SIZE == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle[pos] == needle[size_t(cnd)]) {
            partial_match_table[pos] = partial_match_table[size_t(cnd)];
        } else {
            partial_match_table[pos] = cnd;
            while (cnd >= 0 && needle[pos] != needle[size_t(cnd)]) {
                cnd = partial_match_table[size_t(cnd)];
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t j = 0;  // The position of the current character in haystack
    int k = 0;  // The position of the current character in needle
    while (j < haystack.size() - 1) {
        if (needle[size_t(k)] == haystack[j]) {
            ++j;
            ++k;
            if (k == int(needle.size())) {
                res.push_back(haystack[j]);
                k = partial_match_table[size_t(k)];
            }
        } else {
            k = partial_match_table[size_t(k)];
            if (k < 0) {
                ++j;
                ++k;
            }
        }
    }
    return res;
}
enum class StopCriteria {early, heuristic, never};
size_t MAX_NEW_TOKENS;
size_t N_GROUPS;
size_t GROUP_SIZE;
StopCriteria stop_criteria;
size_t NO_REPEAT_NGRAM_SIZE;
float DIVERSITY_PENALTY;
double LENGTH_PENALTY;  // TODO: align defaults with transformers
int64_t EOS_TOKEN;  // There's no way to extract the value from the tokenizer for now
}

struct Beam {
    float score = -std::numeric_limits<float>::infinity();  // The bigger, the better
    std::vector<int64_t> tokens;
    size_t batch_idx = 0;
    size_t global_beam_idx = 0;
};

bool greater(const Beam& left, const Beam& right) {
    return left.score > right.score;
}

struct Group {
    std::vector<Beam> ongoing;
    std::vector<Beam> min_heap;  // The smallest beam is the first
    bool done = false;
    void finish(Beam&& beam, size_t prompt_light) {
        beam.score = float(double(beam.score) / std::pow(beam.tokens.size() + prompt_light, LENGTH_PENALTY));
        min_heap.push_back(std::move(beam));
        std::push_heap(min_heap.begin(), min_heap.end(), greater);
        if (min_heap.size() > GROUP_SIZE) {
            std::pop_heap(min_heap.begin(), min_heap.end(), greater);
            min_heap.pop_back();
        }
    }
    bool is_done(double best_sum_logprobs, size_t cur_len) {
        if (min_heap.size() < GROUP_SIZE) {
            return false;
        }
        switch (stop_criteria) {
            case StopCriteria::early: return done = true;
            case StopCriteria::heuristic: {
                double worst_score = min_heap.front().score;
                double highest_attainable_score = best_sum_logprobs / std::pow(double(cur_len), LENGTH_PENALTY);
                return done = worst_score >= highest_attainable_score;
            }
            case StopCriteria::never: {
                double worst_score = min_heap.front().score;
                double length = LENGTH_PENALTY > 0.0f ? MAX_NEW_TOKENS : cur_len;
                double highest_attainable_score = best_sum_logprobs / std::pow(length, LENGTH_PENALTY);
                return done = worst_score >= highest_attainable_score;
            }
            default: throw std::runtime_error("Never reached");
        }
    }
};

struct TokenToBeam {int64_t token_idx; size_t beam_idx;};

struct GroupBeamSearcher {
    const ov::Tensor input_ids;  // input_ids is going to be used to prepend to beams, thus take rvalue in constructor to ensure it's not overriden
    const size_t n_groups;
    const size_t group_size;
    const StopCriteria stop_criteria;
    const size_t no_repeat_ngram_size;
    const float diversity_penalty;
    const double length_penalty;
    const int64_t eos_token;
    const int64_t pad_token;
    std::function<bool(const Beam&)> stopping_criteria;
    std::vector<Group> groups;

    GroupBeamSearcher(ov::Tensor&& input_ids, size_t n_groups, size_t group_size, StopCriteria stop_criteria, size_t no_repeat_ngram_size, float diversity_penalty, double length_penalty, int64_t eos_token, int64_t pad_token, std::function<bool(const Beam&)> stopping_criteria=[](const Beam&){return false;}) : input_ids{input_ids}, n_groups{n_groups}, group_size{group_size}, stop_criteria{stop_criteria}, no_repeat_ngram_size{no_repeat_ngram_size}, diversity_penalty{diversity_penalty}, length_penalty{length_penalty}, eos_token{eos_token}, pad_token{pad_token}, stopping_criteria{stopping_criteria}, groups{n_groups} {
        if (1 != input_ids.get_shape()[0]) {
            throw std::runtime_error("input_ids batch size must be 1");
        }
        for (Group & group : groups) {
            group.ongoing.resize(GROUP_SIZE);
            group.ongoing.front().score = 0.0;
        }
    }

    std::vector<TokenToBeam> process(const ov::Tensor& logits) {
        std::vector<TokenToBeam> next_tokens;
        next_tokens.reserve(n_groups * group_size);
        std::vector<std::vector<size_t>> global_beam_ids{n_groups};
        for (std::vector<size_t>& vec : global_beam_ids) {
            vec.resize(GROUP_SIZE);
        }
        size_t temp_count = 0;
        for (size_t group_idx = 0; group_idx < n_groups; ++group_idx) {
            Group& group = groups[group_idx];
            if (group.done) {
                continue;
            }
            for (size_t beam_idx = 0; beam_idx < GROUP_SIZE; ++beam_idx) {
                global_beam_ids[group_idx][beam_idx] = temp_count;
                if (!group.ongoing[beam_idx].tokens.empty() && logits.get_shape()[0] != 1) {  // TODO: all of beams
                    ++temp_count;
                }
            }
        }

        for (size_t group_idx = 0; group_idx < n_groups; ++group_idx) {
            Group& group = groups[group_idx];
            if (group.done) {
                for (Beam& beam : group.ongoing) {
                    beam.tokens.push_back(pad_token);
                }
                continue;
            }
            std::vector<Beam> candidates;
            candidates.reserve(2 * GROUP_SIZE);
            for (size_t beam_idx = 0; beam_idx < GROUP_SIZE; ++beam_idx) {
                if (logits.get_shape()[0] <= global_beam_ids[group_idx][beam_idx]) {
                    throw std::runtime_error("logits batch size doesn't match the number of beams");
                }
                size_t vocab_size = logits.get_shape().back();
                std::vector<float> temp;
                size_t batch_offset = global_beam_ids[group_idx][beam_idx] * logits.get_shape()[1] * logits.get_shape()[2];
                const float* beam_logits = logits.data<const float>() + batch_offset + (logits.get_shape()[1] - 1) * vocab_size;
                float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
                float log_sum = std::log(std::accumulate(beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
                    return accumulated + std::exp(to_add - max_logit);
                }));
                struct Token {double log_prob; int64_t idx;};
                std::vector<Token> tokens;
                tokens.reserve(vocab_size);
                for (size_t idx = 0; idx < vocab_size; ++idx) {
                    tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});
                }
                for (size_t prev_group_idx = 0; prev_group_idx < group_idx; ++prev_group_idx) {  // TODO: range based for
                    for (size_t prev_beam_idx = 0; prev_beam_idx < GROUP_SIZE; ++prev_beam_idx) {
                        tokens[size_t(groups[prev_group_idx].ongoing[prev_beam_idx].tokens.back())].log_prob -= diversity_penalty;
                    }
                }
                std::vector<int64_t>& other_tokens = group.ongoing[beam_idx].tokens;
                std::vector<int64_t> full_text;
                for (size_t idx = 0; idx < input_ids.get_size(); ++idx) {
                    full_text.push_back(input_ids.data<int64_t>()[idx]);
                }
                full_text.insert(full_text.end(), other_tokens.begin(), other_tokens.end());
                if (full_text.size() > 1 && full_text.size() >= no_repeat_ngram_size) {
                    for (int64_t banned_token : kmp_search(full_text, {full_text.end() - ptrdiff_t(no_repeat_ngram_size) + 1, full_text.end()})) {
                        tokens[size_t(banned_token)].log_prob = -std::numeric_limits<float>::infinity();
                    }
                }
                std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                    return left.log_prob > right.log_prob;  // Most probable tokens in front
                });
                size_t new_token_idx = 0;
                for (int added_count = 0; added_count < int(2 * GROUP_SIZE); ++added_count) {
                    Beam new_candidate = group.ongoing[beam_idx];
                    new_candidate.score += tokens[new_token_idx].log_prob;
                    new_candidate.tokens.push_back(tokens[new_token_idx].idx);
                    new_candidate.global_beam_idx = global_beam_ids[group_idx][beam_idx];
                    ++new_token_idx;
                    if (stopping_criteria(new_candidate)) {
                        group.finish(std::move(new_candidate), input_ids.get_size());
                        --added_count;
                    } else {
                        candidates.push_back(std::move(new_candidate));
                    }
                }
            }
            // Sample 2 * GROUP_SIZE next tokens to get at least 1 non EOS token per beam
            std::partial_sort(candidates.begin(), candidates.begin() + 2 * GROUP_SIZE, candidates.end(), greater);  // Highest score beams in front
            size_t cur_len = candidates.front().tokens.size();
            group.ongoing.clear();
            for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
                if (eos_token == candidates[cand_idx].tokens.back()) {  // TODO: idx->token_id
                    // if beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= GROUP_SIZE) {
                        continue;
                    }
                    candidates[cand_idx].tokens.resize(candidates[cand_idx].tokens.size() - 1);
                    group.finish(std::move(candidates[cand_idx]), input_ids.get_size());
                } else {
                    group.ongoing.push_back(std::move(candidates[cand_idx]));
                    next_tokens.push_back({group.ongoing.back().tokens.back(), group.ongoing.back().global_beam_idx});
                    if (group.ongoing.size() == GROUP_SIZE) {
                        break;
                    }
                }
            }
            group.is_done(cur_len + input_ids.get_size(), group.ongoing.front().score);  // TODO: that requires group.ongoing to be not empty
            if (group.done) {
                next_tokens.resize(next_tokens.size() - group.ongoing.size());
            }
        }
        return next_tokens;
    }
};

// Consume GroupBeamSearcher to prohibit usage after because ongoing
// beams are moved and hypotheses are overpopulated while merging
std::vector<std::vector<Beam>> finilize(GroupBeamSearcher&& group_beam_searcher) {
    std::vector<std::vector<Beam>> finalized;
    for (Group& group : group_beam_searcher.groups) {
        if (group.is_done(group.ongoing.front().tokens.size() + group_beam_searcher.input_ids.get_size(), group.ongoing.front().score)) {
            continue;
        }
        for (Beam& beam: group.ongoing) {  // TODO: &&  // TODO: iterator based push
            group.finish(std::move(beam), group_beam_searcher.input_ids.get_size());
        }
    }
    for (Group& group: group_beam_searcher.groups) {
        finalized.emplace_back();
        std::sort_heap(group.min_heap.begin(), group.min_heap.end(), greater);
        for (const Beam& beam: group.min_heap) {
            finalized.back().push_back(beam);
        }
    }
    return finalized;
}

int main(int argc, char* argv[]) try {
    if (argc != 13) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    MAX_NEW_TOKENS = std::stoi(argv[5]);
    N_GROUPS = std::stoi(argv[6]);
    GROUP_SIZE = std::stoi(argv[7]);
    if (std::string{"early"} == argv[8]) {
        stop_criteria = StopCriteria::early;
    } else if (std::string{"heuristic"} == argv[8]) {
        stop_criteria = StopCriteria::heuristic;
    } else if (std::string{"never"} == argv[8]) {
        stop_criteria = StopCriteria::never;
    } else {
        throw std::runtime_error("Unknown stop_criteria value");
    }
    NO_REPEAT_NGRAM_SIZE = std::stoi(argv[9]);
    DIVERSITY_PENALTY = std::stof(argv[10]);
    LENGTH_PENALTY = std::stod(argv[11]);
    EOS_TOKEN = 2;
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            -1, -1
        }},
        {1, ov::PartialShape{
            -1, -1
        }},
        {2, ov::PartialShape{
            -1, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = -1;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::InferRequest ireq = core.compile_model(model, "CPU", ov::cache_dir("llm-cache")).create_infer_request();
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("input_ids").get_size()});
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
    ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::Shape shape = inputs.at(idx).get_partial_shape().get_min_shape();
        shape[0] = 1;
        ireq.get_input_tensor(idx).set_shape(shape);
    }

    int64_t pad_token = std::stoi(argv[12]);  // There's no way to extract the value from the tokenizer for now
    GroupBeamSearcher group_beam_searcher{std::move(input_ids), N_GROUPS, GROUP_SIZE, stop_criteria, NO_REPEAT_NGRAM_SIZE, DIVERSITY_PENALTY, LENGTH_PENALTY, EOS_TOKEN, pad_token};
    for (size_t length_count = 0; length_count < MAX_NEW_TOKENS; ++length_count) {
        ireq.infer();
        std::vector<TokenToBeam> next_tokens = group_beam_searcher.process(ireq.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        ireq.get_tensor("input_ids").set_shape({batch_size, 1});
        ov::Tensor attention_mask = ireq.get_tensor("attention_mask");
        ov::Shape mask_shape = attention_mask.get_shape();
        mask_shape[0] = batch_size;
        ++mask_shape[1];
        attention_mask.set_shape(mask_shape);
        std::fill_n(attention_mask.data<int64_t>(), shape_size(mask_shape), 1);
        ireq.get_tensor("position_ids").set_shape({batch_size, 1});
        std::fill_n(ireq.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape[1] - 1);
        for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
            ov::Shape shape = ireq.get_output_tensor(tensor_idx - 2).get_shape();
            shape[0] = batch_size;
            ireq.get_input_tensor(tensor_idx).set_shape(shape);
        }
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ireq.get_tensor("input_ids").data<int64_t>()[batch_idx] = next_tokens[batch_idx].token_idx;
            for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
                ov::Tensor present = ireq.get_output_tensor(tensor_idx - 2);
                ov::Shape present_begin = {next_tokens[batch_idx].beam_idx, 0, 0, 0};
                ov::Shape present_end = present.get_shape();
                present_end[0] = next_tokens[batch_idx].beam_idx + 1;
                ov::Tensor past = ireq.get_input_tensor(tensor_idx);
                ov::Shape past_begin = {batch_idx, 0, 0, 0};
                ov::Shape past_end = past.get_shape();
                past_end[0] = batch_idx + 1;
                ov::Tensor{present, present_begin, present_end}.copy_to(ov::Tensor{past, past_begin, past_end});
            }
        }
    }
    for (const std::vector<Beam>& group : finilize(std::move(group_beam_searcher))) {
        std::cout << "Group:\n";
        for (const Beam& beam : group) {
            std::cout << beam.score << ": " << openvino_extensions::unpack_strings(detokenize(detokenizer, beam.tokens)).front() << '\n';
        }
    }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
