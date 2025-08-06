// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_npu.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(ov::genai::Tokenizer& lhs, ov::genai::Tokenizer& rhs);
} // ov::genai

namespace {
ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}


ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
}
} // anonymous namespace

namespace ov {
namespace genai {
    LLMInferWrapper::LLMInferWrapper(
    const ov::genai::ModelDesc& model_desc
) : m_properties(model_desc.properties),
    m_generation_config(model_desc.generation_config),
    m_tokenizer(model_desc.tokenizer) {
    m_kv_pos = ov::genai::utils::get_kv_axes_pos(model_desc.model);
    if (model_desc.device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_kvcache_total = kv_desc.max_prompt_len + kv_desc.min_response_len;
        m_request = compiled.create_infer_request();
    } else {
        // TODO: We might need it for manipulations with indices
        // utils::apply_gather_before_matmul_transformation(model_desc.model);
        m_request = ov::genai::utils::singleton_core().compile_model(model_desc.model, model_desc.device, m_properties).create_infer_request();
    }
}

ov::genai::GenerationConfig LLMInferWrapper::get_generation_config() const {
    return m_generation_config;
}

void LLMInferWrapper::set_generation_config(ov::genai::GenerationConfig config) {
    m_generation_config = config;
}

int64_t LLMInferWrapper::infer_first(const ov::Tensor &input_ids,
                                     const ov::Tensor &attention_mask,
                                     const ov::Tensor &position_ids) {
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    m_request.infer();
    m_num_processed_tokens = input_ids.get_shape()[1];
    m_first_prompt_len = m_num_processed_tokens;

    // Initialize placeholder data for next inferences on input_ids of size 1 (if any)
    // with values of previous iteration for simple increment on next iteration:
    m_new_input_token = -1;
    m_new_position_id = m_num_processed_tokens - 1;
    m_new_atten_mask_data = std::vector<int64_t>(m_num_processed_tokens, 1);
    set_already_allocated_input_for_1_token();

    last_token = std::get<int64_t>(sample_tokens(get_logits(), 1u));
    return last_token;
}

bool LLMInferWrapper::can_infer() {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "can_infer() can be called only after infer_first()!");

    // FIXME: Add condition to get out of KV-Cache length for static models.
    auto stop_token_ids = m_generation_config.stop_token_ids;
    if (!m_generation_config.ignore_eos && (last_token == m_generation_config.eos_token_id)) {
        return false;
    }
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), last_token) != stop_token_ids.end()) {
       return false;
    }
    if (m_num_processed_tokens - m_first_prompt_len + 1 >= m_generation_config.get_max_new_tokens()) {
        return false;
    }
       
    return true;
}

int64_t LLMInferWrapper::infer_next(int64_t token) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next() can be called only after infer_first()!");

    // FIXME: Uncomment for static model and throw exception instead
    // if (m_num_processed_tokens + tokens_size == m_kvcache_total) {
    //     m_sequence_group->set_out_of_memory();
    //     return -1;
    // }

    // Just change the variables here, as pointers to them are already set to corresponding tensors
    m_new_input_token = token;
    ++m_new_position_id;
    // However, attention_mask changes its shape on each iteration, it should be re-set explicitly
    m_new_atten_mask_data.push_back(1);
    m_request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1,m_new_atten_mask_data.size()}, (void*)&m_new_atten_mask_data[0]));

    m_request.infer();

    m_num_processed_tokens += 1u;

    last_token = std::get<int64_t>(sample_tokens(get_logits(), 1u));
    return last_token;
}

int64_t LLMInferWrapper::infer_next(const std::vector<int64_t> tokens) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next() can be called only after infer_first()!");

    auto logits = infer_next_internal(tokens);
    last_token = std::get<int64_t>(sample_tokens(logits, 1u));
    return last_token;
}

std::vector<int64_t> LLMInferWrapper::infer_next_return_all(const std::vector<int64_t> tokens) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next_return_all() can be called only after infer_first()!");

    auto logits = infer_next_internal(tokens);
    auto tokens_size = tokens.size();
    auto sampled_tokens = std::get<std::vector<int64_t>>(sample_tokens(logits, tokens_size));
    last_token = sampled_tokens[tokens_size - 1];
    return sampled_tokens;
}

ov::Tensor LLMInferWrapper::get_logits() {
    return m_request.get_tensor("logits");
}

std::size_t LLMInferWrapper::get_num_processed_tokens() const {
    return m_num_processed_tokens;
}

void LLMInferWrapper::trimm_kv_cache(const size_t tokens_to_remove) {
    // Trim kv_cache values on tokens_to_remove
    ov::genai::utils::KVCacheState to_trim_state;
    to_trim_state.num_tokens_to_trim = tokens_to_remove;
    to_trim_state.seq_length_axis =  m_kv_pos.seq_len;
    to_trim_state.reset_mem_state = false;
    ov::genai::utils::trim_kv_cache(m_request, to_trim_state, {});
    m_num_processed_tokens -= tokens_to_remove;
}

void LLMInferWrapper::reset_state() {
    return m_request.reset_state();
}

ov::Tensor LLMInferWrapper::infer_next_internal(const std::vector<int64_t> tokens) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next_internal() can be called only after infer_first()!");

    size_t tokens_size = tokens.size();

    // FIXME: Uncomment for static model and throw exception instead
    // if (m_num_processed_tokens + tokens_size == m_kvcache_total) {
    //     m_sequence_group->set_out_of_memory();
    //     return -1;
    // }

    auto input_ids = m_request.get_tensor("input_ids");
    ov::Tensor new_input_ids(input_ids.get_element_type(), ov::Shape{BATCH_SIZE, tokens_size});
    std::copy_n(tokens.begin(), tokens_size, new_input_ids.data<int64_t>());
    m_request.set_tensor("input_ids", new_input_ids);

    // FIXME: For model with static shapes we can just copy after
    //        the prefilled tokens, no reshape is needed.
    auto attention_mask = m_request.get_tensor("attention_mask");
    ov::Tensor new_attention_mask(attention_mask.get_element_type(), ov::Shape{BATCH_SIZE, m_num_processed_tokens + tokens_size});
    std::copy_n(attention_mask.data<int64_t>(), m_num_processed_tokens, new_attention_mask.data<int64_t>());
    std::fill_n(new_attention_mask.data<int64_t>() + m_num_processed_tokens, tokens_size, 1);
    m_request.set_tensor("attention_mask", new_attention_mask);

    auto position_ids = m_request.get_tensor("position_ids");
    ov::Tensor new_position_ids(position_ids.get_element_type(), ov::Shape{BATCH_SIZE, tokens_size});
    std::iota(new_position_ids.data<int64_t>(),
              new_position_ids.data<int64_t>() + new_position_ids.get_size(),
              m_num_processed_tokens);
    m_request.set_tensor("position_ids", new_position_ids);

    m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    m_request.infer();

    m_num_processed_tokens += tokens_size;

    // Update pre-allocated inputs for 1 token and return back to use it
    // in case if next infer will be called on input_ids of size 1
    // (most frequent case).
    m_new_input_token = -1;
    m_new_position_id = m_num_processed_tokens - 1;
    for (std::size_t i = 0; i < tokens_size; ++i) {
        m_new_atten_mask_data.push_back(1);
    } 
    set_already_allocated_input_for_1_token();

    return get_logits();
}

void LLMInferWrapper::set_already_allocated_input_for_1_token() {
    m_request.set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1},  reinterpret_cast<void*>(&m_new_input_token)));
    m_request.set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1}, reinterpret_cast<void*>(&m_new_position_id)));
}

// TODO: Use already provided Sampler API, that will support both greedy and
//       multinomial decoding.
std::variant<int64_t, std::vector<int64_t>>
    LLMInferWrapper::sample_tokens(const ov::Tensor& logits, std::size_t num_tokens_to_sample) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "sample_tokens() can be called only after infer_first()!");

    // logits.shape = [1, seq_len, vocab_size].
    auto logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    std::size_t batch_size = logits_shape[0];
    OPENVINO_ASSERT(batch_size == 1);
    std::size_t seq_len = logits_shape[1];
    OPENVINO_ASSERT(num_tokens_to_sample <= seq_len);
    std::size_t vocab_size = logits_shape[2];

    auto sample_token = [&](const ov::Tensor& logits, std::size_t idx) {
        size_t sequence_offset = idx * vocab_size;
        float* logits_data = logits.data<float>() + sequence_offset;
        return std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    };

    if (num_tokens_to_sample == 1) {
        // Sample last logit:
        return sample_token(logits, seq_len - 1);
    } else {
        // Sample last num_tokens_to_sample logits:
        std::vector<int64_t> sampled_tokens;
        for (std::size_t i = 0; i < num_tokens_to_sample; i++) {
            sampled_tokens.push_back(sample_token(logits, seq_len - num_tokens_to_sample + i));
        }
        return sampled_tokens;
    }
}

void SpeculativeConfig::update_candidate_strategy(const size_t num_matches) {
    // Dynamically adjust number of generated candidates based on number of matches
    // we want to balance the benefits of getting candidates tokens correct with the
    // cost of forecasting incorrect candidates tokens.
    if (num_matches == num_pred_tokens) {
        num_pred_tokens = std::min(num_pred_tokens + 2, max_pred_tokens);
    } else {
        num_pred_tokens = std::max(int64_t(num_pred_tokens) - 1, int64_t(1));
    }
}

SpeculativeLLMPipelineNPU::SpeculativeLLMPipelineNPU(
    const ov::genai::ModelDesc& main_model_desc, 
    const ov::genai::ModelDesc& draft_model_desc
) : LLMPipelineImplBase(main_model_desc.tokenizer, main_model_desc.generation_config) {
    auto draft_model = draft_model_desc.model;

    // FIXME: slicing produces incorrect results for some models on NPU.
    // On NPU, applying slice the safe way is done by the underlying plugin
    if (draft_model_desc.device != "NPU") {
        utils::apply_slice_before_matmul_transformation(draft_model);
        // As draft_model_desc contains std::shared_ptr<ov::Model>,
        // this model update will be reflected in draft_model_desc
    }

    // TODO: We might need it for manipulations with indices
    // utils::apply_gather_before_matmul_transformation(main_model);
    // utils::apply_gather_before_matmul_transformation(draft_model);
    
    // Main and Draft model can have different tokenizers
    // to do: support retokenization: 154103
    ov::genai::Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    ov::genai::Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;
    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    m_tokenizer = main_model_tokenizer;
    
    // Draft model (which is smaller, less accurate but faster)
    auto draft_model_desc_copy = draft_model_desc;
    if (draft_model_desc_copy.device.empty()) {
        draft_model_desc_copy.device = main_model_desc.device;
    }
    if (draft_model_desc_copy.properties.empty()) {
        draft_model_desc_copy.properties = main_model_desc.properties;
    }
    m_draft_request = std::make_unique<LLMInferWrapper>(draft_model_desc_copy);

    // Main model (which is bigger, more accurate but slower)
    // FIXME: Need to support full logits tensor as output for main model on NPU.
    m_main_request = std::make_unique<LLMInferWrapper>(main_model_desc);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();

    // FIXME: Where to take it when draft model will be on NPU?
    size_t max_sequence_length = main_model_desc.generation_config.max_length;
    if (max_sequence_length == SIZE_MAX) {
        // FIXME: NPUW_LLM_MAX_PROMPT_LEN + NPUW_LLM_MIN_RESPONSE_LEN
        max_sequence_length = 100;
    }
    // FIXME: ? Use main_model.generation_config.num_assistant_tokens; It should be > 0, if we want draft_model.generation_config.is_speculative_decoding() == true.
    const std::size_t candidates_num = 5;
    m_speculative_config.max_seq_length = max_sequence_length;
    m_speculative_config.num_pred_tokens = candidates_num;
}

DecodedResults SpeculativeLLMPipelineNPU::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    auto start_time = std::chrono::steady_clock::now();

    std::string prompt = std::visit(overloaded{
        [](const std::string& prompt) {
            return prompt;
        },
        [](std::vector<std::string>& prompts) {
            OPENVINO_ASSERT(prompts.size() == 1u, "Currently only batch size=1 is supported");
            return prompts.front();
        }
    }, inputs);

    const GenerationConfig& config = generation_config.has_value() ? *generation_config : m_generation_config;

    ov::genai::TokenizedInputs tokenized_input;
    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        prompt = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
        tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
    } else {
        if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;
            auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            tokenized_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
        } else {
            // in case when chat_template was not found in tokenizer_config.json or set
            tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
        }
    }

    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(tokenized_input, config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    auto decode_stop_time =  std::chrono::steady_clock::now();

    if (m_is_chat_conversation) {
        auto answer = decoded_results.texts[0];
        if (m_chat_generation_finish_status == GenerationStatus::CANCEL)
            // If chat generation process was cancelled by user, let's rollback to previous state of history
            m_history.pop_back();
        else
            m_history.push_back({{"role", "assistant"}, {"content", answer}});
    }

    // generate_durations
    // decoded_results.perf_metrics = encoded_results.perf_metrics;
    // auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    // auto stop_time = std::chrono::steady_clock::now();
    // raw_counters.generate_durations.clear();
    // raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    // raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - start_time));
    // raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));
    // decoded_results.perf_metrics.m_evaluated = false;
    // decoded_results.perf_metrics.evaluate_statistics(start_time);
    return decoded_results;
}

EncodedResults SpeculativeLLMPipelineNPU::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    // from step()
    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;
    //

    auto start_time = std::chrono::steady_clock::now();

    // from generate()
    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();
    //

    ov::Tensor input_ids;
    ov::Tensor attention_mask;

    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    ov::Shape prompt_shape = input_ids.get_shape();
    const size_t batch_size = prompt_shape[0];
    OPENVINO_ASSERT(batch_size == 1u, "Currently only batch size=1 is supported");

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    OPENVINO_ASSERT(config.is_greedy_decoding() || config.is_multinomial(),
        "Currently only greedy and multinomial decoding are supported");

    OPENVINO_ASSERT(config.num_return_sequences == 1u,
        "Currently only \"num_return_sequences\" equal to 1 is supported!");

    // FIXME: Update conditionally:
    m_main_request->set_generation_config(config);
    auto prompt_len = prompt_shape[1];
    m_speculative_config.max_seq_length = prompt_len + config.get_max_new_tokens(prompt_len);

    // Config draft model to not stop on EOS and remove stop strings:
    ov::genai::GenerationConfig draft_config = m_draft_request->get_generation_config();
    draft_config.ignore_eos = true;
    draft_config.stop_strings = {};
    draft_config.validate();
    m_draft_request->set_generation_config(draft_config);

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // FIXME: Return back for the static draft model.
    // NB: Check if there is enough space in KV-cache to process input prompt
    // auto prompt_len = prompts_shape[1];
    // if (prompt_len > m_max_prompt_len) {
    //     OPENVINO_THROW("Static Stateful LLM pipeline may only process prompts up to "
    //                    + std::to_string(m_max_prompt_len) + " tokens. "
    //                    + "Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
    // }

    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // To collect KV-cache for the prompt and to get the next token, run the very first infer request
    // for draft and main models:
    auto out_token = m_main_request->infer_first(input_ids, attention_mask, position_ids);
    m_draft_request->infer_first(input_ids, attention_mask, position_ids);

    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto draft_logits = m_draft_request->get_logits();
    auto main_logits = m_main_request->get_logits();
    size_t draft_vocab_size = draft_logits.get_shape().back();
    size_t main_vocab_size = main_logits.get_shape().back();
    OPENVINO_ASSERT(draft_vocab_size == main_vocab_size,
                    "Vocab sizes should be the same for the both: main and draft models!");


    // FIXME: Apply this logic carefully in LLMInferRequest of prefill model,
    //        if needed.
    // FIXME: Here is workaround to get only useful units of returned logits.
    //        If SliceOut is applied, there will be only 1 useful logit returned,
    //        nothing is required here.
    //        Other way, model will return logits of full context length,
    //        as internally prefill model is specially reshaped to return them.
    //        Fix should be done on OpenVINO side, so the model should return only
    //        useful logits of input prompt length, dropping the implementation-related
    //        padding ones.
    // auto sequence_len = all_logits.get_shape()[1];
    // if (sequence_len > 1) {
    //     logits = make_tensor_slice(all_logits, 1, sequence_len - prompt_len, sequence_len);
    // }
    OPENVINO_ASSERT(draft_logits.get_shape().at(1) <= main_logits.get_shape().at(1),
                    "Num of generated useful logits from draft models should be less"
                    "or equal than ones from main model.");

    auto streaming_status = stream_generated_tokens(streamer_ptr, std::vector<int64_t> {out_token});

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
    // Last generated token by draft model needs to be prepended before next run if it is accepted by the main model!
    // So it will get into context too.
    int64_t draft_prefix_token = -1;
    while (m_main_request->can_infer() && (streaming_status == ov::genai::StreamingStatus::RUNNING)) {
        // Phase 1: Generation of candidates with the draft model:
        std::vector<int64_t> candidates;
        // Limit candidates size by num_pred_tokens or by max_seq_length:
        // FIXME: draft_prefix_token isn't taken into account!
        // FIXME: How max_seq_length will limit further generation of main model?
        size_t candidates_to_generate = std::min(m_speculative_config.num_pred_tokens,
            m_speculative_config.max_seq_length - m_draft_request->get_num_processed_tokens() - 1);
        candidates.reserve(candidates_to_generate);

        // If draft_prefix_token is present, prepend it to out_token in order to collect KV cache for it
        auto candidate = out_token;
        if (draft_prefix_token != -1) {
            std::vector<int64_t> tokens_to_infer = {draft_prefix_token, out_token};
            // TODO: Handle OOM exception for static model here.
            candidate = m_draft_request->infer_next(tokens_to_infer);
            candidates.push_back(candidate);
            candidates_to_generate--;
        }
        for (size_t i = 0; i < candidates_to_generate; i++) {
            // TODO: Handle OOM exception for static model here.
            candidate = m_draft_request->infer_next(candidate);
            candidates.push_back(candidate);
        }
        
        // Phase 2. Main inference.
        // For the main network, candidates_size + 1 tokens will be fed at once in a single infer request:
        // last token from previous main inference + all candidates from the draft stage
        // FIXME: How max_seq_length will be handled?
        std::vector<int64_t> input_for_main(candidates.begin(), candidates.end());
        input_for_main.insert(input_for_main.begin(), {out_token});
        // Note: If model isn't sliced to return logit only for the last element,
        // then it returns logits for all elements of the input prompt.
        // In that tensor, for each token `t` of the input prompt it contains
        // distribution (over the vocabulary) for the next possible token
        // that is generated based on subsequence [first token,...,`t`]
        // of the input prompt.
        // TODO: Handle OOM exception for static model here.
        auto ref_out_tokens = m_main_request->infer_next_return_all(input_for_main);

        // Phase 3. Check if main model produced the same tokens as input candidates:
        size_t accepted_tokens_number = 0u;
        // Last token is a new token from the main model, skip it:
        for (size_t i = 0; i < ref_out_tokens.size() - 1; ++i) {
            if (ref_out_tokens[i] != candidates[i]) {
                break;
            }
            accepted_tokens_number++;
        }

        auto mismatched_candidates = candidates.size() - accepted_tokens_number;
        std::vector<int64_t> validated_tokens(candidates.begin(), candidates.end() - mismatched_candidates);
        out_token = ref_out_tokens.back();
        validated_tokens.push_back(out_token);
    
        // Phase 4: Update inference wrappers based on found matches and mismatches
        // This is the case when main model accepted all candidates from draft model
        // we need to collect kv cache for draft last generated token by infering it.n
        if (mismatched_candidates == 0) {
            draft_prefix_token = candidate;
        } else {
            m_draft_request->trimm_kv_cache(mismatched_candidates - 1);
            m_main_request->trimm_kv_cache(mismatched_candidates);
        }

        m_speculative_config.update_candidate_strategy(accepted_tokens_number);
        // Should be enough, if all will be streamed from logits?
        streaming_status = stream_generated_tokens(streamer_ptr, validated_tokens);

        // raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        // raw_perf_counters.m_batch_sizes.emplace_back(batch_size);
    }

    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    m_draft_request->reset_state();
    m_main_request->reset_state();
    
    ov::genai::EncodedResults results;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);
    // results.tokens[0] = sequence->get_generated_ids();
    // results.scores[0] = sequence->get_cumulative_log_prob();
    // m_chat_generation_finish_status = m_streaming_status;

    // auto stop_time = std::chrono::steady_clock::now();
    // If is called without tokenization then that stat will not be reported.
    // auto& metrics = results.perf_metrics;
    // metrics.num_input_tokens = batch_size * input_ids.get_shape().at(1);
    // metrics.load_time = this->m_load_time_ms;
    // metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    // metrics.evaluate_statistics(start_time);
    return results;
}

void SpeculativeLLMPipelineNPU::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void SpeculativeLLMPipelineNPU::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

SpeculativeLLMPipelineNPU::~SpeculativeLLMPipelineNPU() {
    // FIXME: Do we need it?
    // m_request.get_compiled_model().release_memory();
}
}  // namespace genai
}  // namespace ov
