// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/pipeline_static.hpp"

#include <chrono>
#include <regex>

#include "openvino/runtime/intel_npu/properties.hpp"
#include "utils.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper.hpp"
#include "whisper/config.hpp"
#include "whisper/whisper_utils.hpp"

#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset13.hpp"

using ov::genai::MicroSeconds;

namespace {

constexpr size_t MAX_PROMPT_LEN = 4;

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val) {
    auto* tensor_data = tensor.data<T>();
    std::fill(tensor_data, tensor_data + tensor.get_size(), fill_val);
}

template <typename T>
void copy_to_tensor(const std::vector<T>& src_vec, ov::Tensor dst_tensor) {
    auto* dst_ptr = dst_tensor.data<T>();
    OPENVINO_ASSERT(src_vec.size() == dst_tensor.get_size());
    std::copy(src_vec.begin(), src_vec.end(), dst_ptr);
}

ov::Tensor encode(ov::InferRequest& request,
                  std::vector<float>& mel_data,
                  const size_t feature_size,
                  const size_t nb_max_frames,
                  ov::genai::RawPerfMetrics& raw_metrics) {
    OPENVINO_ASSERT(mel_data.size() == feature_size * nb_max_frames,
                    "Mel spectrogram required size: ",
                    feature_size,
                    " * ",
                    nb_max_frames,
                    ". Actual size: ",
                    mel_data.size(),
                    ".");
    copy_to_tensor(mel_data, request.get_tensor("input_features"));

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    return request.get_tensor("last_hidden_state");
}

// FIXME: Duplicate from llm_pipeline_static.cpp - need to reuse instead of copy-paste
ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

void set_cross_attn_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
    // NB: Source outputs:
    // present.0.encoder.key
    // present.0.encoder.value

    // NB: Dest inputs:
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("encoder") == std::string::npos) {
            continue;
        }
        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");
        dest.set_tensor(with_past_input_name, source.get_tensor(source_output_name));
    }
}

void update_past_key_value(ov::InferRequest& source, ov::InferRequest& dest, const size_t kv_pos = 0u) {
    // NB: Source outputs:
    // present.0.decoder.key
    // present.0.decoder.value

    // NB: Dest inputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("decoder") == std::string::npos) {
            continue;
        }

        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");

        auto src_kv_tensor = source.get_tensor(source_output_name);
        auto dst_kv_tensor = dest.get_tensor(with_past_input_name);
        auto kv_size = src_kv_tensor.get_shape()[2];
        auto dst_kv_tensor_slice = make_tensor_slice(dst_kv_tensor, 2u, kv_pos, kv_pos + kv_size);
        src_kv_tensor.copy_to(dst_kv_tensor_slice);
    }
}

void set_decoder_input_ids(ov::InferRequest& decoder,
                           const std::vector<int64_t>& init_ids) {
    // NB: input_ids format: [token1, token2, pad, pad]
    auto padded_input_ids = decoder.get_tensor("input_ids");
    OPENVINO_ASSERT(padded_input_ids.get_size() >= init_ids.size());
    OPENVINO_ASSERT(padded_input_ids.get_element_type() == ov::element::i64);
    std::fill_n(padded_input_ids.data<int64_t>(), padded_input_ids.get_size(), 0u);
    std::copy_n(
        &init_ids[0], init_ids.size(),
        padded_input_ids.data<int64_t>()
    );

    // NB: attention_mask format: [1, 1, 0, 0]
    auto padded_attention_mask = decoder.get_tensor("attention_mask");
    OPENVINO_ASSERT(padded_attention_mask.get_size() >= init_ids.size());
    auto* padded_mask_data = padded_attention_mask.data<int64_t>();
    std::fill_n(padded_mask_data, padded_attention_mask.get_size(), 0u);
    std::fill_n(padded_mask_data, init_ids.size(), 1u);
}

void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const std::vector<int64_t>&  generated_tokens) {
    const bool initial_step = generated_tokens.empty();

    if (initial_step) {
        ov::genai::do_suppress_tokens(logits, 0, config.begin_suppress_tokens);
    }

    ov::genai::do_suppress_tokens(logits, 0, config.suppress_tokens);

    if (return_timestamps) {
        ov::genai::process_whisper_timestamp_logits(logits, 0, config, generated_tokens, initial_step);
    }

}

ov::Tensor decode(ov::Tensor& encoder_hidden_state,
                  ov::InferRequest& decoder,
                  const std::vector<int64_t>& init_ids,
                  ov::genai::RawPerfMetrics& raw_metrics) {
    // NB: Fill decoder inputs
    encoder_hidden_state.copy_to(decoder.get_tensor("encoder_hidden_states"));
    set_decoder_input_ids(decoder, init_ids);
    ov::genai::utils::infer_with_perf_metrics(decoder, raw_metrics);
    // NB: Processing here only non-empty tokens
    return make_tensor_slice(decoder.get_tensor("logits"), 1, 0, init_ids.size());
}

ov::Tensor decode_with_past(ov::InferRequest& decoder_with_past,
                            const int64_t input_id,
                            const int64_t position_id,
                            ov::genai::RawPerfMetrics& raw_metrics) {
    decoder_with_past.get_tensor("input_ids").data<int64_t>()[0] = input_id;
    decoder_with_past.get_tensor("cache_position").data<int64_t>()[0] = position_id;
    OPENVINO_ASSERT(position_id >= 1);
    decoder_with_past.get_tensor("attention_mask").data<float>()[position_id - 1] = 0.0f;

    ov::genai::utils::infer_with_perf_metrics(decoder_with_past, raw_metrics);
    return decoder_with_past.get_tensor("logits");
}

void zero_past_key_values(ov::InferRequest& request) {
    for (auto& input : request.get_compiled_model().inputs()) {
        std::string past_key_value_decoder_name = input.get_any_name();
        if (past_key_value_decoder_name.find("decoder") == std::string::npos ||
            past_key_value_decoder_name.find("past_key_values") == std::string::npos) {
            continue;
        }
        fill_tensor<ov::float16>(request.get_tensor(past_key_value_decoder_name), 0);
    }
}

void prepare_decoder_with_past(ov::InferRequest& decoder_with_past, ov::InferRequest& decoder, const size_t init_ids_size) {
    // NB: Prepare attention mask to be in a format [0, 0, 0, 1, 1, 1, ..., 1, 0, 1]
    auto padded_attention_mask = decoder_with_past.get_tensor("attention_mask");
    OPENVINO_ASSERT(padded_attention_mask.get_size() >= init_ids_size);
    OPENVINO_ASSERT(padded_attention_mask.get_size() >= 2);
    auto* padded_mask_data = padded_attention_mask.data<float>();
    std::fill(padded_mask_data, padded_mask_data + init_ids_size, 0);
    std::fill(padded_mask_data + init_ids_size, padded_mask_data + padded_attention_mask.get_size() - 2, 1);
    padded_mask_data[padded_attention_mask.get_size() - 2] = 0;
    padded_mask_data[padded_attention_mask.get_size() - 1] = 1;

    // NB: Zero past_key_values.*.decoder.value tensors
    zero_past_key_values(decoder_with_past);
    // NB: Copy KV-caches from decoder
    set_cross_attn_key_value(decoder, decoder_with_past);
    update_past_key_value(decoder, decoder_with_past);
};

int64_t detect_language(ov::Tensor& encoder_hidden_state,
                        ov::InferRequest& decoder,
                        const ov::genai::WhisperGenerationConfig& config,
                        ov::genai::RawPerfMetrics& raw_metrics) {
    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int64_t> init_ids{config.decoder_start_token_id};
    set_decoder_input_ids(decoder, init_ids);

    const auto infer_start = std::chrono::steady_clock::now();
    decoder.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    auto output_tensor = decoder.get_tensor("logits");

    auto logits_data = output_tensor.data<float>();

    int64_t output_token;
    float max_prob = -std::numeric_limits<float>::infinity();

    for (auto [_, lang_token] : config.lang_to_id) {
        auto prob = logits_data[lang_token];
        if (prob > max_prob) {
            max_prob = prob;
            output_token = lang_token;
        }
    }

    return output_token;
}

std::vector<int64_t> prepare_init_ids(ov::Tensor& encoder_hidden_state,
                                      ov::InferRequest& decoder,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const bool return_timestamps,
                                      ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        if (return_timestamps) {
            return std::vector<int64_t>{config.decoder_start_token_id};
        } else {
            return std::vector<int64_t>{config.decoder_start_token_id,
                                        config.no_timestamps_token_id};
        }
    }

    int64_t language_token_id = 0;
    if (config.language.has_value()) {
        std::string language = *config.language;
        if (config.lang_to_id.count(language)) {
            language_token_id = config.lang_to_id.at(language);
        }
    } else {
        language_token_id = detect_language(encoder_hidden_state, decoder, config, raw_metrics);
    }

    int64_t task_token_id = config.transcribe_token_id;
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = config.translate_token_id;
    }

    if (return_timestamps) {
        return std::vector<int64_t>{config.decoder_start_token_id,
                                    language_token_id,
                                    task_token_id};
    }

    return std::vector<int64_t>{config.decoder_start_token_id,
                                language_token_id,
                                task_token_id,
                                config.no_timestamps_token_id};
}

void stream_generated_tokens(const std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                             ov::genai::GenerationHandle& handle,
                             const bool return_timestamps) {
    if (return_timestamps || !streamer_ptr || !handle->can_read()) {
        return;
    }

    std::unordered_map<uint64_t, ov::genai::GenerationOutput> token = handle->read();

    auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
    if (streaming_status != ov::genai::StreamingStatus::RUNNING) {
        streaming_status == ov::genai::StreamingStatus::CANCEL ? handle->cancel() : handle->stop();
    }
}

std::pair<ov::genai::EncodedResults, bool> full_decode(ov::Tensor& encoder_hidden_state,
                                                       const ov::genai::WhisperGenerationConfig& config,
                                                       ov::genai::WhisperInitializedModels& models,
                                                       std::vector<int64_t> init_ids,
                                                       const bool return_timestamps,
                                                       ov::genai::RawPerfMetrics& raw_metrics,
                                                       const std::shared_ptr<ov::genai::StreamerBase> streamer,
                                                       ov::genai::Sampler& sampler,
                                                       ov::genai::SequenceGroup::Ptr sequence_group) {
    auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                    sequence_group->get_sampling_parameters());

    auto logits = decode(encoder_hidden_state, models.decoder, init_ids, raw_metrics);
    process_whisper_logits(logits, config, return_timestamps, {});

    // sample last token only
    int64_t output_sequence_len = logits.get_shape().at(1);
    sequence_group->schedule_tokens(sequence_group->get_prompt_len());
    sequence_group->set_output_seq_len(output_sequence_len);

    sampler.sample({sequence_group}, logits);
    stream_generated_tokens(streamer, handle, return_timestamps);

    prepare_decoder_with_past(models.decoder_with_past, models.decoder, init_ids.size());

    while (!sequence_group->has_finished() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled()) {
        sequence_group->schedule_tokens(1);
        const auto running_sequences = sequence_group->get_running_sequences();
        OPENVINO_ASSERT(running_sequences.size() == 1u);
        auto last_token = running_sequences.front()->get_generated_ids().back();
        auto last_idx = running_sequences.front()->get_generated_ids().size() - 1;

        auto logits = decode_with_past(models.decoder_with_past,
                                       last_token,
                                       last_idx + init_ids.size(),
                                       raw_metrics);
        process_whisper_logits(logits, config, return_timestamps, running_sequences.front()->get_generated_ids());
        update_past_key_value(models.decoder_with_past, models.decoder_with_past, last_idx + init_ids.size());

        sampler.sample({sequence_group}, logits);
        stream_generated_tokens(streamer, handle, return_timestamps);
    }

    ov::genai::EncodedResults results;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    OPENVINO_ASSERT(sequence_group->get_finished_sequences().size() == 1u);
    auto sequence = sequence_group->get_finished_sequences().front();
    results.tokens[0] = sequence->get_generated_ids();
    results.scores[0] = sequence->get_cumulative_log_prob();

    sampler.clear_request_info(sequence_group->get_request_id());

    results.perf_metrics.raw_metrics = raw_metrics;

    return {results, (sequence_group->handle_stopped() || sequence_group->handle_cancelled())};
}

bool check_decoder_model_compatibility(const std::shared_ptr<ov::Model>& decoder) {
    for (auto input : decoder->inputs()) {
        if (input.get_any_name() == "attention_mask") {
            return true;
        }
    }
    return false;
}

void add_attention_mask_input(std::shared_ptr<ov::Model> model) {
    using namespace ov::pass::pattern;
    using namespace ov::op;
    class AttentionMaskInput : public ov::pass::MatcherPass {
    public:
        OPENVINO_MATCHER_PASS_RTTI("AttentionMaskInput");

        AttentionMaskInput(std::shared_ptr<ov::Model> model) {
            auto range = wrap_type<v4::Range>();
            auto convert1 = wrap_type<v0::Convert>({range});
            auto greater = wrap_type<v1::Greater>({convert1, any_input()});
            auto convert2 = wrap_type<v0::Convert>({greater});

            register_matcher(std::make_shared<Matcher>(convert2, this->get_type_info().name), [model](Matcher& m) {
                auto node = m.get_match_root();
                auto attention_mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                model->add_parameters({attention_mask});
                ov::replace_node(node, attention_mask);
                return false;
            });
        }
    };

    class AttentionMaskInput_2 : public ov::pass::MatcherPass {
    public:
        OPENVINO_MATCHER_PASS_RTTI("AttentionMaskInput_2");

        AttentionMaskInput_2(std::shared_ptr<ov::Model> model) {
            auto range = wrap_type<v4::Range>();
            auto unsqueeze1 = wrap_type<v0::Unsqueeze>({range, any_input()});
            auto unsqueeze2 = wrap_type<v0::Unsqueeze>({unsqueeze1, any_input()});
            auto unsqueeze3 = wrap_type<v0::Unsqueeze>({unsqueeze2, any_input()});
            auto lessequal = wrap_type<v1::LessEqual>({unsqueeze3, any_input()});

            register_matcher(std::make_shared<Matcher>(lessequal, this->get_type_info().name), [model](Matcher& m) {
                auto node = m.get_match_root();
                auto attention_mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                model->add_parameters({attention_mask});

                auto cst_0_0 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{1}, 0.0f);
                auto cst_0 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
                auto cst_1 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
                auto cst_2 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 2);

                auto attn_mask_shape = std::make_shared<v3::ShapeOf>(attention_mask, ov::element::i32)->output(0);
                auto gather = std::make_shared<v8::Gather>(attn_mask_shape, cst_1, cst_0)->output(0);
                auto attn_mask_size_minus_one = std::make_shared<v1::Subtract>(gather, cst_1)->output(0);
                auto slice = std::make_shared<v8::Slice>(attention_mask->output(0), cst_0, attn_mask_size_minus_one, cst_1, cst_1);

                auto unsqueeze_1 = std::make_shared<v0::Unsqueeze>(slice->output(0), cst_1->output(0));
                auto unsqueeze_2 = std::make_shared<v0::Unsqueeze>(unsqueeze_1->output(0), cst_2->output(0));

                auto equal = std::make_shared<v1::Equal>(unsqueeze_2->output(0), cst_0_0->output(0));

                ov::replace_node(node, equal);
                return false;
            });
        }
    };

    ov::pass::Manager pm;
    pm.register_pass<AttentionMaskInput>(model);
    pm.register_pass<AttentionMaskInput_2>(model);
    pm.run_passes(model);
}

void add_attention_mask_input(std::shared_ptr<ov::Model> model, bool transform_cross_attn, const uint32_t& hidden_state_seq_size) {
    using namespace ov::pass::pattern;
    using namespace ov::op;
    class AttentionMaskInput : public ov::pass::MatcherPass {
    public:
        OPENVINO_MATCHER_PASS_RTTI("AttentionMaskInput");

        AttentionMaskInput(std::shared_ptr<ov::Model> model, bool transform_cross_attn, const uint32_t& hidden_state_seq_size) {
            std::vector<std::shared_ptr<ov::Node>> self_attn_nodes;
            std::vector<std::shared_ptr<ov::Node>> cross_attn_nodes;
            const auto kAttnMaskPort = 3;
            for (const auto &node : model->get_ops()) {
                if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
                    if (node->inputs().size() > kAttnMaskPort && ov::is_type<v8::Slice>(node->input(kAttnMaskPort).get_source_output().get_node())) {
                        self_attn_nodes.push_back(node);
                    } else {
                        cross_attn_nodes.push_back(node);
                    }
                }
            }

            // Self-attention
            OPENVINO_ASSERT(!self_attn_nodes.empty());

            auto attention_mask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
            attention_mask->get_output_tensor(0).set_names({"attention_mask"});
            model->add_parameters({attention_mask});

            auto slice = self_attn_nodes[0]->input(kAttnMaskPort).get_source_output().get_node();
            auto cvt = std::make_shared<v0::Convert>(attention_mask->output(0), ov::element::f32);
            auto add = std::make_shared<v1::Add>(slice->output(0), cvt->output(0));

            auto trps = std::make_shared<v1::Transpose>(cvt->output(0), v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 0}));
            auto mtpl = std::make_shared<v1::Multiply>(trps->output(0), add->output(0));

            auto cst_ninf = std::make_shared<v0::Constant>(
                ov::element::f32,
                ov::Shape{1},
                std::vector<float>{-std::numeric_limits<float>::max()}
            );
            auto cst_1 = std::make_shared<v0::Constant>(
                ov::element::f32,
                ov::Shape{1},
                std::vector<float>{1}
            );
            auto cst_0 = std::make_shared<v0::Constant>(
                ov::element::f32,
                ov::Shape{1},
                std::vector<float>{0}
            );

            auto equal = std::make_shared<v1::Equal>(mtpl->output(0), cst_1->output(0));
            auto select = std::make_shared<v1::Select>(
                equal->output(0), cst_0->output(0), cst_ninf->output(0)
            );

            for (const auto& self_attn : self_attn_nodes) {
                self_attn->input(3).replace_source_output(select->output(0));
            }

            if (transform_cross_attn) {
                // Cross attn
                OPENVINO_ASSERT(!cross_attn_nodes.empty());
                auto shape_cst = std::make_shared<v0::Constant>(
                    ov::element::i64,
                    ov::Shape{2},
                    std::vector<int64_t>{MAX_PROMPT_LEN, 1}
                );

                auto target_shape = std::make_shared<v0::Constant>(
                    ov::element::i64,
                    ov::Shape{2},
                    std::vector<int64_t>{MAX_PROMPT_LEN, static_cast<int64_t>(hidden_state_seq_size)}
                );
                // FIXME: Must be transpose if batch present
                auto reshape = std::make_shared<v1::Reshape>(cvt->output(0), shape_cst->output(0), false);
                auto equal = std::make_shared<v1::Equal>(reshape->output(0), cst_1->output(0));
                auto select = std::make_shared<v1::Select>(
                    equal->output(0), cst_0->output(0), cst_ninf->output(0)
                );
                auto broadcast = std::make_shared<v3::Broadcast>(select->output(0), target_shape->output(0));
                auto unsq1 = std::make_shared<v0::Unsqueeze>(broadcast->output(0), cst_0->output(0));
                auto unsq2 = std::make_shared<v0::Unsqueeze>(unsq1->output(0), cst_1->output(0));
                for (const auto& cross_attn_node : cross_attn_nodes) {
                    if (cross_attn_node->inputs().size() == 3) {
                        auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(
                            cross_attn_node->input(0).get_source_output(),
                            cross_attn_node->input(1).get_source_output(),
                            cross_attn_node->input(2).get_source_output(),
                            unsq2->output(0),
                            false
                        );
                        ov::replace_node(cross_attn_node, sdpa);
                    } else {
                        cross_attn_node->input(3).replace_source_output(unsq2->output(0));
                    }
                }
            }
        }
    };

    ov::pass::Manager pm;
    pm.register_pass<AttentionMaskInput>(model, transform_cross_attn, hidden_state_seq_size);
    pm.run_passes(model);
}


ov::PartialShape get_encoder_hidden_state_shape(const std::shared_ptr<ov::Model>& encoder) {
    return encoder->output("last_hidden_state").get_partial_shape();
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const ov::PartialShape& lhstate_shape,
                       const bool with_past = false) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            if (with_past)
                new_shape = ov::PartialShape({1, kvcache_size + 1});
            else
                new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("cache_position") != std::string::npos) {
            new_shape = ov::PartialShape({1});
        } else if (input_name.find("encoder_hidden_states") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;     // batch_dim
            new_shape[1] = lhstate_shape[1];  // from encoder output{'last_hidden_state'}
            new_shape[2] = lhstate_shape[2];
        } else if (input_name.find("past_key_values") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;  // Use batch dim here
            new_shape[2] = input_name.find(".decoder") != std::string::npos
                               ? kvcache_size - input_size // kv_size for decoder
                               : lhstate_shape[1];  // hidden state size for encoder
        }
        new_shapes.emplace(input_name, new_shape);
    }

    model->reshape(new_shapes);
}

void reshape_to_static_encoder(std::shared_ptr<ov::Model> model, const size_t feature_size) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_features") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            OPENVINO_ASSERT(partial_shape.size() >= 3);
            new_shape = partial_shape;
            new_shape[0] = 1;  // batch_dim
            new_shape[1] = feature_size;
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

void reshape_input_ids(std::shared_ptr<ov::Model> model, const uint32_t input_size) {
    model->reshape({{"input_ids", ov::PartialShape({1, input_size})}});
}

void preprocess_encoder(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor preprocessor(model);

    preprocessor.input("input_features").tensor().set_element_type(ov::element::Type_t::f32);
    preprocessor.input("input_features").preprocess().convert_element_type(ov::element::Type_t::f32);
    preprocessor.output("last_hidden_state").tensor().set_element_type(ov::element::Type_t::f16);

    model = preprocessor.build();
}

void preprocess_decoder(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor preprocessor(model);

    for (auto tensor : model->inputs()) {
        if (tensor.get_any_name().find("encoder_hidden_states") != std::string::npos) {
            preprocessor.input("encoder_hidden_states").tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.input("encoder_hidden_states").preprocess().convert_element_type(ov::element::Type_t::f32);
        } else if (tensor.get_any_name().find("past_key_values") != std::string::npos) {
            preprocessor.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.input(tensor.get_any_name()).preprocess().convert_element_type();
        }
    }

    for (auto tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            preprocessor.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.output(tensor.get_any_name()).postprocess().convert_element_type();
        }
    }

    model = preprocessor.build();
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (int i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
    model->validate_nodes_and_infer_types();
    return model;
}

void set_name(std::shared_ptr<ov::Node> result, const std::string& name) {
    result->set_friendly_name(name);
    result->get_output_tensor(0).set_names({name});
}

void remove_input_kv_tensors(std::shared_ptr<ov::Model>& model) {
    const int CONCAT_CURR_KV_PORT = 1;

    ov::ParameterVector params_to_remove;
    ov::ResultVector results_to_add;
    ov::ResultVector results_to_remove;

    std::shared_ptr<ov::op::v0::Constant> cst_node;
    for (const auto& input: model->inputs()) {
        if (input.get_any_name().find("past_key_values") == std::string::npos) {
            continue;
        }

        // Remember this to remove later on when all reconnections performed
        params_to_remove.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr()));
        // KV-cache input tensor is connected with Concat and additionally can be connected with ShapeOf
        std::shared_ptr<ov::Node> concat_node;
        auto target_inputs = input.get_node()->output(0).get_target_inputs();
        for (const auto& target_input : target_inputs) {
            auto target_node = target_input.get_node();
            // Get Concat node
            if (strstr(target_node->get_type_name(), "Concat") != nullptr) {
                concat_node = target_node->shared_from_this();
            }
        }

        // Remove concat node
        OPENVINO_ASSERT(concat_node);
        auto cat_readers = concat_node->outputs()[0].get_target_inputs();

        // Result and SDPA
        OPENVINO_ASSERT(cat_readers.size() == 2);
        for (const auto& cat_reader : cat_readers) {
            if (strstr(cat_reader.get_node()->get_type_name(), "Result") != nullptr) {
                auto result_in = cat_reader;

                // Re-assign Result
                auto result_to_remove = ov::as_type_ptr<ov::op::v0::Result>(result_in.get_node()->shared_from_this());
                auto result_to_add    = std::make_shared<ov::op::v0::Result>(concat_node->inputs()[CONCAT_CURR_KV_PORT].get_source_output());
                set_name(result_to_add, result_to_remove->get_friendly_name());

                results_to_remove.push_back(std::move(result_to_remove));
                results_to_add.push_back(std::move(result_to_add));
            }
            if (strstr(cat_reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr || strstr(cat_reader.get_node()->get_type_name(), "FakeConvert") != nullptr) {
                auto sdpa_in = cat_reader;

                // Redirect KV from concat to SDPA or FakeConvert(for fp8)
                auto curr_kv = concat_node->inputs()[CONCAT_CURR_KV_PORT].get_source_output();
                sdpa_in.replace_source_output(curr_kv);
            }
        }

        // In case KV-cache also connected with ShapeOf
        if (target_inputs.size() == 2) {
            for (const auto& target_in : target_inputs) {
                if (strstr(target_in.get_node()->get_type_name(), "ShapeOf") != nullptr) {
                    auto shapeof_node = ov::as_type_ptr<ov::op::v3::ShapeOf>(target_in.get_node()->shared_from_this());
                    auto shape = std::vector<size_t>{1, size_t(input.get_partial_shape()[1].get_length()), 0, size_t(input.get_partial_shape()[3].get_length())};
                    cst_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, shape);

                    ov::replace_node(shapeof_node, cst_node);
                }
            }
        }
    }

    OPENVINO_ASSERT(cst_node);

    for (const auto& r: results_to_remove) {
        model->remove_result(r);
    }
    for (const auto& p: params_to_remove) {
        model->remove_parameter(p);
    }
    model->add_results(results_to_add);
    model->validate_nodes_and_infer_types();
}

auto remove_encoder_attn_read_value(const std::shared_ptr<ov::Node>& rv_node,
                                    const ov::Output<ov::Node>& kv_out,
                                    const ov::Input<ov::Node>& sdpa_in) {
    // Find Assign node
    OPENVINO_ASSERT(rv_node->outputs().size() == 1);
    auto rv_out = rv_node->outputs()[0];
    ov::NodeVector rv_readers;
    for (const auto& target_in: rv_out.get_target_inputs()) {
        rv_readers.push_back(target_in.get_node()->shared_from_this());
    }
    // Assign and SDPA
    OPENVINO_ASSERT(rv_readers.size() == 2);
    auto assign_node = (strstr(rv_readers[0]->get_type_name(), "Assign") != nullptr) ? rv_readers[0] : rv_readers[1];
    OPENVINO_ASSERT(strstr(assign_node->get_type_name(), "Assign") != nullptr);
    // Redirect KV-cache tensor to SDPA
    sdpa_in.replace_source_output(kv_out);
    return std::make_pair(std::make_shared<ov::op::v0::Result>(kv_out), ov::as_type_ptr<ov::op::v6::Assign>(assign_node));
}

std::string transform_key_value_name(std::string input_string, std::string prefix, std::string enc_or_dec, std::string key_or_value) {
    std::regex pattern("[0-9]+");
    std::smatch match;
    std::regex_search(input_string, match, pattern);

    if (match.empty())
        OPENVINO_THROW("Input string does not match the expected pattern");

    auto number = std::string(match[0]);
    return prefix + "." + number + enc_or_dec + key_or_value;
}

bool is_fake_cvt_to_key_tensor(const ov::Input<ov::Node>& reader) {
    auto fc_reader = reader.get_node()->outputs()[0].get_target_inputs();
    // FakeConvert node has only 1 consumer
    OPENVINO_ASSERT(fc_reader.size() == 1);
    // FakeConvert -> SDPA : 'key' tensor is input with index 1 to SDPA
    return fc_reader.begin()->get_index() == 1;
}


void expose_runtime_states_as_outputs(std::shared_ptr<ov::Model>& model) {
    // Find all ReadValue nodes
    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    // Holds result layers for cross-attn KV-cache tensors
    ov::ResultVector results;
    ov::SinkVector assigns;

    // Go through all ReadValue nodes and remove them
    for (const auto& rv_node : read_value_nodes) {
        OPENVINO_ASSERT(rv_node->inputs().size() == 1);
        OPENVINO_ASSERT(rv_node->outputs().size() == 1);
        auto rv_in  = rv_node->inputs()[0];
        auto x = rv_in.get_source_output();
        auto rv_out = rv_node->outputs()[0];
        // Gather all nodes that read from ReadValue, there must be SDPA and Assign
        auto rv_readers = rv_out.get_target_inputs();
        OPENVINO_ASSERT(rv_readers.size() == 2);
        // Input port for SDPA node
        for (const auto& reader : rv_readers) {
            bool is_fake_cvt = strstr(reader.get_node()->get_type_name(), "FakeConvert") != nullptr;
            if (strstr(reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr || is_fake_cvt) {
                auto sdpa_in = reader;

                // In case there's additional FakeConvert node(fp8): ReadValue -> FakeConvert -> SDPA
                auto is_fc_key_tensor = is_fake_cvt ? is_fake_cvt_to_key_tensor(reader) : false;

                // Remove ReadValue, store new Result and Assign
                auto key_or_value = (sdpa_in.get_index() == 1 || is_fc_key_tensor) ? "key" : "value";
                auto [result, assign] = remove_encoder_attn_read_value(rv_node, rv_in.get_source_output(), sdpa_in);
                auto normalized_name = transform_key_value_name(
                    rv_node->inputs()[0].get_source_output().get_node()->get_friendly_name(),
                    "present",
                    ".encoder.",
                    key_or_value
                );
                set_name(result, normalized_name);
                results.push_back(result);
                assigns.push_back(assign);
            }
        }
    }

    // Add, remove, validate
    model->add_results(results);
    for (const auto& assign : assigns) {
        model->remove_sink(assign);
    }
    model->validate_nodes_and_infer_types();
}

void remove_cache_position(std::shared_ptr<ov::Model>& model) {
    // Build subgraph that will replace cache_pos
    auto input_ids = model->input("input_ids").get_node();
    auto shape_of_node = std::make_shared<ov::op::v3::ShapeOf>(input_ids->outputs()[0]);

    std::vector<int> v_0{0};
    std::vector<int> v_1{1};

    auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    indices->set_friendly_name("indices");
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    axis->set_friendly_name("axis");

    auto gather_node = std::make_shared<ov::op::v8::Gather>(shape_of_node->outputs()[0], indices, axis);
    
    auto cst_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_0);
    auto step     = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, v_1);
    step->set_friendly_name("step");
    auto range_node = std::make_shared<ov::op::v4::Range>(cst_node->outputs()[0], gather_node->outputs()[0], step->outputs()[0], ov::element::i64);
    // Replace cache_position
    auto cache_pos = ov::as_type_ptr<ov::op::v0::Parameter>(model->input("cache_position").get_node()->shared_from_this());
    for (const auto& target_input : cache_pos->outputs()[0].get_target_inputs()) {
        target_input.replace_source_output(range_node->outputs()[0]);
    }

    model->remove_parameter(cache_pos);
    model->validate_nodes_and_infer_types();
}

void normalize_input_key_value_names(std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& in : model->inputs()) {
        if (in.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (in.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name = transform_key_value_name(in.get_any_name(), "past_key_values", ".decoder.", key_or_value);
        set_name(in.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void normalize_output_key_value_names(std::shared_ptr<ov::Model>& model) {
    ov::ResultVector new_results, old_results;
    for (const auto& out : model->outputs()) {
        if (out.get_any_name().find("decoder") == std::string::npos) {
            continue;
        }

        auto key_or_value = (out.get_any_name().find(".key") != std::string::npos) ? "key" : "value";
        auto normalized_name = transform_key_value_name(out.get_any_name(), "present", ".decoder.", key_or_value);
        set_name(out.get_node_shared_ptr(), normalized_name);
    }

    model->validate_nodes_and_infer_types();
}

void expose_runtime_states_as_inputs(std::shared_ptr<ov::Model>& model) {
    // Store Assign nodes to perform remove_sink later on
    ov::SinkVector assigns;
    // To add new Params to the model
    ov::ParameterVector params;

    ov::NodeVector read_value_nodes;
    for (const auto& op : model->get_ops()) {
        if (strstr(op->get_type_name(), "ReadValue") != nullptr) {
            read_value_nodes.push_back(op);
        }
    }

    for (const auto& rv_node : read_value_nodes) {
        auto rv_out = rv_node->outputs()[0];
        auto rv_readers = rv_out.get_target_inputs();
        for (const auto& rv_reader: rv_readers) {
            bool is_fake_cvt = strstr(rv_reader.get_node()->get_type_name(), "FakeConvert") != nullptr;
            if (strstr(rv_reader.get_node()->get_type_name(), "Assign") != nullptr) {
                auto assign_node = ov::as_type_ptr<ov::op::v6::Assign>(rv_reader.get_node()->shared_from_this());
                assigns.push_back(assign_node);
            } else if (strstr(rv_reader.get_node()->get_type_name(), "ScaledDotProductAttention") != nullptr || is_fake_cvt) {
                auto sdpa_in = rv_reader;

                auto shape = rv_node->get_output_partial_shape(0);
                auto new_param = std::make_shared<ov::op::v0::Parameter>(rv_node->get_output_element_type(0), shape);

                // In case there's additional FakeConvert node(fp8): ReadValue -> FakeConvert -> SDPA
                auto is_fc_key_tensor = is_fake_cvt ? is_fake_cvt_to_key_tensor(rv_reader) : false;

                auto key_or_value = (sdpa_in.get_index() == 1 || is_fc_key_tensor) ? "key" : "value";
                auto normalized_name = transform_key_value_name(sdpa_in.get_node()->get_friendly_name(),
                                                                "past_key_values",
                                                                ".encoder.",
                                                                key_or_value);
                set_name(new_param, normalized_name);

                params.push_back(new_param);
                sdpa_in.replace_source_output(new_param->outputs()[0]);
            }
        }
    }

    // Remove sinks and add new params
    model->add_parameters(params);
    for (const auto& assign: assigns) {
        model->remove_sink(assign);
    }
}

std::shared_ptr<ov::Model> prepare_decoder_model(std::shared_ptr<ov::Model>& model) {
    auto decoder_model = model->clone();
    // 2) Remove all non-runtime states from inputs (they empty on first iteration)
    remove_input_kv_tensors(decoder_model);
    // 3) Expose all states that requires initialization on the first run as outputs
    expose_runtime_states_as_outputs(decoder_model);
    // 4) Remove cache_position input
    remove_cache_position(decoder_model);
    // 5) Normalize output names - should be done in stateful_to_stateless_transformation
    normalize_output_key_value_names(decoder_model);

    decoder_model->validate_nodes_and_infer_types();
    return decoder_model;
}

std::shared_ptr<ov::Model> prepare_decoder_with_past_model(std::shared_ptr<ov::Model>& model) {
    auto decoder_with_past_model = model->clone();
    // FIXME: normalization should be done inside stateful_to_stateless_transformation
    normalize_input_key_value_names(decoder_with_past_model);
    normalize_output_key_value_names(decoder_with_past_model);
    expose_runtime_states_as_inputs(decoder_with_past_model);

    decoder_with_past_model->reshape({{"input_ids", ov::PartialShape({-1, 1})}});
    decoder_with_past_model->set_friendly_name("Model6");

    decoder_with_past_model->validate_nodes_and_infer_types();
    return decoder_with_past_model;
}

}  // namespace

namespace ov {
namespace genai {

WhisperPipeline::StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& models_path,
                                                              const ov::AnyMap& properties)
    : WhisperPipelineImplBase{models_path}
    , m_sampler(m_tokenizer) {
    ov::Core core = utils::singleton_core();

    auto encoder_model = core.read_model(models_path / "openvino_encoder_model.xml", {}, properties);
    reshape_to_static_encoder(encoder_model, m_feature_extractor.feature_size);
    auto last_hidden_state_shape = get_encoder_hidden_state_shape(encoder_model);

    std::shared_ptr<ov::Model> decoder_model;
    std::shared_ptr<ov::Model> decoder_with_past_model;

    if (std::filesystem::exists(models_path / "openvino_decoder_with_past_model.xml") ) {
        decoder_model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);
        decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml", {}, properties);
    } else {
        auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);
        ov::pass::StatefulToStateless().run_on_model(model);

        decoder_model = prepare_decoder_model(model);
        decoder_with_past_model = prepare_decoder_with_past_model(model);
    }

    if (!decoder_model || !decoder_with_past_model)
        OPENVINO_THROW("Decoder/decoder_with_past model is not valid !");

    add_attention_mask_input(decoder_model, true /* transform_cross_attn */, last_hidden_state_shape[1].get_length());
    // NB: Note, there is no need to transform cross attention for decoder_with_past_model
    // as it accepts only single token and there can't be any padding.
    // "attention_mask" for "self-attention" is needed to control actual KV-cache size
    add_attention_mask_input(decoder_with_past_model);

    const size_t max_sequence_length = 448;

    reshape_to_static(decoder_model, MAX_PROMPT_LEN, MAX_PROMPT_LEN, last_hidden_state_shape);
    reshape_to_static(decoder_with_past_model, 1, max_sequence_length, last_hidden_state_shape, true /*with_past*/);

    // Replace KV-tensors for the entire cache to tensors only for new token
    decoder_with_past_model = redirect_new_kv_to_output(decoder_with_past_model);

    preprocess_encoder(encoder_model);
    preprocess_decoder(decoder_model);
    preprocess_decoder(decoder_with_past_model);

    ov::CompiledModel compiled_model;
    compiled_model = core.compile_model(encoder_model, "NPU", properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper encoder model");
    m_models.encoder = compiled_model.create_infer_request();

    compiled_model = core.compile_model(decoder_with_past_model, "NPU", properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper decoder with past model");
    m_models.decoder_with_past = compiled_model.create_infer_request();

    compiled_model = core.compile_model(decoder_model, "NPU", properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper decoder model");
    m_models.decoder = compiled_model.create_infer_request();

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1) {
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
    }

    m_sampler.set_seed(m_generation_config.rng_seed);
}

WhisperDecodedResults WhisperPipeline::StaticWhisperPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    OptionalWhisperGenerationConfig generation_config,
    const std::shared_ptr<StreamerBase> streamer_ptr) {
    auto start_time = std::chrono::steady_clock::now();
    WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    OPENVINO_ASSERT(!config.initial_prompt.has_value(), "'initial_prompt' parameter is not supported on NPU device.");
    OPENVINO_ASSERT(!config.hotwords.has_value(), "'hotwords' parameter is not supported on NPU device.");

    size_t max_new_tokens = config.get_max_new_tokens();

    WhisperPerfMetrics perf_metrics;
    perf_metrics.num_input_tokens = 0;
    RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const auto extract_start = std::chrono::steady_clock::now();
    auto input_features = m_feature_extractor.extract(raw_speech_input);
    const auto extract_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start);
    perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(extract_ms);

    const bool is_shortform = input_features.n_frames <= m_feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    std::vector<int64_t> init_ids;
    std::vector<int64_t> output_tokens;
    std::vector<Segment> segments;

    // 0.02 by default
    const float time_precision =
        static_cast<float>(m_feature_extractor.chunk_length) / m_model_config.max_source_positions;
    size_t segment_offset = 0;

    OPENVINO_ASSERT(m_feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
    const float frame_length_in_seconds =
        static_cast<float>(m_feature_extractor.hop_length) / m_feature_extractor.sampling_rate;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {

        const float chunk_time_offset = chunk_offset * frame_length_in_seconds;

        auto input_features_chunk =
            input_features.get_data_with_offset(chunk_offset, m_feature_extractor.nb_max_frames);

        ov::Tensor hidden_state_tensor = encode(m_models.encoder,
                                                input_features_chunk,
                                                m_feature_extractor.feature_size,
                                                m_feature_extractor.nb_max_frames,
                                                raw_metrics);

        // prepare init_ids just once for whole input
        if (init_ids.empty()) {
            init_ids = prepare_init_ids(hidden_state_tensor, m_models.decoder, config, return_timestamps, raw_metrics);
        }

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, init_ids, config, 1);

        auto [results, cancelled] = full_decode(hidden_state_tensor,
                                                config,
                                                m_models,
                                                init_ids,
                                                return_timestamps,
                                                raw_metrics,
                                                streamer_ptr,
                                                m_sampler,
                                                sequence_group);

        std::vector<int64_t> chunk_output_tokens = results.tokens[0];

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  m_feature_extractor.nb_max_frames,
                                                                  time_precision,
                                                                  chunk_time_offset);

            ov::genai::utils::filter_non_segment_metrics(raw_metrics, output_tokens.size(), extracted_segments.segment_ranges);

            segments.insert(segments.end(), extracted_segments.segments.begin(), extracted_segments.segments.end());

            output_tokens.insert(output_tokens.end(),
                                 extracted_segments.non_timestamp_tokens.begin(),
                                 extracted_segments.non_timestamp_tokens.end());

            if (streamer_ptr && streamer_ptr->write(extracted_segments.non_timestamp_tokens) != StreamingStatus::RUNNING) {
                cancelled = true;
                break;
            }

            segment_offset = extracted_segments.last_offset;
        } else {
            output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());
        }

        if (is_shortform) {
            segment_offset = input_features.n_frames;
        }

        if (cancelled) {
            break;
        }
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    auto decode_start_time = std::chrono::steady_clock::now();
    WhisperDecodedResults result{std::vector{m_tokenizer.decode(output_tokens)}, std::vector{1.f}};
    result.perf_metrics = perf_metrics;
    result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));

    // if return_timestamps wasn't enabled by user
    if (!config.return_timestamps) {
        return result;
    }

    if (segments.size()) {
        std::vector<WhisperDecodedResultChunk> chunks;
        chunks.reserve(segments.size());

        for (auto& segment : segments) {
            decode_start_time = std::chrono::steady_clock::now();
            chunks.push_back(
                WhisperDecodedResultChunk{segment.m_start, segment.m_end, m_tokenizer.decode(segment.m_tokens)});
            result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
                    PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
        }

        result.chunks = chunks;
    }

    auto& metrics = result.perf_metrics;
    metrics.load_time = this->m_load_time_ms;
    auto stop_time = std::chrono::steady_clock::now();
    metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    metrics.raw_metrics.tokenization_durations.emplace_back(MicroSeconds(0.0f));
    metrics.evaluate_statistics(start_time);

    return result;
}

}  // namespace genai
}  // namespace ov
