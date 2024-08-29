// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vlm_pipeline.hpp"
#include "vlm_sampling.hpp"
#include "clip.hpp"
#include <openvino/openvino.hpp>
#include "../src/text_callback_streamer.hpp"
#include "utils.hpp"
#include <optional>
#include <random>

using namespace ov::genai;

namespace {
constexpr size_t BATCH_SIZE = 1;

struct Args {
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7f;
    float temp = 0.95f;
    float repeat_penalty = 1.0f;
};

int64_t get_out_token_id(const std::vector<int>& input_ids, float* logits, size_t vocab_size, Args args) {
    int64_t out_token;

    // logits pre-process
    if (args.repeat_penalty != 1.f) {
        sampling_repetition_penalty(logits, logits + vocab_size, input_ids, args.repeat_penalty);
    }

    if (args.do_sample)
    {
        if (args.temp > 0) {
            sampling_temperature(logits, logits + vocab_size, args.temp);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < args.top_k && args.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + args.top_k,
                token_scores.data() + token_scores.size());
            token_scores.resize(args.top_k);
        }

        // top_p sampling
        if (0.f < args.top_p && args.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), args.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        out_token = token_scores[dist(gen)].id;
    }
    else {
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    return out_token;
}

ov::Tensor process_prompt(Tokenizer& tokenizer, ov::InferRequest& embedding, const std::string& prompt) {
    std::string user_prompt;
    size_t idx;
    int scale_emb = 12;

    embedding.set_input_tensor(tokenizer.encode(prompt + "<AI>").input_ids);
    embedding.infer();

    const ov::Tensor& embed_output_tensor = embedding.get_output_tensor();

    ov::Shape out_shape = embed_output_tensor.get_shape();
    float* data = embed_output_tensor.data<float>();

    //embedding * scale_emb
    for (idx = 0; idx < embed_output_tensor.get_size(); idx++) {
        data[idx] = data[idx] * scale_emb;
    }
    return embed_output_tensor;
}

ov::Tensor concatenate(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    size_t res_d_2 = first.get_shape().at(2) * 2;
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    OPENVINO_ASSERT(pos.get_shape().size() == 3);
    OPENVINO_ASSERT(pos.get_shape().at(0) == 1);
    size_t d0 = pos.get_shape().at(1);
    size_t d1 = pos.get_shape().at(2);
    size_t d2 = embed_dim / 2;
    std::vector<float> omega(d2);
    for (size_t idx = 0; idx < omega.size(); ++idx) {
        omega.at(idx) = idx / (embed_dim / 2.0f);
        omega.at(idx) = 1.0f / std::pow(10000.0f, omega.at(idx));  // (D/2,)
    }
    const float* const pos_data = pos.data<float>();
    ov::Tensor out(ov::element::f32, {d0, d1, d2});  // (H, W, D/2), outer product
    float* out_data = out.data<float>();
    for (size_t i = 0; i < d0; ++i) {
        for (size_t j = 0; j < d1; ++j) {
            for (size_t k = 0; k < d2; ++k) {
                out_data[i * d1 * d2 + j * d2 + k]
                    = pos_data[i * d1 + j] * omega[k];
            }
        }
    }

    ov::Tensor emb_sin{out.get_element_type(), out.get_shape()};  // (H, W, D/2)
    float* emb_sin_data = emb_sin.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_sin_data, [](float arg) {
        return std::sin(arg);
    });
    ov::Tensor emb_cos{out.get_element_type(), out.get_shape()};  // (H, W, D/2)
    float* emb_cos_data = emb_cos.data<float>();
    std::transform(out_data, out_data + out.get_size(), emb_cos_data, [](float arg) {
        return std::cos(arg);
    });
    return concatenate(emb_sin, emb_cos); // (H, W, D)
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    // use half of dimensions to encode grid_h
    ov::Coordinate begin_h{0, 0, 0};
    ov::Coordinate end_h{grid.get_shape()};
    end_h.at(0) = 1;
    ov::Coordinate begin_w{1, 0, 0};
    ov::Coordinate end_w{grid.get_shape()};
    end_w.at(0) = 2;
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{grid, begin_h, end_h});  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{grid, begin_w, end_w});  // (H, W, D/2)
    return concatenate(emb_h, emb_w);
}

/// image_size: image_size or (image_height, image_width)
/// return:
/// pos_embed: [image_height, image_width, embed_dim]
ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const HeightWidth& image_size) {
    size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
    ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
    float* data = grid.data<float>();
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0);
        data += grid_w_size;
    }
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

ov::Tensor resample(VLMPipeline& pipe, const ov::Tensor& encoded_image, const std::vector<HeightWidth>& target_sizes) {
    size_t bs = encoded_image.get_shape().at(0);
    std::vector<size_t> patch_len{target_sizes.size()};
    std::transform(target_sizes.begin(), target_sizes.end(), patch_len.begin(), [](const HeightWidth& height_width) {
        return height_width.height * height_width.width;
    });
    pipe.adjust_pos_cache(target_sizes);
    size_t max_patch_len = *std::max_element(patch_len.begin(), patch_len.end());
    ov::Tensor key_padding_mask(ov::element::boolean, {bs, max_patch_len});
    bool* mask_data = key_padding_mask.data<bool>();
    size_t embed_len = pipe.pos_embed_cache.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len});  // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* cache_data = pipe.pos_embed_cache.data<float>();
    size_t _d0 = pipe.pos_embed_cache.get_shape().at(0);
    size_t _d1 = pipe.pos_embed_cache.get_shape().at(1);
    for (size_t i = 0; i < bs; ++i) {
        size_t target_h = target_sizes.at(i).height;
        size_t target_w = target_sizes.at(i).width;
        for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
            for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                std::copy_n(
                    cache_data + h_idx * _d1 + w_idx,
                    embed_len,
                    pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                );
            }
        }
        for (size_t flat = target_h * target_w; flat < max_patch_len; ++flat) {
            std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
        }
        std::fill_n(mask_data + i * max_patch_len, patch_len[i], false);
        std::fill_n(mask_data + i * max_patch_len + patch_len[i], max_patch_len - patch_len[i], true);
    }
    pipe.resampler.set_tensor("x", encoded_image);
    pipe.resampler.set_tensor("pos_embed", pos_embed);
    pipe.resampler.set_tensor("key_padding_mask", key_padding_mask);
    pipe.resampler.infer();
    return pipe.resampler.get_output_tensor();
}

ov::Tensor get_image_embedding(const EncodedImage& encoded_image, Tokenizer& tokenizer, ov::InferRequest& embedding, VLMPipeline& pipe) {
    std::string user_prompt;
    size_t embedding_dim;
    size_t embedding_len = 0;
    size_t idx;
    int scale_emb = 12;

    user_prompt = "<用户>";
    const ov::Tensor& input_ids = tokenizer.encode(user_prompt).input_ids;

    auto input_len = input_ids.get_size();
    embedding_len += input_len;

    embedding.set_input_tensor(input_ids);
    embedding.infer();

    const ov::Tensor& embed_output_tensor = embedding.get_output_tensor();

    ov::Shape out_shape = embed_output_tensor.get_shape();
    float* data = embed_output_tensor.data<float>();

    embedding_dim = out_shape[out_shape.size() - 1];

    //input ids embed * config.scale_emb(12)
    for (idx = 0; idx < embed_output_tensor.get_size(); idx++) {
        data[idx] = data[idx] * scale_emb;
    }

    //compute inputs_embedding length
    embedding_len += 2;
    constexpr size_t n_img_pos = 64;  // RESAMPLER query_num minicpmv-2 64, minicpmv-2.5 96
    embedding_len += n_img_pos;

    const ov::Tensor& slices = encoded_image.slices;
    const ov::Shape& slices_shape = slices.get_shape();
    const std::vector<HeightWidth>& sliced_sizes = encoded_image.slices_sizes;
    if (!sliced_sizes.empty()) {
        embedding_len += 1;
        for (size_t i = 0; i < slices_shape.at(0); ++i) {
            for (size_t j = 0; j < slices_shape.at(1); ++j) {
                embedding_len += 2;
                embedding_len += n_img_pos;

                if (j == slices_shape.at(1) - 1) {
                    embedding_len += 1;
                }
            }
        }

        embedding_len += 1;
    }

    ov::Tensor imgEmbedding = ov::Tensor(ov::element::f32, {1, embedding_len, embedding_dim});
    auto imgEmbedData = imgEmbedding.data<float>();

    //copy <用户> embedding info
    memcpy(imgEmbedData, data, embed_output_tensor.get_byte_size());
    imgEmbedData += embed_output_tensor.get_size();

    //get special token embedding info
    user_prompt = "\n<image></image><slice></slice>";
    embedding.set_input_tensor(tokenizer.encode(user_prompt).input_ids);
    embedding.infer();

    const ov::Tensor& embed_spec_tensor = embedding.get_output_tensor();
    data = embed_spec_tensor.data<float>();

    //input ids embed * config.scale_emb(12)
    for (idx = embedding_dim; idx < embed_spec_tensor.get_size(); idx++) {
        data[idx] = data[idx] * scale_emb;
    }


    //fill "<image>" embedding
    std::copy(data + embedding_dim * 2, data + embedding_dim * 3, imgEmbedData);
    imgEmbedData += embedding_dim;

    const ov::Tensor& vision_embded_tensor = resample(pipe, encoded_image.resized_source, {encoded_image.resized_source_size});
    //fill image_embed_slices[0][0]
    std::copy_n(vision_embded_tensor.data<float>(), vision_embded_tensor.get_size(), imgEmbedData);
    imgEmbedData += n_img_pos * embedding_dim;

    //fill "</image>" embedding
    std::copy(data + embedding_dim * 3, data + embedding_dim * 4, imgEmbedData);
    imgEmbedData += embedding_dim;

    if (!sliced_sizes.empty()) {
        //fill "<slice>" embedding
        std::copy(data + embedding_dim * 4, data + embedding_dim * 5, imgEmbedData);
        imgEmbedData += embedding_dim;

        for (size_t i = 0; i < slices_shape.at(0); ++i) {
            for (size_t j = 0; j < slices_shape.at(1); ++j) {
                //fill "<image>" embedding
                std::copy(data + embedding_dim * 2, data + embedding_dim * 3, imgEmbedData);
                imgEmbedData += embedding_dim;

                //Resampler inference with OpenVINO
                size_t d2 = slices_shape.at(2);
                size_t d3 = slices_shape.at(3);
                ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, slices.data<float>() + (i * slices_shape.at(1) + j) * d2 * d3};
                const ov::Tensor& vision_embed_tensor_i_j = resample(pipe, encoded_view, {sliced_sizes.at(i * slices_shape.at(1) + j)});
                // fill image_embed_slices[i][j]
                std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), imgEmbedData);
                imgEmbedData += n_img_pos * embedding_dim;

                //fill "</image>" embedding
                std::copy(data + embedding_dim * 3, data + embedding_dim * 4, imgEmbedData);
                imgEmbedData += embedding_dim;

                if (j == slices_shape.at(1) - 1) {
                    //fill "\n" embedding
                    std::copy(data + embedding_dim, data + embedding_dim * 1, imgEmbedData);
                    imgEmbedData += embedding_dim;
                }
            }
        }
        //fill "</slice>" embedding
        std::copy(data + embedding_dim * 5, data + embedding_dim * 6, imgEmbedData);
        imgEmbedData += embedding_dim;
    }
    return imgEmbedding;
}
}

void VLMPipeline::set_2d_pos_cache(const HeightWidth& max_size) {
    this->pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, max_size);
}

void VLMPipeline::adjust_pos_cache(const std::vector<HeightWidth>& target_sizes) {
    size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const HeightWidth& left, const HeightWidth& right) {
        return left.height < right.height;
    })->height;
    size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const HeightWidth& left, const HeightWidth& right) {
        return left.width < right.width;
    })->width;
    size_t allocated_height, allocated_width;
    if (pos_embed_cache) {
        const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
        allocated_height = allocated_shape.at(0);
        allocated_width = allocated_shape.at(1);
    } else {
        allocated_height = allocated_width = 70;
    }
    if (max_h > allocated_height || max_w > allocated_width) {
        allocated_height = std::max(max_h, allocated_height);
        allocated_width = std::max(max_w, allocated_width);
        this->set_2d_pos_cache({allocated_height, allocated_width});
    }
}

VLMPipeline::VLMPipeline(
    const Tokenizer& tokenizer,
    const VisionEncoder& vision_encoder,
    const ov::InferRequest& resampler,
    const ov::InferRequest& embedding,
    const ov::InferRequest& language_model,
    const VLMConfig& vlm_config
) :
    m_vlm_config{vlm_config},
    tokenizer{tokenizer},
    m_vision_encoder{vision_encoder},
    resampler{resampler},
    ireq_embed{embedding},
    ireq{language_model},
    language_embeddings_history(2048),
    pos_embed_cache{
        get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70})
    } {}

VLMPipeline::VLMPipeline(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config,
    ov::Core core
) : VLMPipeline{
    ov::genai::Tokenizer(model_dir.string(), device_config),
    VisionEncoder(model_dir, device, device_config, core),
    core.compile_model(
        model_dir / "openvino_resampler.xml", device, device_config
    ).create_infer_request(),
    core.compile_model(
        model_dir / "openvino_embedding.xml", device, device_config
    ).create_infer_request(),
    core.compile_model(
        model_dir / "openvino_model.xml", device, device_config
    ).create_infer_request(),
    utils::from_config_json_if_exists<ov::genai::VLMConfig>(
        model_dir, "config.json"
    )
} {}

std::string VLMPipeline::generate(
    const PromptImage& pair,
    const ProcessorConfig& processor_config,
    const VLMConfig& vlm_config,
    const std::function<bool(std::string&&)>& callback
) {
    return generate(
        pair,
        processor_config,
        vlm_config,
        std::make_unique<TextCallbackStreamer>(tokenizer, callback)
    );
}

std::string VLMPipeline::generate(
    const PromptImage& pi,
    const ProcessorConfig& processor_config,
    const VLMConfig& vlm_config,
    const std::shared_ptr<StreamerBase>& streamer
) {
    if (pi.image) {
        EncodedImage embeds = m_vision_encoder.encode(pi.image, processor_config);
        ov::Tensor imgEmbedTensor = get_image_embedding(embeds, tokenizer, this->ireq_embed, *this);

        ov::Shape img_embed_shape = imgEmbedTensor.get_shape();
        OPENVINO_ASSERT(
            vlm_config.hidden_size == img_embed_shape.at(2),
            "Unexpected embedding size");

        this->language_embeddings_history.resize((language_embeddings_history.size() * vlm_config.hidden_size));

        //<用户> + image embedding + prompt + <AI> LLM first input
        ov::Tensor promtTensor = process_prompt(tokenizer, ireq_embed, pi.prompt);
        history_length = img_embed_shape[1] + promtTensor.get_shape()[1];

        //memcpy image embedding buf
        if (history_length > language_embeddings_history.size()) {
            language_embeddings_history.resize((history_length + 256) * vlm_config.hidden_size);
        }

        memcpy(language_embeddings_history.data(), imgEmbedTensor.data<float>(), imgEmbedTensor.get_byte_size());
        memcpy(language_embeddings_history.data() + img_embed_shape[1] * vlm_config.hidden_size, promtTensor.data<float>(), promtTensor.get_byte_size());
    } else {
        //<用户> + prompt + <AI>  LLM first input
        ov::Tensor promtTensor;
        promtTensor = process_prompt(tokenizer, ireq_embed, "<用户>" + pi.prompt);

        if ((history_length + promtTensor.get_shape()[1]) > language_embeddings_history.size()) {
            language_embeddings_history.resize((history_length + 256) * vlm_config.hidden_size);
        }

        memcpy(language_embeddings_history.data() + history_length * vlm_config.hidden_size, promtTensor.data<float>(), promtTensor.get_byte_size());
        history_length = history_length + promtTensor.get_shape()[1];
    }
    ov::Tensor llmEmbedTensor = ov::Tensor(ov::element::f32, {1, history_length, vlm_config.hidden_size}, language_embeddings_history.data());
    auto input_len = llmEmbedTensor.get_shape()[1];

    ireq.set_tensor("inputs_embeds", llmEmbedTensor);
    ireq.get_tensor("attention_mask").set_shape({ llmEmbedTensor.get_shape()[0], llmEmbedTensor.get_shape()[1] });
    std::fill_n(ireq.get_tensor("attention_mask").data<float>(), ireq.get_tensor("attention_mask").get_size(), 1.0f);
    ireq.get_tensor("position_ids").set_shape({ llmEmbedTensor.get_shape()[0], llmEmbedTensor.get_shape()[1] });
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
    ireq.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
    ireq.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    for (auto&& state : ireq.query_state()) {
        state.reset();
    }

    ireq.infer();

    ov::Shape logits_shape = ireq.get_tensor("logits").get_shape();
    auto attention_size = ireq.get_tensor("attention_mask").get_size();

    int64_t sequence_len = ireq.get_tensor("logits").get_shape().at(1) - 1;
    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>() + sequence_len * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

    ireq.get_tensor("inputs_embeds").set_shape({BATCH_SIZE, 1, vlm_config.hidden_size});
    ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

    ireq_embed.get_tensor("inputs_id").set_shape({ 1, 1 });

    int64_t eos_token_id = tokenizer.get_eos_token_id();
    std::vector<int64_t> generated;
    while (true) {  //(out_token != eos_token_id)
        //out_token embedding
        ireq_embed.get_tensor("inputs_id").data<int64_t>()[0] = out_token;
        ireq_embed.start_async();
        ireq_embed.wait();
        const ov::Tensor& embed_prompt_tensor = ireq_embed.get_output_tensor();
        float* embed_data = embed_prompt_tensor.data<float>();

        //input_ids * config.scale_emb
        for (auto idx = 0; idx < embed_prompt_tensor.get_size(); idx++) {
            embed_data[idx] = embed_data[idx] * 12;
        }

        //record answer token info
        if ((history_length + 1) > language_embeddings_history.size()) {
            language_embeddings_history.resize((history_length + 256) * vlm_config.hidden_size);
        }

        memcpy(language_embeddings_history.data() + history_length * vlm_config.hidden_size, embed_prompt_tensor.data<float>(), embed_prompt_tensor.get_byte_size());
        history_length = history_length + 1;

        ireq.set_tensor("inputs_embeds", embed_prompt_tensor);

        ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
        std::fill_n(ireq.get_tensor("attention_mask").data<float>(), ireq.get_tensor("attention_mask").get_size(), 1.0f);
        ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;

        ireq.start_async();
        ireq.wait();

        generated.push_back(out_token);
        if (streamer && streamer->put(out_token)) {
            break;
        }
        logits = ireq.get_tensor("logits").data<float>();

        out_token = std::max_element(logits, logits + vocab_size) - logits;
        if (out_token == eos_token_id) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }
    return tokenizer.decode(generated);
}

std::string VLMPipeline::generate(
    const PromptImage& pair,
    const ov::AnyMap& config_map
) {
    auto iter = config_map.find("vlm_config");
    VLMConfig extracted_config = config_map.end() != iter ?
        iter->second.as<VLMConfig>() : m_vlm_config;
    using utils::read_anymap_param;
    read_anymap_param(config_map, "hidden_size", extracted_config.hidden_size);
    read_anymap_param(config_map, "scale_emb", extracted_config.scale_emb);
    read_anymap_param(config_map, "query_num", extracted_config.query_num);
    return generate(pair, utils::from_any_map(
        config_map, m_vision_encoder.m_processor_config
    ), extracted_config);
}
