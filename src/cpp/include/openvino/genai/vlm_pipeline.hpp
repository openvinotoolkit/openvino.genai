// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vlm_minicpmv.hpp"
#include "openvino/genai/vlm_sampling.hpp"
#include "openvino/genai/clip.hpp"
#include <openvino/openvino.hpp>
#include "../src/text_callback_streamer.hpp"
#include <optional>
#include <random>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

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

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

ov::Tensor get_image_embedding(const std::pair<std::vector<std::vector<ov::Tensor>>, size_t>& slices, ov::genai::Tokenizer& tokenizer, ov::InferRequest& embedding, ov::InferRequest& resampler) {
    auto [image_embed_slices, ratio] = slices;
    std::string user_prompt;
    size_t embedding_dim;
    size_t embedding_len = 0;
    size_t idx;
    int scale_emb = 12;

    user_prompt = "<用户>";
    ov::Tensor input_ids = tokenizer.encode(user_prompt).input_ids;

    auto input_len = input_ids.get_size();
    embedding_len += input_len;

    ov::Tensor input_tensor = ov::Tensor(ov::element::i64, { 1, input_ids.get_size() }, input_ids.data());

    embedding.set_input_tensor(input_tensor);
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

    if (image_embed_slices.size() > 1) {
        embedding_len += 1;
        for (size_t i = 1; i < image_embed_slices.size(); ++i) {
            for (size_t j = 0; j < image_embed_slices[i].size(); ++j) {
                embedding_len += 2;
                embedding_len += n_img_pos;

                if (j == image_embed_slices[i].size() - 1) {
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
    input_ids = tokenizer.encode(user_prompt).input_ids;

    input_len = input_ids.get_size();

    input_tensor = ov::Tensor(ov::element::i64, { 1, input_ids.get_size() }, input_ids.data());

    embedding.set_input_tensor(input_tensor);
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

    const ov::Tensor& vision_output_tensor = image_embed_slices[0][0];

    //Resampler inference with OpenVINO
    resampler.set_tensor("x", vision_output_tensor);
    resampler.get_tensor("tgt_size").set_shape({ 1, 1 });
    resampler.get_tensor("tgt_size").data<int64_t>()[0] = ratio;
    resampler.get_tensor("tgt_size").data<int64_t>()[1] = ratio;

    resampler.infer();
    const ov::Tensor& vision_embded_tensor = resampler.get_output_tensor();
    //fill image_embed_slices[0][0]
    std::copy_n(vision_embded_tensor.data<float>(), vision_embded_tensor.get_size(), imgEmbedData);
    imgEmbedData += n_img_pos * embedding_dim;

    //fill "</image>" embedding
    std::copy(data + embedding_dim * 3, data + embedding_dim * 4, imgEmbedData);
    imgEmbedData += embedding_dim;

    if (image_embed_slices.size() > 1) {
        //fill "<slice>" embedding
        std::copy(data + embedding_dim * 4, data + embedding_dim * 5, imgEmbedData);
        imgEmbedData += embedding_dim;

        for (size_t i = 1; i < image_embed_slices.size(); ++i) {
            for (size_t j = 0; j < image_embed_slices[i].size(); ++j) {
                //fill "<image>" embedding
                std::copy(data + embedding_dim * 2, data + embedding_dim * 3, imgEmbedData);
                imgEmbedData += embedding_dim;

                const ov::Tensor& vision_output_tensor_i_j = image_embed_slices[i][j];

                //Resampler inference with OpenVINO
                resampler.set_tensor("x", vision_output_tensor);
                resampler.get_tensor("tgt_size").set_shape({ 1, 1 });
                resampler.get_tensor("tgt_size").data<int64_t>()[0] = ratio;
                resampler.get_tensor("tgt_size").data<int64_t>()[1] = ratio;

                resampler.infer();
                const ov::Tensor& vision_embded_tensor_i_j = resampler.get_output_tensor();
                // fill image_embed_slices[i][j]
                std::copy_n(vision_embded_tensor_i_j.data<float>(), vision_embded_tensor_i_j.get_size(), imgEmbedData);
                imgEmbedData += n_img_pos * embedding_dim;

                //fill "</image>" embedding
                std::copy(data + embedding_dim * 3, data + embedding_dim * 4, imgEmbedData);
                imgEmbedData += embedding_dim;

                if (j == image_embed_slices[i].size() - 1) {
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

ov::Tensor process_prompt(ov::genai::Tokenizer& tokenizer, ov::InferRequest& embedding, std::string prompt) {
    std::string user_prompt;
    size_t idx;
    int scale_emb = 12;

    ov::Tensor input_ids = tokenizer.encode(prompt + "<AI>").input_ids;
    auto input_len = input_ids.get_size();

    ov::Tensor input_tensor = ov::Tensor(ov::element::i64, { 1, input_ids.get_size() }, input_ids.data());

    embedding.set_input_tensor(input_tensor);
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

class VisionEncoder {
public:
    class Config {
        size_t scale_resolution = 448, max_slice_nums = 9, patch_size = 14;
    };
    ov::InferRequest encoder;
    VisionEncoder(const ov::InferRequest& encoder) : encoder{encoder} {}
    explicit VisionEncoder(const std::filesystem::path& model_dir, const std::string& device="CPU", const ov::AnyMap device_config={}, ov::Core core=ov::Core{}) :
        VisionEncoder{core.compile_model(
            // CPU only because of 146022.
            model_dir / "openvino_vision.xml", "CPU", device_config
        ).create_infer_request()} {}
    std::pair<std::vector<std::vector<ov::Tensor>>, size_t> encode(const ov::Tensor image, const Config& config = Config{448, 9, 14}) {
        clip_ctx ctx_clip;
        for (int i = 0; i < 3; ++i) {
            ctx_clip.image_mean[i] = 0.5;
            ctx_clip.image_std[i] = 0.5;
        }
        ctx_clip.ireq_vision = encoder;
        return llava_image_embed_make_with_bytes_slice(&ctx_clip, image);
    }
};

struct PromptImage {
    std::string prompt;
    ov::Tensor image;
};

class VLMPipeline {
public:
    size_t query_num = 64;  // query_num throughput this impl - the number of <unk> to insert into the prompt per image slice.
    float scale_emb = 12.0f;  // multiply embeddings by it. Hardcoded throughout this impl
    bool slice_mode = true;  // Don't resize and slice an input image.
    ov::genai::Tokenizer tokenizer;
    VisionEncoder vision_encoder;
    ov::InferRequest resampler, ireq_embed, ireq;
    ov::Tensor imgEmbedTensor;
    ov::Shape img_embed_shape;
    size_t embed_dim;
    std::vector<float> llm_inputs_embeds;
    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, double, double>> perf_records;
    size_t max_lenth = 2048;
    size_t embed_lenth = 0;
    int count = 0;
    double total_time = 0;
    const size_t BATCH_SIZE = 1;

    VLMPipeline(
        const ov::genai::Tokenizer& tokenizer,
        const VisionEncoder& vision_encoder,
        const ov::InferRequest& resampler,
        const ov::InferRequest& embedding,
        const ov::InferRequest& language_model
    ) :
        tokenizer{tokenizer},
        vision_encoder{vision_encoder},
        resampler{resampler},
        ireq_embed{embedding},
        ireq{language_model} {}

    explicit VLMPipeline(const std::filesystem::path& model_dir, const std::string& device="CPU", const ov::AnyMap device_config={}, ov::Core core=ov::Core{}) :
        VLMPipeline{
            ov::genai::Tokenizer(model_dir.string(), device_config),
            VisionEncoder(model_dir, device, device_config, core),
            core.compile_model(
                // CPU randomly fails: 149560.
                model_dir / "openvino_resampler.xml", "GPU"
            ).create_infer_request(),
            core.compile_model(
                model_dir / "openvino_embedding.xml", device, device_config
            ).create_infer_request(),
            core.compile_model(
                model_dir / "openvino_model.xml", device, device_config
            ).create_infer_request()
        } {}

    void generate(const PromptImage& pi, const std::function<bool(std::string&&)>& callback) {
        generate(pi, std::make_unique<ov::genai::TextCallbackStreamer>(tokenizer, callback));
    }

    void generate(const PromptImage& pi, const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr) {
        if (pi.image) {
            std::pair<std::vector<std::vector<ov::Tensor>>, size_t> embeds = vision_encoder.encode(pi.image);
            ov::Tensor imgEmbedTensor = get_image_embedding(embeds, this->tokenizer, this->ireq_embed, this->resampler);

            ov::Shape img_embed_shape = imgEmbedTensor.get_shape();
            size_t embed_dim = img_embed_shape[2];

            this->imgEmbedTensor = imgEmbedTensor;
            this->img_embed_shape = img_embed_shape;
            this->embed_dim = embed_dim;
            this->llm_inputs_embeds.resize((this->max_lenth * embed_dim));

            //<用户> + image embedding + prompt + <AI> LLM first input
            ov::Tensor promtTensor;
            promtTensor = process_prompt(tokenizer, ireq_embed, pi.prompt);
            embed_lenth = img_embed_shape[1] + promtTensor.get_shape()[1];

            //memcpy image embedding buf
            if (embed_lenth > max_lenth) {
                llm_inputs_embeds.resize((embed_lenth + 256) * img_embed_shape[2]);
                max_lenth = embed_lenth + 256;
            }

            memcpy(llm_inputs_embeds.data(), imgEmbedTensor.data<float>(), imgEmbedTensor.get_byte_size());
            memcpy(llm_inputs_embeds.data() + img_embed_shape[1] * img_embed_shape[2], promtTensor.data<float>(), promtTensor.get_byte_size());
        } else {
            //<用户> + prompt + <AI>  LLM first input
            ov::Tensor promtTensor;
            promtTensor = process_prompt(tokenizer, ireq_embed, "<用户>" + pi.prompt);

            if ((embed_lenth + promtTensor.get_shape()[1]) > max_lenth) {
                llm_inputs_embeds.resize((embed_lenth + 256) * img_embed_shape[2]);
                max_lenth = embed_lenth + 256;
            }

            memcpy(llm_inputs_embeds.data() + embed_lenth * img_embed_shape[2], promtTensor.data<float>(), promtTensor.get_byte_size());
            embed_lenth = embed_lenth + promtTensor.get_shape()[1];
        }
        ov::Tensor llmEmbedTensor = ov::Tensor(ov::element::f32, { 1, embed_lenth, img_embed_shape[2] }, llm_inputs_embeds.data());
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

        auto startTime = Time::now();
        ireq.infer();
        auto duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "First token took " << duration_ms << " ms" << std::endl;
        auto first_time = duration_ms;

        ov::Shape logits_shape = ireq.get_tensor("logits").get_shape();
        auto attention_size = ireq.get_tensor("attention_mask").get_size();

        int64_t sequence_len = ireq.get_tensor("logits").get_shape().at(1) - 1;
        size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
        float* logits = ireq.get_tensor("logits").data<float>() + sequence_len * vocab_size;
        int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

        ireq.get_tensor("inputs_embeds").set_shape({ BATCH_SIZE, 1,  embed_dim });
        ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

        ireq_embed.get_tensor("inputs_id").set_shape({ 1, 1 });

        int64_t eos_token_id = tokenizer.get_eos_token_id();
        while (true) {  //(out_token != eos_token_id)
            startTime = Time::now();

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
            if ((embed_lenth + 1) > max_lenth) {
                llm_inputs_embeds.resize((embed_lenth + 256) * img_embed_shape[2]);
                max_lenth = embed_lenth + 256;
            }

            memcpy(llm_inputs_embeds.data() + embed_lenth * img_embed_shape[2], embed_prompt_tensor.data<float>(), embed_prompt_tensor.get_byte_size());
            embed_lenth = embed_lenth + 1;

            ireq.set_tensor("inputs_embeds", embed_prompt_tensor);

            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(ireq.get_tensor("attention_mask").data<float>(), ireq.get_tensor("attention_mask").get_size(), 1.0f);
            ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;

            ireq.start_async();
            ireq.wait();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            total_time += duration_ms;

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

        if (count > 0) {
            double avg_time = total_time / count;
            std::cout << "Other Avg inference took total " << total_time << " ms token num " << count << " first " << first_time << " ms " << " avg " << total_time / (count) << " ms" << std::endl;
            perf_records.push_back({ input_len, count, first_time, avg_time });
        }
    }

    void start_chat() {}
    void finish_chat() {}
};

ov::Tensor read_jpg(const char* path) {
    auto file = fopen(path, "rb");
    OPENVINO_ASSERT(nullptr != file, "Can't read file");
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    ov::Tensor image{ov::element::u8, {fileSize}};

    errno = 0;
    size_t ret = fread(image.data(), 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        std::cerr << "Read error\n";
    }
    if (ret != (size_t)fileSize) {
        std::cerr << "unexpectedly reached end of file\n";
    }
    fclose(file); // Close the file
    clip_image_u8 img;
    OPENVINO_ASSERT(clip_image_load_from_bytes(image.data<uint8_t>(), fileSize, &img), "Can't load image from bytes, is it a valid image?");
    ov::Tensor tensor{ov::element::u8, {1, size_t(img.ny), size_t(img.nx), 3}};
    std::copy_n(img.buf.begin(), img.buf.size(), tensor.data<uint8_t>());
    return tensor;
}
