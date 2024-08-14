
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <openvino/genai/tokenizer.hpp>
#include <regex>
#include <random>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "sampling.hpp"

#include "clip.h"
#include "minicpmv.h"

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;


namespace {

struct Args {
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.0;
    int output_fixed_len = 0;
};

struct minicpmv_embed {
    float *embed;
    int embed_length;
    std::vector<float> buf;
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

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::genai::Tokenizer tokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = tokenizer.decode(token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = tokenizer.decode(token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

void get_image_embedding(std::vector<std::vector<struct llava_image_embed*>> image_embed_slices, ov::genai::Tokenizer& tokenizer, ov::InferRequest& embedding, ov::Tensor &imgEmbedding) {
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
    embedding_len += image_embed_slices[0][0]->n_image_pos;

    if (image_embed_slices.size() > 1) {
        embedding_len += 1;
        for (size_t i = 1; i < image_embed_slices.size(); ++i) {
            for (size_t j = 0; j < image_embed_slices[i].size(); ++j) {
                embedding_len += 2;
                embedding_len += image_embed_slices[i][j]->n_image_pos;

                if (j == image_embed_slices[i].size() - 1) {
                    embedding_len += 1;
                }
            }
        }

        embedding_len += 1;
    }

    imgEmbedding = ov::Tensor(ov::element::f32, {1, embedding_len, embedding_dim});
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

    //fill image_embed_slices[0][0]
    std::copy(image_embed_slices[0][0]->embed, image_embed_slices[0][0]->embed + image_embed_slices[0][0]->n_image_pos * embedding_dim, imgEmbedData);
    imgEmbedData += image_embed_slices[0][0]->n_image_pos * embedding_dim;

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

                // fill image_embed_slices[i][j]
                std::copy(image_embed_slices[i][j]->embed, image_embed_slices[i][j]->embed + image_embed_slices[i][j]->n_image_pos * embedding_dim, imgEmbedData);
                imgEmbedData += image_embed_slices[i][j]->n_image_pos * embedding_dim;

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
}



ov::Tensor process_prompt(ov::genai::Tokenizer& tokenizer, ov::InferRequest& embedding, std::string prompt, bool isAddUser) {
    std::string user_prompt;
    size_t embedding_dim;
    size_t idx;
    int scale_emb = 12;

    if (isAddUser) {
        user_prompt = "<用户>" + prompt + "<AI>";
    }
    else {
        user_prompt = prompt + "<AI>";
    }

    ov::Tensor input_ids = tokenizer.encode(user_prompt).input_ids;
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
}

class VLMPipeline {
public:
    struct ModelConfig {
        std::filesystem::path model_dir;
        std::string device = "CPU";
        ov::AnyMap device_config;
        // per model devices and configs
        mutable ov::Core core{};
    };

    ov::genai::Tokenizer tokenizer;
    ov::InferRequest ireq_embed;
    ov::InferRequest ireq;
    ov::InferRequest ireq_vision;
    ov::InferRequest ireq_resampler;
    ov::Tensor imgEmbedTensor;
    ov::Shape img_embed_shape;
    int output_fixed_len;
    size_t embed_dim;
    TextStreamer text_streamer;
    std::vector<float> llm_inputs_embeds;
    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, double, double>> perf_records;
    size_t max_lenth = 2048;
    size_t embed_lenth = 0;
    int count = 0;
    double total_time = 0;
    size_t round = 0;
    const size_t BATCH_SIZE = 1;

    explicit VLMPipeline(const ModelConfig& conf) :
        tokenizer{conf.model_dir.string()},
        ireq_embed{conf.core.compile_model(
            conf.model_dir / "openvino_embedding.xml", conf.device, conf.device_config
        ).create_infer_request()},
        ireq{conf.core.compile_model(
            conf.model_dir / "openvino_model.xml", conf.device, conf.device_config
        ).create_infer_request()},
        ireq_vision{conf.core.compile_model(
            // CPU only because of 146022.
            conf.model_dir / "openvino_vision.xml", "CPU", conf.device_config
        ).create_infer_request()},
        ireq_resampler{conf.core.compile_model(
            // CPU randomly fails: 149560.
            conf.model_dir / "openvino_resampler.xml", "GPU"
        ).create_infer_request()},
        text_streamer{tokenizer} {}

    void generate(const std::string& prompt) {
        ov::Tensor llmEmbedTensor;

        //first round
        if (0 == round) {
            //<用户> + image embedding + prompt + <AI> LLM first input
            std::cout << "first round " << std::endl;
            ov::Tensor promtTensor;
            promtTensor = process_prompt(tokenizer, ireq_embed, prompt, false);
            embed_lenth = img_embed_shape[1] + promtTensor.get_shape()[1];

            //memcpy image embedding buf
            if (embed_lenth > max_lenth) {
                llm_inputs_embeds.resize((embed_lenth + 256) * img_embed_shape[2]);
                max_lenth = embed_lenth + 256;
            }

            memcpy(llm_inputs_embeds.data(), imgEmbedTensor.data<float>(), imgEmbedTensor.get_byte_size());
            memcpy(llm_inputs_embeds.data() + img_embed_shape[1] * img_embed_shape[2], promtTensor.data<float>(), promtTensor.get_byte_size());

            llmEmbedTensor = ov::Tensor(ov::element::f32, { 1, embed_lenth, img_embed_shape[2] }, llm_inputs_embeds.data());
        }
        else {
            //<用户> + prompt + <AI>  LLM first input
            //first inference
            std::cout << "round index " << round << std::endl;

            ov::Tensor promtTensor;
            promtTensor = process_prompt(tokenizer, ireq_embed, prompt, true);

            if ((embed_lenth + promtTensor.get_shape()[1]) > max_lenth) {
                llm_inputs_embeds.resize((embed_lenth + 256) * img_embed_shape[2]);
                max_lenth = embed_lenth + 256;
            }

            memcpy(llm_inputs_embeds.data() + embed_lenth * img_embed_shape[2], promtTensor.data<float>(), promtTensor.get_byte_size());
            embed_lenth = embed_lenth + promtTensor.get_shape()[1];

            llmEmbedTensor = ov::Tensor(ov::element::f32, { 1, embed_lenth, img_embed_shape[2] }, llm_inputs_embeds.data());
        }

        auto input_len = llmEmbedTensor.get_shape()[1];

        ireq.set_tensor("inputs_embeds", llmEmbedTensor);
        ireq.get_tensor("attention_mask").set_shape({ llmEmbedTensor.get_shape()[0], llmEmbedTensor.get_shape()[1] });
        std::fill_n(ireq.get_tensor("attention_mask").data<float>(), ireq.get_tensor("attention_mask").get_size(), 1);
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

        constexpr int64_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the detokenizer for now
        while (true) {  //(out_token != SPECIAL_EOS_TOKEN)
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
            //ireq.get_tensor("inputs_embeds").data<int64_t>()[0] = out_token;

            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(ireq.get_tensor("attention_mask").data<float>(), ireq.get_tensor("attention_mask").get_size(), 1);
            ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;

            ireq.start_async();
            ireq.wait();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            total_time += duration_ms;

            text_streamer.put(out_token);
            logits = ireq.get_tensor("logits").data<float>();

            out_token = std::max_element(logits, logits + vocab_size) - logits;

            if (output_fixed_len > 0) {
                if (count >= (output_fixed_len - 1))
                    break;
            }
            else {
                if (out_token == SPECIAL_EOS_TOKEN) {
                    break;
                }
            }
        }

        text_streamer.end();

        if (count > 0) {
            double avg_time = total_time / count;
            std::cout << "Other Avg inference took total " << total_time << " ms token num " << count << " first " << first_time << " ms " << " avg " << total_time / (count) << " ms" << std::endl;
            perf_records.push_back({ input_len, count, first_time, avg_time });
        }

        round++;
    }
};

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE>");
    }
    Args args;
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(argv[2], &image_bytes, &image_bytes_length);
    if (!loaded) {
        std::cout << "failed to load " << argv[2] << std::endl;
        return 0;
    }

    clip_ctx* ctx_clip = new clip_ctx;
    int n_threads = 1;
    for (int i = 0; i < 3; ++i) {
        ctx_clip->image_mean[i] = 0.5;
        ctx_clip->image_std[i] = 0.5;
    }

    std::string device = "CPU";

    size_t group_size = 32;
    ov::AnyMap device_config = {};
    if (device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    if (device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }
    VLMPipeline pipe({argv[1], device, device_config});
    ctx_clip->ireq_vision = pipe.ireq_vision;
    ctx_clip->ireq_resampler = pipe.ireq_resampler;

    double first_time;

    //extract image embedding
    std::vector<std::vector<struct llava_image_embed*>> embeds = llava_image_embed_make_with_bytes_slice(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    //get image embedding
    ov::Tensor imgEmbedTensor;
    get_image_embedding(embeds, pipe.tokenizer, pipe.ireq_embed, imgEmbedTensor);

    ov::Shape img_embed_shape = imgEmbedTensor.get_shape();
    size_t embed_dim = img_embed_shape[2];

    std::cout << "question:\n";
    pipe.imgEmbedTensor = imgEmbedTensor;
    pipe.img_embed_shape = img_embed_shape;
    pipe.output_fixed_len = args.output_fixed_len;
    pipe.embed_dim = embed_dim;
    pipe.llm_inputs_embeds.resize((pipe.max_lenth * embed_dim));
    std::string prompt;
    while (std::getline(std::cin, prompt)) {
        if (prompt == "clear") {
            pipe.round = 0;
            std::cout << "please input prompt:  " << std::endl;
            continue;
        }
        pipe.generate(prompt);
        std::cout << "question:\n";
    }
    llava_image_embed_free_slice(embeds);

    std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
    size_t index = 0;
    for (auto i : pipe.perf_records) {
        std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", " << std::get<3>(i) << std::endl;
        index++;
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (...) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (...) {}
    return EXIT_FAILURE;
}
