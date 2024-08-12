
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
    std::string ov_model_path = "openvino_model.xml";
    std::string vision_model_path = "minicpm-v-2_vision.xml";
    std::string resam_model_path = "minicpm-v-2_resampler.xml";
    std::string embed_model_path = "minicpm-v-2_embedding.xml";
    std::string token_model_path;
    std::string img_file = "airplane.jpg";
    std::string device = "GPU";
    bool reduce_logits = false;
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

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        minicpm OpenVINO model path (default: openvino_model.xml)\n"
        << "  -vision PATH            minicpmv vision model path (default: minicpm-v-2_vision.xml)\n"
        << "  -resam PATH             minicpmv resampler model path (default: minicpm-v-2_resampler.xml)\n"
        << "  -embed PATH             minicpmv embedding model path (default: minicpm-v-2_embedding.xml)\n"
        << "  -token PATH             Tokenizer model folder\n"
        << "  -d, --device            Device (default: CPU)\n"
        << "  --reduce_logits         Reduce_logits (default: False)\n"
        << "  --do_sample             Search (default: False)\n"
        << "  --top_k N               top-k sampling (default: 0)\n"
        << "  --top_p N               top-p sampling (default: 0.7)\n"
        << "  --temp N                temperature (default: 0.95)\n"
        << "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
        << "  --output_fixed_len N    set output fixed lenth (default: 0, output lenth is determined by the model)\n"
        << "  --image PATH            path to an image file. use with multimodal models. Specify multiple times for batching\n";
}

static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.ov_model_path = argv[++i];
        }
        else if (arg == "-vision") {
            args.vision_model_path = argv[++i];
        }
        else if (arg == "-resam") {
            args.resam_model_path = argv[++i];
        }
        else if (arg == "-embed") {
            args.embed_model_path = argv[++i];
        }
        else if (arg == "-token") {
            args.token_model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        else if (arg == "--reduce_logits") {
            args.reduce_logits = true;
        }
        else if (arg == "--do_sample") {
            args.do_sample = true;
        }
        else if (arg == "--top_k") {
            args.top_k = std::stoi(argv[++i]);
        }
        else if (arg == "--top_p") {
            args.top_p = std::stof(argv[++i]);
        }
        else if (arg == "--temp") {
            args.temp = std::stof(argv[++i]);
        }
        else if (arg == "--repeat_penalty") {
            args.repeat_penalty = std::stof(argv[++i]);
        }
        else if (arg == "--output_fixed_len") {
            args.output_fixed_len = std::stoi(argv[++i]);
        }
        else if (arg == "--image") {
            args.img_file = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

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

/*@brief Insert slice transformation matches following graph, start from logits (Results) to search along root->parent-> grandparent node,
 * then insert slice between Reshape (grandparent node) and Matmul to keep only last dim of matmul first input, first input shape reduced
 * from [1, seq_len, 4096] to [1, 1,4096]. Therefore, after graph transformation, we can reduce matmul computation
 * from [1, seq_len, 4096] * [1, 4096, 151936] = [1, seq_len, 151936] to [1,1,4096]*[4096,151936] = [1,1,151936]
 *
 * Original graph
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 *
 * Modified graph after insert slice:
 *
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *         +----------+                 |
 *         |  Slice   |                 |
 *         +----------+                 |
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
*/

class InsertSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertSlice", "0");
    explicit InsertSlice() {
        auto label = ov::pass::pattern::wrap_type<ov::op::v0::Result>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = std::dynamic_pointer_cast<ov::op::v0::Result>(m.get_match_root());
            if (!root) {
                return false;
            }
            std::string root_name = root->get_friendly_name();
            if (root->get_output_partial_shape(0).size() == 3) {
                std::cout << "Find target root node name: " << root_name << "\n";
                auto parent = root->input_value(0).get_node_shared_ptr();
                std::cout << "Find parent node name: " << parent->get_friendly_name() << "\n";

                //llama2
                auto grand_parent = parent->input_value(0).get_node_shared_ptr();
                std::cout << "Find grandparent node name: " << grand_parent->get_friendly_name() << "\n";

                ov::Output<ov::Node> grand_parent_output = parent->get_input_source_output(0); // parent->get_input_source_output(0);
                std::set<ov::Input<ov::Node>> consumers = grand_parent_output.get_target_inputs();
                auto partial_shape = grand_parent_output.get_partial_shape().get_min_shape();
                int32_t dims = static_cast<int32_t>(partial_shape[2]);
                std::vector<int32_t> start_v = { 0, -1, 0 };
                std::vector<int32_t> stop_v = { 1, -2, dims };
                std::vector<int32_t> step_v = { 1, -1, 1 };

                std::cout << "Original reshape node output shape:" << grand_parent_output.get_partial_shape() << std::endl;
                auto starts = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    start_v);
                auto stop = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    stop_v);
                auto step = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    step_v);
                auto slice = std::make_shared<ov::opset13::Slice>(grand_parent, starts, stop, step); //data, starts, ends, steps
                std::cout << "After insert slice node, output shape" << slice->output(0).get_partial_shape() << std::endl;
                for (auto consumer : consumers) {
                    consumer.replace_source_output(slice->output(0));
                }
                register_new_node(slice);
            }

            return true;
            };
        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertSlice");
        // Register Matcher
        register_matcher(m, callback);
    }
};


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

static bool get_utf8_line(std::string& line) {
#ifdef _WIN32
    std::wstring wline;
    bool ret = !!std::getline(std::wcin, wline);
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    line = converter.to_bytes(wline);
    return ret;
#else
    return !!std::getline(std::cin, line);
#endif
}

}

class VLMPipeline {
public:
    ov::genai::Tokenizer tokenizer;
    ov::InferRequest ireq_embed;
    ov::InferRequest ireq;
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
    Args args = parse_args(argc, argv);
    ov::Core core;

    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    //core.add_extension("D:\\openvino\\runtime\\openvino_tokenizers_windows_2024.4.0.0.dev20240723_x86_64\\runtime\\bin\\intel64\\Release\\openvino_tokenizers.dll");
    auto startTime = Time::now();
    ov::genai::Tokenizer tokenizer{args.token_model_path};
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load minicpm tokenizer took " << duration_ms << " ms" << std::endl;

    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(args.img_file.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        std::cout << "failed to load " << args.img_file << std::endl;
        return 0;
    }

    clip_ctx* ctx_clip = new clip_ctx;
    int n_threads = 1;
    for (int i = 0; i < 3; ++i) {
        ctx_clip->image_mean[i] = 0.5;
        ctx_clip->image_std[i] = 0.5;
    }

    std::string device = args.device;
    size_t convert_model;

    if (args.reduce_logits){
        convert_model = 1;
    }
    else {
        convert_model = 0;
    }

    size_t group_size = 32;
    ov::AnyMap device_config = {};
    if (device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
        //device_config[ov::hint::dynamic_quantization_group_size.name()] = group_size;
    }

    if (device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
        //device_config[ov::hint::dynamic_quantization_group_size.name()] = group_size;
        //std::cout << "set dynamic quantization group size 32" << std::endl;
    }

    double first_time;
    // Read OpenVINO Model
    if (1 == convert_model) {
        startTime = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(args.ov_model_path);
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "Read minicpm Model took " << duration_ms << " ms" << std::endl;

        std::cout << "######## [Model Graph Optimization] Step 2: Insert slice node after reshape to reduce logits operation ########\n";
        ov::pass::Manager manager;
        manager.register_pass<InsertSlice>();
        manager.run_passes(model);

        std::string modifiled_file = std::regex_replace(args.ov_model_path, std::regex("openvino_model"), "modified_openvino_model");
        std::cout << "Save modified model in " << modifiled_file << "\n";
        ov::serialize(model, modifiled_file);

        ov::CompiledModel compilemodel = core.compile_model(modifiled_file, device, device_config);

        return 0;
    }

    ov::CompiledModel vision_compilemodel = core.compile_model(args.vision_model_path, "CPU"); // "AUTO:GPU,CPU");
    ctx_clip->ireq_vision = vision_compilemodel.create_infer_request();

    ov::CompiledModel resam_compilemodel = core.compile_model(args.resam_model_path, device);
    ctx_clip->ireq_resampler = resam_compilemodel.create_infer_request();

    ov::CompiledModel embed_compilemodel = core.compile_model(args.embed_model_path, device, device_config);
    ov::InferRequest ireq_embed = embed_compilemodel.create_infer_request();

    //Compile model
    startTime = Time::now();
    ov::CompiledModel compilemodel = core.compile_model(args.ov_model_path, device, device_config); // "AUTO:GPU,CPU");
    ov::InferRequest ireq = compilemodel.create_infer_request();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile LLM model took " << duration_ms << " ms" << std::endl;
    TextStreamer text_streamer{ std::move(tokenizer) };

    //extract image embedding
    std::vector<std::vector<struct llava_image_embed*>> embeds = llava_image_embed_make_with_bytes_slice(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    //get image embedding
    ov::Tensor imgEmbedTensor;
    get_image_embedding(embeds, tokenizer, ireq_embed, imgEmbedTensor);

    ov::Shape img_embed_shape = imgEmbedTensor.get_shape();
    size_t embed_dim = img_embed_shape[2];

    std::cout << "please input prompt: " << std::endl;
    VLMPipeline pipe{tokenizer, ireq_embed, ireq, imgEmbedTensor, img_embed_shape, args.output_fixed_len, embed_dim, text_streamer};
    pipe.llm_inputs_embeds.resize((pipe.max_lenth * embed_dim));
    while (true) {
        std::string prompt;
        if (!get_utf8_line(prompt) || prompt == "stop") {
            break;
        }
        if (prompt.empty()) {
            std::cout << "prompt empty " << std::endl;
            continue;
        }

        if (prompt == "clear") {
            pipe.round = 0;
            std::cout << "please input prompt:  " << std::endl;
            continue;
        }
        pipe.generate(prompt);
    }

    std::cout << "bye" << std::endl;
    llava_image_embed_free_slice(embeds);

    std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
    size_t index = 0;
    for (auto i : pipe.perf_records) {
        std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", " << std::get<3>(i) << std::endl;
        index++;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
