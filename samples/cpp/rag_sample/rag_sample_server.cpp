// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

#include "openvino/genai/llm_pipeline.hpp"
#include "httplib.h"
#include "json.hpp"    

using json = nlohmann::ordered_json;
/*
std::string apply_chat_template(std::string user_prompt){
    std::string system_message = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    std::ostringstream oss_prompt;
    oss_prompt << system_message << "\n<|im_start|>user\n"
                << user_prompt << "<|im_end|>\n<|im_start|>assistant\n";
    std::string processed_prompt = oss_prompt.str();

    return processed_prompt;
}
*/
/*
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    //auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};

    //tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    std::cout << "Init input tensor works\n";
    tokenizer.set_input_tensor(input_tensor);
    std::cout << "Set input tensor works\n";
    tokenizer.infer();
    std::cout << "Tokenizer infer works\n";
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}*/

/*
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    constexpr size_t BATCH_SIZE = 1;
    auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};
    std::cout << "prompt: " << prompt;
    //auto input_tensor = ov::Tensor{ov::element::string, {prompt.length() + 1}, &prompt};
    auto shape = input_tensor.get_shape();
    std::cout << "input tensor shape: [ ";
    for(auto s: shape){
        std::cout << s << " ";
    }
    std::cout << "]\n";
    std::cout << "Init input tensor works\n";
    tokenizer.set_input_tensor(input_tensor);
    std::cout << "Set input tensor works\n";
    tokenizer.infer();
    std::cout << "Tokenizer infer works\n";
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void run_bert_embedding(std::string query){
    std::string bert_path = "bge-large-zh-v1.5/openvino_model.xml";
    std::string bert_tokenizer_path = "bge-large-zh-v1.5/openvino_tokenizer.xml";
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    //Read the tokenizer model information from the file to later get the runtime information
    ov::InferRequest embedding_model = core.compile_model(bert_path, "CPU").create_infer_request();
    std::cout << "Load embedding model successed\n";
    auto tokenizer_model = core.read_model(bert_tokenizer_path);
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    std::cout << "Load tokenizer model successed\n";
    //char *temp_query = new char[query.length() + 1];
    //std::strcpy(temp_query, query.c_str());
    auto [input_ids, attention_mask] = tokenize(tokenizer, query);
    std::cout << "tokenize encode successed\n";
    auto seq_len = input_ids.get_size();

    // Initialize inputs
    embedding_model.set_tensor("input_ids", input_ids);
    embedding_model.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = embedding_model.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
    constexpr size_t BATCH_SIZE = 1;
    embedding_model.infer();
    std::cout << "embedding infer successed\n";
}
*/

#include "openvino/genai/llm_pipeline.hpp"

// (TODO)
// 1. Add init llm pipeline function                                  - handle_init_llm
// 2. Add init bert pipeline function                                 - handle_init_embedings
// 3. Add bert embedding feature extraction function                  - handle_embeddings
// 4. Add server health check function                                - handle_health
// 5. Add llm pipeline unloading function                             - handle_unload_llm
// 6. Add bert pipeline unloading function                            - handle_unload_embeddings
// 7. Add store embeding in vector database function                  - handle_db_store_embeddings
// 8. Add document retrival from database based on embedding function - handle_db_retrival
// 9. Add request cancel function to stop generation                  - handle_request_cancel
// 10. Add reset function to reset rag sever status                   - handle_reset
/*
bool init_llm_pipeline(ov::genai::LLMPipeline& pipe, std::string llm_model_path, std::string device = "CPU"){
    pipe = ov::genai::LLMPipeline(llm_model_path, device);
    //status = true;
    //return std::move(pipe);
    return true;
}
*/
struct Args
{
  std::string llm_model_path = "";
  std::string llm_device = "CPU";
  std::string embedding_model_path = "";
  std::string embedding_device = "CPU";
  int max_new_tokens = 256;
  bool do_sample = false;
  int top_k = 0;
  float top_p = 0.7;
  float temp = 0.95;
  float repeat_penalty = 1.0;
  bool verbose = false;
};

static auto usage(const std::string &prog) -> void
{
  std::cout << "Usage: " << prog << " [options]\n"
            << "\n"
            << "options:\n"
            << "  -h,    --help                        Show this help message and exit\n"
            << "  --llm_model_path         PATH        Directory contains OV LLM model and tokenizers\n"
	    << "  --llm_device             STRING      Specify which device used for llm inference\n"
	    << "  --embedding_model_path   PATH        Directory contains OV Bert model and tokenizers\n"
	    << "  --embedding_device       STRING      Specify which device used for bert inference\n"
            << "  --max_new_tokens         N           Specify max new generated tokens (default: 256)\n"
            << "  --do_sample              BOOL        Specify whether do random sample (default: False)\n"
            << "  --top_k                  N           Specify top-k parameter for sampling (default: 0)\n"
            << "  --top_p                  N           Specify top-p parameter for sampling (default: 0.7)\n"
            << "  --temperature            N           Specify temperature parameter for sampling (default: 0.95)\n"
            << "  --repeat_penalty         N           Specify penalize sequence of tokens (default: 1.0, means no repeat penalty)\n"
            << "  --verbose                BOOL        Display verbose output including config/system/performance info\n";
}

static auto parse_args(const std::vector<std::string> &argv) -> Args
{
  Args args;

  for (size_t i = 1; i < argv.size(); i++)
  {
    const std::string &arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    }
    else if (arg == "--llm_model_path")
    {
      args.llm_model_path = argv[++i];
    }
    else if (arg == "--llm_device")
    {
      args.llm_device = argv[++i];
    }
    else if (arg == "--embedding_model_path")
    {
      args.embedding_model_path = argv[++i];
    }
    else if (arg == "--embedding_device")
    {
      args.embedding_device = argv[++i];
    }
    else if (arg == "--max_new_tokens")
    {
      args.max_new_tokens = std::stoi(argv[++i]);
    }
    else if (arg == "--do_sample")
    {
      args.do_sample = true;
    }
    else if (arg == "--top_k")
    {
      args.top_k = std::stoi(argv[++i]);
    }
    else if (arg == "--top_p")
    {
      args.top_p = std::stof(argv[++i]);
    }
    else if (arg == "--temperature")
    {
      args.temp = std::stof(argv[++i]);
    }
    else if (arg == "--repeat_penalty")
    {
      args.repeat_penalty = std::stof(argv[++i]);
    }
    else if (arg == "--verbose")
    {
      args.verbose = true;
    }
    else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  return args;
}

static auto parse_args(int argc, char **argv) -> Args
{
  std::vector<std::string> argv_vec;
  argv_vec.reserve(argc);

#ifdef _WIN32
  LPWSTR *wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  for (int i = 0; i < argc; i++)
  {
    argv_vec.emplace_back(converter.to_bytes(wargs[i]));
  }

  LocalFree(wargs);

#else
  for (int i = 0; i < argc; i++)
  {
    argv_vec.emplace_back(argv[i]);
  }
#endif

  return parse_args(argv_vec);
}

int main(int argc, char **argv) try {
    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    std::cout << "Init http server\n";

    svr->Options(R"(.*)", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin",      req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    Args args = parse_args(argc, argv);
    ov::genai::LLMPipeline pipe(args.llm_model_path, args.llm_device);

    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = args.max_new_tokens;
    //std::function<bool(std::string)> streamer = [](std::string word) { std::cout << word << std::flush; return false; };
    const auto handle_completions = [&pipe, &config](const httplib::Request & req, httplib::Response & res) {
            res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
            std::cout << "req.body: " << req.body << "\n";
            std::string prompt = req.body;
	    pipe.start_chat();
	    std::string response = pipe.generate(prompt, config);
	    pipe.finish_chat();
	    //std::string processed_prompt = apply_chat_template(prompt);
            //std::cout << "processed_prompt: " << processed_prompt << "\n";
            res.set_content(response, "text/plain");
            //json data = json::parse(req.body);
            //std::string s = data.dump();
            //std::cout << "data: " << s << "\n";
    };
    //svr->Post ("/health",       handle_health);
    svr->Post("/completions",    handle_completions);
    //svr->Post("/embeddings",   handle_embeddings);
    svr->listen("0.0.0.0", 8080);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
