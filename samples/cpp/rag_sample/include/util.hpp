#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include "openvino/genai/llm_pipeline.hpp"

class util {
public:
    struct LLMPipelineUtil {
        std::shared_ptr<ov::genai::LLMPipeline> pipe_pointer;
        ov::genai::GenerationConfig config;
    };

    struct Args {
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
    static auto usage(const std::string& prog) -> void {
        std::cout
            << "Usage: " << prog << " [options]\n"
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
            << "  --repeat_penalty         N           Specify penalize sequence of tokens (default: 1.0, means no "
               "repeat "
               "penalty)\n"
            << "  --verbose                BOOL        Display verbose output including config/system/performance "
               "info\n";
    }

    static auto parse_args(const std::vector<std::string>& argv) -> Args {
        Args args;

        for (size_t i = 1; i < argv.size(); i++) {
            const std::string& arg = argv[i];

            if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                exit(EXIT_SUCCESS);
            } else if (arg == "--llm_model_path") {
                args.llm_model_path = argv[++i];
            } else if (arg == "--llm_device") {
                args.llm_device = argv[++i];
            } else if (arg == "--embedding_model_path") {
                args.embedding_model_path = argv[++i];
            } else if (arg == "--embedding_device") {
                args.embedding_device = argv[++i];
            } else if (arg == "--max_new_tokens") {
                args.max_new_tokens = std::stoi(argv[++i]);
            } else if (arg == "--do_sample") {
                args.do_sample = true;
            } else if (arg == "--top_k") {
                args.top_k = std::stoi(argv[++i]);
            } else if (arg == "--top_p") {
                args.top_p = std::stof(argv[++i]);
            } else if (arg == "--temperature") {
                args.temp = std::stof(argv[++i]);
            } else if (arg == "--repeat_penalty") {
                args.repeat_penalty = std::stof(argv[++i]);
            } else if (arg == "--verbose") {
                args.verbose = true;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        }

        return args;
    }

    static auto parse_args(int argc, char** argv) -> Args {
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
};
