// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: genai-gguf-reference model.gguf inputs.json reference.json\n";
        return 2;
    }
    try {
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
        if (!devices[0])
            throw std::runtime_error("llama.cpp CPU backend is required");
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        Model model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model)
            throw std::runtime_error("Failed to load generated GGUF");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        std::ifstream input(argv[2]);
        const auto spec = nlohmann::json::parse(input);
        const int steps = spec.at("max_new_tokens").get<int>();
        nlohmann::json reference = nlohmann::json::array();
        for (const auto& prompt : spec.at("prompts")) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 128;
            cp.n_batch = cp.n_ubatch = 64;
            cp.n_threads = cp.n_threads_batch = 2;
            cp.type_k = cp.type_v = GGML_TYPE_F32;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
            Context ctx(llama_init_from_model(model.get(), cp), llama_free);
            if (!ctx)
                throw std::runtime_error("Failed to create llama.cpp context");
            auto tokens = prompt.get<std::vector<llama_token>>();
            if (tokens.empty() || tokens.size() + steps > cp.n_ctx || steps <= 0)
                throw std::runtime_error("Invalid prompt or generation length");
            for (auto token : tokens) {
                if (token < 0 || token >= vocab)
                    throw std::runtime_error("Input token outside the generated vocabulary");
            }
            std::vector<llama_token> generated;
            std::vector<double> log_probs;
            for (int step = 0; step < steps; ++step) {
                auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
                if (llama_decode(ctx.get(), batch))
                    throw std::runtime_error("llama.cpp decode failed");
                const auto* logits = llama_get_logits_ith(ctx.get(), -1);
                if (!logits || !std::all_of(logits, logits + vocab, [](float v) {
                        return std::isfinite(v);
                    }))
                    throw std::runtime_error("Invalid reference logits");
                const auto next = static_cast<llama_token>(std::max_element(logits, logits + vocab) - logits);
                double sum = 0;
                for (int i = 0; i < vocab; ++i)
                    sum += std::exp(static_cast<double>(logits[i]) - logits[next]);
                generated.push_back(next);
                log_probs.push_back(-std::log(sum));
                tokens = {next};
            }
            reference.push_back({{"input_ids", prompt}, {"generated_ids", generated}, {"log_probs", log_probs}});
        }
        std::ofstream output(argv[3]);
        output << reference.dump(2) << '\n';
        if (!output)
            throw std::runtime_error("Failed to write reference");
        model.reset();
        llama_backend_free();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
