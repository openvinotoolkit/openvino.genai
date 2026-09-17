// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Offline CPU reference only; never linked into GenAI.
// Usage: oracle language.gguf mmproj.gguf image.png|- prompt.txt history.txt
// Reports the reference greedy choice at each step on the supplied token history.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 4096;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* context = llama_init_from_model(model, cp);
    auto mmp = mtmd_context_params_default();
    mmp.use_gpu = false;
    mmp.warmup = false;
    mmp.n_threads = 4;
    mmp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* multimodal = mtmd_init_from_file(argv[2], model, mmp);
    if (!context || !multimodal)
        return 4;
    std::ifstream text_file(argv[4]);
    std::string prompt((std::istreambuf_iterator<char>(text_file)), {});
    mtmd_input_text text{prompt.data(), prompt.size(), false, true};
    std::vector<const mtmd_bitmap*> images;
    mtmd_helper_bitmap_wrapper bitmap{};
    if (std::string(argv[3]) != "-") {
        bitmap = mtmd_helper_bitmap_init_from_file(multimodal, argv[3], false);
        if (!bitmap.bitmap)
            return 5;
        images.push_back(bitmap.bitmap);
    }
    auto* chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(multimodal, chunks, &text, images.data(), images.size()))
        return 6;
    llama_pos past = 0;
    if (mtmd_helper_eval_chunks(multimodal, context, chunks, 0, 0, 4096, true, &past))
        return 7;
    std::ifstream history_file(argv[5]);
    std::vector<llama_token> history;
    llama_token token;
    while (history_file >> token)
        history.push_back(token);
    const auto vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> choices;
    for (size_t i = 0; i < history.size(); ++i) {
        const float* logits = llama_get_logits_ith(context, -1);
        choices.push_back(std::max_element(logits, logits + vocab_size) - logits);
        if (i + 1 < history.size()) {
            auto batch = llama_batch_get_one(&history[i], 1);
            if (llama_decode(context, batch))
                return 8;
        }
    }
    std::cout << "CHOICES";
    for (auto choice : choices)
        std::cout << ' ' << choice;
    std::cout << '\n';
    mtmd_input_chunks_free(chunks);
    if (bitmap.bitmap)
        mtmd_bitmap_free(bitmap.bitmap);
    mtmd_free(multimodal);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
