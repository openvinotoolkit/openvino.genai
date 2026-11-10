// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "load_image.hpp"
#include "openvino/genai/bagel/pipeline.hpp"

namespace {

const char* USAGE = "Usage: bagel <MODELS_DIR> <TOKENIZER_DIR> <DEVICE> <vqa|t2i|edit> <PROMPT> [options]\n"
                    "  --image PATH       input image (vqa, edit)\n"
                    "  --steps N          denoising steps (default 50)\n"
                    "  --size N           text-to-image side (default 512)\n"
                    "  --cfg-text F       text CFG scale (default 4.0)\n"
                    "  --cfg-img F        image CFG scale for edit (default 2.0)\n"
                    "  --seed N           noise seed (default 0)\n"
                    "  --max-new-tokens N vqa answer length (default 256)\n"
                    "  --out PATH         output image (default bagel_out.bmp)\n"
                    "  --repeat N         run the task N times (default 1)\n"
                    "  --props \"K=V;...\" extra properties applied on top of the per-model defaults\n";

// Writes a u8 RGB [1, H, W, 3] tensor as a 24-bit BMP
void write_bmp(const std::string& path, const ov::Tensor& image) {
    const auto& shape = image.get_shape();
    const uint32_t h = static_cast<uint32_t>(shape[1]), w = static_cast<uint32_t>(shape[2]);
    const uint32_t row = (w * 3 + 3) & ~3u;
    const uint32_t size = 54 + row * h;
    uint8_t header[54] = {'B', 'M'};
    const auto put32 = [&](size_t off, uint32_t v) {
        for (size_t i = 0; i < 4; ++i) {
            header[off + i] = static_cast<uint8_t>(v >> (8 * i));
        }
    };
    put32(2, size);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h);
    header[26] = 1;
    header[28] = 24;
    put32(34, row * h);

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    const uint8_t* px = image.data<const uint8_t>();
    std::vector<uint8_t> line(row, 0);
    for (uint32_t y = h; y-- > 0;) {
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t* p = px + (static_cast<size_t>(y) * w + x) * 3;
            line[x * 3 + 0] = p[2];
            line[x * 3 + 1] = p[1];
            line[x * 3 + 2] = p[0];
        }
        f.write(reinterpret_cast<const char*>(line.data()), row);
    }
}

void print_report(const ov::genai::BagelPerfMetrics& perf) {
    std::printf("compilation\n  %-44s%11s\n", "model / variant", "ms");
    float total = 0.0f;
    for (const auto& [name, ms] : perf.compile_ms) {
        std::printf("  %-44s%11.1f\n", name.c_str(), ms);
        total += ms;
    }
    std::printf("  %-44s%11.1f\n", "total", total);

    std::vector<std::string> order;
    std::map<std::string, std::vector<float>> by_stage;
    for (const auto& [stage, ms] : perf.infer_ms) {
        if (by_stage.find(stage) == by_stage.end()) {
            order.push_back(stage);
        }
        by_stage[stage].push_back(ms);
    }
    std::printf("inference\n  %-26s%6s%11s%10s%11s%10s%10s\n",
                "stage",
                "calls",
                "total ms",
                "mean ms",
                "median ms",
                "min ms",
                "max ms");
    for (const auto& stage : order) {
        auto v = by_stage[stage];
        std::sort(v.begin(), v.end());
        const float sum = std::accumulate(v.begin(), v.end(), 0.0f);
        const size_t n = v.size();
        const float median = n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0f;
        std::printf("  %-26s%6zu%11.1f%10.1f%11.1f%10.1f%10.1f\n",
                    stage.c_str(),
                    n,
                    sum,
                    sum / n,
                    median,
                    v.front(),
                    v.back());
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 6) {
        std::cerr << USAGE;
        return EXIT_FAILURE;
    }
    const std::string models_dir = argv[1], tokenizer_dir = argv[2], device = argv[3], task = argv[4], prompt = argv[5];
    std::map<std::string, std::string> opts;
    for (int i = 6; i + 1 < argc; i += 2) {
        opts[argv[i]] = argv[i + 1];
    }
    const auto opt = [&](const std::string& key, const std::string& def) {
        const auto it = opts.find(key);
        return it == opts.end() ? def : it->second;
    };
    if (task != "vqa" && task != "t2i" && task != "edit") {
        std::cerr << USAGE;
        return EXIT_FAILURE;
    }

    ov::genai::BagelImageGenerationConfig config;
    config.num_inference_steps = std::stoul(opt("--steps", "50"));
    config.size = std::stoul(opt("--size", "512"));
    config.cfg_text_scale = std::stof(opt("--cfg-text", "4.0"));
    config.cfg_img_scale = std::stof(opt("--cfg-img", "2.0"));
    config.seed = static_cast<uint32_t>(std::stoul(opt("--seed", "0")));
    const size_t max_new_tokens = std::stoul(opt("--max-new-tokens", "256"));
    const std::string out_path = opt("--out", "bagel_out.bmp");
    const int repeat = std::stoi(opt("--repeat", "1"));

    std::optional<ov::Tensor> image;
    if (!opt("--image", "").empty()) {
        image = utils::load_image(opt("--image", ""));
    }
    if (task == "edit" && !image) {
        std::cerr << "edit requires --image\n";
        return EXIT_FAILURE;
    }

    ov::AnyMap props;
    std::stringstream props_list(opt("--props", ""));
    for (std::string kv; std::getline(props_list, kv, ';');) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) {
            std::cerr << "--props expects KEY=VALUE pairs\n";
            return EXIT_FAILURE;
        }
        props[kv.substr(0, eq)] = kv.substr(eq + 1);
    }

    ov::genai::BagelPipeline pipe(models_dir, device, tokenizer_dir, props);

    for (int r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (task == "vqa") {
            const auto answer = pipe.generate_text(prompt, image, max_new_tokens);
            std::cout << "Q: " << prompt << "\nA: " << answer << std::endl;
        } else {
            const auto result =
                task == "t2i" ? pipe.generate_image(prompt, config) : pipe.edit_image(*image, prompt, config);
            write_bmp(out_path, result);
            std::cout << "saved " << out_path << " " << result.get_shape()[2] << "x" << result.get_shape()[1]
                      << std::endl;
        }
        const float wall = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();

        const auto& perf = pipe.get_perf_metrics();
        float variant_compile_ms = 0.0f;
        for (const auto& [name, ms] : perf.compile_ms) {
            if (name.rfind("compile_model", 0) != 0) {
                variant_compile_ms += ms;
            }
        }
        std::printf("\n== run %d: wall %.1f s | variant compile %.1f s | inference %.1f s\n",
                    r,
                    wall,
                    variant_compile_ms / 1e3f,
                    wall - perf.first_call_ms / 1e3f);
        print_report(perf);
        std::printf("\n");
        pipe.reset_perf_metrics();
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}
