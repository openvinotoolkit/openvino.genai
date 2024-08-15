
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <openvino/genai/vlm_pipeline.hpp>
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
#include "openvino/genai/vlm_sampling.hpp"

#include "openvino/genai/clip.hpp"
#include "openvino/genai/vlm_minicpmv.hpp"

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

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
    std::string prompt;
    std::cout << "question:\n";
    if (!std::getline(std::cin, prompt)) {
        throw std::runtime_error("std::cin failed");
    }
    pipe.generate(std::shared_ptr<unsigned char[]>{image_bytes}, image_bytes_length, args.output_fixed_len, prompt);
    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        if (prompt == "clear") {
            pipe.round = 0;
            std::cout << "please input prompt:  " << std::endl;
            continue;
        }
        pipe.generate(prompt);
        std::cout << "question:\n";
    }

    std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
    size_t index = 0;
    for (auto i : pipe.perf_records) {
        std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", " << std::get<3>(i) << std::endl;
        index++;
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
