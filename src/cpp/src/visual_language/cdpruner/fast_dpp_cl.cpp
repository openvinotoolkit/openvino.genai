// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fast_dpp_cl.hpp"

#ifdef ENABLE_OPENCL_DPP

#    include <algorithm>
#    include <cmath>
#    include <cstring>
#    include <map>
#    include <stdexcept>

#    include "logger.hpp"
#    include "openvino/openvino.hpp"

namespace ov::genai::cdpruner {

/**
 * @brief OpenCL context and state management
 */
struct OpenCLDPP::OpenCLState {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
    cl::Program program;
    std::map<std::string, cl::Kernel> kernels;
    bool initialized = false;

    cl::Kernel get_kernel(const std::string& name) {
        auto it = kernels.find(name);
        if (it != kernels.end()) {
            return it->second;
        }

        // Create kernel if not exists
        cl::Kernel kernel(program, name.c_str());
        kernels[name] = kernel;
        return kernel;
    }
};

OpenCLDPP::OpenCLDPP(const Config& config) : m_config(config), m_state(std::make_unique<OpenCLState>()) {
    m_initialized = initialize_opencl();
}

OpenCLDPP::~OpenCLDPP() {
    cleanup_opencl();
}

std::vector<size_t> OpenCLDPP::select(const ov::Tensor& kernel, size_t num_tokens) {
    OPENVINO_ASSERT(m_initialized, "OpenCL DPP not initialized");

    // Validate input tensor
    auto shape = kernel.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Kernel must be 3D tensor [B, N, N]");

    size_t batch_size = shape[0];
    OPENVINO_ASSERT(batch_size <= 2, "Batch size must be 1 for single batch or 2 for split matrix");
    size_t total_tokens = shape[1];

    OPENVINO_ASSERT(shape[1] == shape[2], "Kernel matrix must be square [B, N, N]");

    OPENVINO_ASSERT(num_tokens / batch_size <= total_tokens,
                    "Cannot select more tokens [",
                    num_tokens / batch_size,
                    "] than available [",
                    total_tokens,
                    "]");

    // Use OpenCL DPP implementation directly with ov::Tensor
    auto opencl_results = run_dpp_split_kernel_impl(kernel, num_tokens);

    return opencl_results;
}

bool OpenCLDPP::initialize_opencl() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            GENAI_WARN("[OpenCLDPP] No OpenCL platforms found");
            return false;
        }

        // Use default device
        m_state->device = cl::Device::getDefault();
        m_state->context = cl::Context(m_state->device);
        m_state->queue = cl::CommandQueue(m_state->context, m_state->device);

        return load_and_compile_kernels();
    } catch (const std::exception& e) {
        GENAI_WARN("[OpenCLDPP] OpenCL initialization failed: %s", e.what());
        return false;
    }
}

bool OpenCLDPP::load_and_compile_kernels() {
    try {
        const char* kernel_source = dpp_kernel_split_cl;
        size_t kernel_length = std::strlen(kernel_source);

        cl::Program::Sources sources;
        sources.push_back({kernel_source, kernel_length});

        m_state->program = cl::Program(m_state->context, sources);
        cl_int result = m_state->program.build({m_state->device});

        if (result != CL_SUCCESS) {
            // Get build log for debugging
            std::string build_log;
            m_state->program.getBuildInfo(m_state->device, CL_PROGRAM_BUILD_LOG, &build_log);
            GENAI_WARN("[OpenCLDPP] Kernel compilation failed with error: %d", result);
            GENAI_WARN("[OpenCLDPP] Build log: %s", build_log.c_str());
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        GENAI_WARN("[OpenCLDPP] Kernel compilation failed: %s", e.what());
        return false;
    }
}

void OpenCLDPP::cleanup_opencl() {
    // Cleanup is handled automatically by cl::* destructors
}

std::vector<size_t> OpenCLDPP::run_dpp_split_kernel_impl(const ov::Tensor& kernel, size_t selected_token_num) {
    float numerical_threshold = m_config.numerical_threshold;

    // Get tensor dimensions from ov::Tensor
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens_num = shape[1];

    std::vector<int> output_ids(selected_token_num, -1);
    auto selected_token_num_per_batch = selected_token_num / batch_size;

    // Prepare initial diagonal values from ov::Tensor
    std::vector<float> vec_di2s(total_tokens_num * batch_size);
    const float* kernel_data = kernel.data<const float>();

    for (size_t b = 0; b < batch_size; b++) {
        size_t offset_base = b * total_tokens_num * total_tokens_num;
        for (size_t i = 0; i < total_tokens_num; i++) {
            // Access diagonal elements from ov::Tensor data
            size_t offset = offset_base + i * total_tokens_num + i;
            vec_di2s[b * total_tokens_num + i] = kernel_data[offset];
        }
    }

    // Create OpenCL buffers
    cl::Buffer buffer_mat(m_state->context, CL_MEM_READ_ONLY, kernel.get_byte_size());
    cl::Buffer buffer_di2s(m_state->context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens_num * batch_size);
    cl::Buffer buffer_cis(m_state->context,
                          CL_MEM_READ_WRITE,
                          sizeof(float) * selected_token_num_per_batch * total_tokens_num * batch_size);
    cl::Buffer buffer_output_ids(m_state->context,
                                 CL_MEM_READ_WRITE,
                                 sizeof(int) * selected_token_num_per_batch * batch_size);

    // Use merged kernel approach (ENABLE_KERNEL_MERGE = 1)
    auto merged_kernel = m_state->get_kernel("dpp_impl");
    cl::NDRange gws = cl::NDRange(batch_size, (total_tokens_num + 15) / 16 * 16, 1);
    cl::NDRange lws = cl::NDRange(1, std::min(total_tokens_num, static_cast<size_t>(16)), 1);

    // Set kernel arguments
    merged_kernel.setArg(0, buffer_mat);
    merged_kernel.setArg(1, static_cast<int>(total_tokens_num));
    // arg 3 will be set in the loop (iteration)
    merged_kernel.setArg(3, buffer_cis);
    merged_kernel.setArg(4, buffer_di2s);
    merged_kernel.setArg(5, numerical_threshold);
    merged_kernel.setArg(6, static_cast<int>(selected_token_num_per_batch));
    merged_kernel.setArg(7, buffer_output_ids);
    merged_kernel.setArg(8, sizeof(float) * lws[1], nullptr);  // local memory for reduction
    merged_kernel.setArg(9, sizeof(int) * lws[1], nullptr);    // local memory for argmax

    // Initialize buffers
    m_state->queue.enqueueWriteBuffer(buffer_di2s,
                                      CL_TRUE,
                                      0,
                                      sizeof(float) * total_tokens_num * batch_size,
                                      vec_di2s.data());
    m_state->queue.enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, kernel.get_byte_size(), kernel_data);

    // Main DPP algorithm loop using OpenCL kernels
    std::vector<cl::Event> eventList;
    for (size_t t = 0; t < selected_token_num_per_batch; ++t) {
        cl::Event event;

        // Set current iteration
        merged_kernel.setArg(2, static_cast<int>(t));

        // Execute the merged kernel (combines argmax, update orthogonal vector, and update marginal gains)
        m_state->queue.enqueueNDRangeKernel(merged_kernel, cl::NullRange, gws, lws, &eventList, &event);
        eventList.push_back(event);
    }

    // Wait for all kernels to complete
    m_state->queue.finish();

    // Read back results
    m_state->queue.enqueueReadBuffer(buffer_output_ids,
                                     CL_TRUE,
                                     0,
                                     sizeof(int) * selected_token_num_per_batch * batch_size,
                                     output_ids.data());

    std::vector<size_t> results;
    for (auto id : output_ids)
        results.push_back(id);

    if (batch_size == 1) {
        std::sort(results.begin(), results.end());
        results.erase(std::unique(results.begin(), results.end()), results.end());
    }

    return results;
}

}  // namespace ov::genai::cdpruner

#endif  // ENABLE_OPENCL_DPP
