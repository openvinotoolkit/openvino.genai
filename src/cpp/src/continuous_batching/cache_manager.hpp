// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <set>
#include <cstring>
#include <thread>
#include <future>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils.hpp"

namespace ov::genai {

/**
 * Check if verbose logging is enabled via OV_GENAI_VERBOSE environment variable.
 * Set OV_GENAI_VERBOSE=1 (or higher) to enable debug logging.
 * Set OV_GENAI_VERBOSE=2 for more detailed timing information.
 * Set OV_GENAI_VERBOSE=3 for full trace-level logging.
 */
inline int get_verbose_level() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("OV_GENAI_VERBOSE");
        level = (env != nullptr) ? std::atoi(env) : 0;
    }
    return level;
}

// Convenience macros for verbose logging
#define OV_GENAI_VERBOSE_ENABLED (ov::genai::get_verbose_level() >= 1)
#define OV_GENAI_VERBOSE_TIMING  (ov::genai::get_verbose_level() >= 2)
#define OV_GENAI_VERBOSE_TRACE   (ov::genai::get_verbose_level() >= 3)

// Logging macros that respect OV_GENAI_VERBOSE level
// Usage: OV_GENAI_LOG(1, "[Pipeline] some message");
//        OV_GENAI_LOG_TIMING("[CacheManager] Load took " << ms << "ms");
#define OV_GENAI_LOG(level, msg) \
    do { if (ov::genai::get_verbose_level() >= level) { std::cout << msg << std::endl; } } while(0)

#define OV_GENAI_LOG_INFO(msg)   OV_GENAI_LOG(1, msg)
#define OV_GENAI_LOG_TIMING(msg) OV_GENAI_LOG(2, msg)
#define OV_GENAI_LOG_TRACE(msg)  OV_GENAI_LOG(3, msg)

// Legacy constant for backward compatibility (now controlled by environment variable)
#define LOG_CACHE_MANAGER_VERBOSE OV_GENAI_VERBOSE_ENABLED

// Set to true to enable USM (Unified Shared Memory) buffer optimization for GPU tensors
constexpr bool USE_USM_BUFFERS = true;

// Set to true to enable parallel tensor loading (experimental)
constexpr bool USE_PARALLEL_LOADING = false;

// Set to true to enable pre-allocated contiguous GPU buffer optimization
constexpr bool USE_CONTIGUOUS_GPU_BUFFER = true;

class CacheManager {
    size_t m_num_decoder_layers = 0;
    std::string m_device;
    size_t m_block_size = 0; // block size is per inference device 
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0, m_block_size_in_bytes = 0;
    ov::InferRequest m_request;
    ov::RemoteContext m_context;
    // simple mutex to allow snapshotting the cache consistently
    mutable std::mutex m_cache_mutex;
    
    // GPU memory usage tracking for performance analysis
    mutable size_t m_total_gpu_bytes_allocated = 0;
    
    // Contiguous GPU buffer optimization
    mutable ov::Tensor m_contiguous_key_buffer;
    mutable ov::Tensor m_contiguous_value_buffer;
    mutable bool m_contiguous_buffers_allocated = false;

    // Helper method to check if USM is supported and create USM buffers
    // Returns the USM type that succeeded, or empty string if none worked
    std::string try_create_usm_tensor(const ov::element::Type& precision, const ov::Shape& shape, ov::Tensor& out_tensor) {
        if (!m_context) {
            return "";
        }
        
        if (LOG_CACHE_MANAGER_VERBOSE) {
            size_t total_elements = ov::shape_size(shape);
            std::cout << "[CacheManager] 🔍 USM DEBUG: Attempting USM optimization with precision=" << precision << ", total_elements=" << total_elements << std::endl;
        }

        // APPROACH 2: Try USM_DEVICE_BUFFER
        try {
            ov::AnyMap usm_params = {
                {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER}
            };
            out_tensor = m_context.create_tensor(precision, shape, usm_params);
            if (LOG_CACHE_MANAGER_VERBOSE) {
                std::cout << "[CacheManager] ✅ USM DEBUG: USM DEVICE BUFFER tensor created successfully!" << std::endl;
            }
            return "USM_DEVICE_BUFFER";
        } catch (const std::exception& ) {} 
        
        // APPROACH 3: Try USM_HOST_BUFFER
        try {
            ov::AnyMap usm_params = {
                {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}
            };
            out_tensor = m_context.create_tensor(precision, shape, usm_params);
            if (LOG_CACHE_MANAGER_VERBOSE) {
                std::cout << "[CacheManager] ✅ USM DEBUG: USM HOST BUFFER tensor created successfully!" << std::endl;
            }
            return "USM_HOST_BUFFER";
        } catch (const std::exception& ) {} 
        
        return "";
    }

    void dump_kv_cache_to_dir_impl(const std::string& dir, size_t num_kv_blocks, size_t used_blocks, bool optimized) const {
        std::lock_guard<std::mutex> lock(m_cache_mutex);
        std::filesystem::create_directories(dir);
        ov::Coordinate start_full{0,0,0,0};

        for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
            ov::Shape key_shape_full = set_kv_blocks(m_key_shapes[layer], num_kv_blocks);
            ov::Shape value_shape_full = set_kv_blocks(m_value_shapes[layer], num_kv_blocks);
            ov::Shape key_shape_used = set_kv_blocks(m_key_shapes[layer], used_blocks);  
            ov::Shape value_shape_used = set_kv_blocks(m_value_shapes[layer], used_blocks);

            auto key_precision = m_key_precisions[layer];
            auto value_precision = m_value_precisions[layer];

            std::string key_bin = dir + "/layer_" + std::to_string(layer) + "_key.bin";
            std::string key_meta = dir + "/layer_" + std::to_string(layer) + "_key.meta";
            std::string val_bin = dir + "/layer_" + std::to_string(layer) + "_value.bin";
            std::string val_meta = dir + "/layer_" + std::to_string(layer) + "_value.meta";

            auto dump_tensor = [&](const ov::Tensor& tensor, const ov::Shape& shape_full, const ov::Shape& shape_used, const ov::element::Type& prec, const std::string& binfile, const std::string& metafile){
                bool is_remote = tensor.is<ov::RemoteTensor>();
                ov::Tensor host_tensor(prec, shape_used);

                if (is_remote) {
                    ov::Coordinate end = shape_used;
                    ov::RemoteTensor src_roi(tensor, start_full, end);
                    src_roi.copy_to(host_tensor);
                } else {
                    size_t used_bytes = host_tensor.get_byte_size();
                    std::memcpy(host_tensor.data(), tensor.data(), used_bytes);
                }

                size_t num_blocks = shape_full.size() > 0 ? shape_full[0] : 0;
                size_t used_blocks_actual = shape_used.size() > 0 ? shape_used[0] : 0;
                size_t num_heads = shape_full.size() > 1 ? shape_full[1] : 0;
                size_t block_size = shape_full.size() > 2 ? shape_full[2] : 0;
                size_t head_dim = shape_full.size() > 3 ? shape_full[3] : 0;
                size_t bytes_per_element = prec.size();
                size_t estimated_bytes = num_blocks * num_heads * block_size * head_dim * bytes_per_element;
                if (prec == ov::element::i4 || prec == ov::element::u4) {
                    estimated_bytes = (num_blocks * num_heads * block_size * head_dim + 1) / 2;
                }
                size_t host_bytes = host_tensor.get_byte_size();

                std::ofstream meta_out(metafile, std::ios::out);
                if (meta_out) {
                    meta_out << "element_type=" << prec << "\n";
                    meta_out << "shape=";
                    for (size_t i = 0; i < shape_full.size(); ++i) {
                        meta_out << shape_full[i];
                        if (i + 1 < shape_full.size()) meta_out << ",";
                    }
                    meta_out << "\n";
                    meta_out << "num_blocks=" << num_blocks << "\n";
                    if (optimized) {
                        meta_out << "used_blocks=" << used_blocks_actual << "\n";
                        meta_out << "optimized=true\n";
                    }
                    meta_out << "num_heads=" << num_heads << "\n";
                    meta_out << "block_size=" << block_size << "\n";
                    meta_out << "head_dim=" << head_dim << "\n";
                    meta_out << "bytes_per_element=" << bytes_per_element << "\n";
                    meta_out << "estimated_bytes=" << estimated_bytes << "\n";
                    meta_out << "actual_bytes=" << host_bytes << "\n";
                    meta_out.close();
                }
                std::ofstream out(binfile, std::ios::out | std::ios::binary);
                if (out) {
                    out.write(reinterpret_cast<const char*>(host_tensor.data()), static_cast<std::streamsize>(host_bytes));
                    out.close();
                    if (LOG_CACHE_MANAGER_VERBOSE) {
                        std::cout << "[CacheManager] dumped " << binfile << " (actual_bytes=" << host_bytes << ")" << std::endl;
                    }
                }
            };

            if (layer < m_key_cache.size()) dump_tensor(m_key_cache[layer], key_shape_full, key_shape_used, key_precision, key_bin, key_meta);
            if (layer < m_value_cache.size()) dump_tensor(m_value_cache[layer], value_shape_full, value_shape_used, value_precision, val_bin, val_meta);
        }
    }

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
        pshape[0] = num_kv_blocks;
        return pshape.get_shape();
    }

    void update_request_tensor(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
    }

public:
    explicit CacheManager(ov::InferRequest request) :
        m_request(request) {
        // extract information about inference device
        ov::CompiledModel compiled_model = request.get_compiled_model();
        std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
        const bool all_gpu_device =
            std::all_of(execution_devices.begin(), execution_devices.end(), [&](const std::string& device) {
                return device.find("GPU") != std::string::npos;
            });
        OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                        "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
        m_device = execution_devices[0];

        if (all_gpu_device) {
            m_context = m_request.get_compiled_model().get_context();
        }
        // extract information about KV cache precisions and shapes
        size_t kv_input_index = 0;
        bool first_key_logged = false;
        for (const auto& input : compiled_model.inputs()) {
            for (auto & name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                ov::PartialShape pshape;

                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_key_shapes.push_back(pshape);
                    m_key_precisions.push_back(cache_precision);
                    if (!first_key_logged) {
                        std::cout << "[CacheManager] Detected KV cache precision from compiled model: " << cache_precision << std::endl;
                        first_key_logged = true;
                    }
                    break;
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_value_shapes.push_back(pshape);
                    m_value_precisions.push_back(cache_precision);
                    ++kv_input_index;
                    break;
                }
            }
        }

        // set block_size depending on device
        const size_t cpu_block_size = 32, gpu_block_size = 16, gpu_block_size_xattn = 256;
        bool has_xattention = false;
        if (all_gpu_device) {
            if (m_value_shapes[0][2].get_length() == gpu_block_size_xattn) {
                has_xattention = true;
            }
            if (utils::env_setup_for_print_debug_info()) {
                if (has_xattention)
                    std::cout << "[XAttention]: ENABLED on GPU device." << std::endl;
                else
                    std::cout << "[XAttention]: DISABLED on GPU device." << std::endl;
            }
        }
        m_block_size = all_gpu_device ? ( has_xattention ? gpu_block_size_xattn : gpu_block_size ) : cpu_block_size;
        m_num_decoder_layers = m_value_precisions.size();
        OPENVINO_ASSERT(m_num_decoder_layers == m_key_precisions.size(), "Invalid case: a different number of K and V caches in a LLM model");
    }

    size_t get_num_decoder_layers() const {
        return m_num_decoder_layers;
    }

    std::string get_device() const {
        return m_device;
    }

    size_t get_block_size() const {
        return m_block_size;
    }

    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_precisions.size());
        return m_key_precisions[decoder_layer_id];
    }

    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_precisions.size());
        return m_value_precisions[decoder_layer_id];
    }

    size_t get_block_size_in_bytes() const {
        return m_block_size_in_bytes;
    }

    size_t sub_byte_data_type_multiplier(const ov::element::Type data_type) const {
        if (data_type == ov::element::i4 || data_type == ov::element::u4)
            return 2;
        return 1;
    }

    void allocate_cache_if_needed(size_t num_kv_blocks) {
        if (m_num_allocated_kv_blocks >= num_kv_blocks) {
            return;
        }
        
        try {
            m_num_allocated_kv_blocks = num_kv_blocks;

            ov::Coordinate start_key{0,0,0,0};
            ov::Coordinate start_value{0,0,0,0};

            if (m_context) {// Allocate KV caches
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                    ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                    ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                    ov::Tensor key_cache = m_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
                    ov::Tensor value_cache = m_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);

                    if (m_key_cache.size() > decoder_layer_id && m_key_cache[decoder_layer_id]) {
                        ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                        ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                        // copy current cache data
                        ov::RemoteTensor dst_key_roi(key_cache, start_key, end_key);
                        ov::RemoteTensor dst_value_roi(value_cache, start_value, end_value);
                        dst_key_roi.copy_from(m_key_cache[decoder_layer_id]);
                        dst_value_roi.copy_from(m_value_cache[decoder_layer_id]);
                    }

                    // set new cache tensors
                    if (m_key_cache.size() > decoder_layer_id) {
                        m_key_cache[decoder_layer_id] = key_cache;
                        m_value_cache[decoder_layer_id] = value_cache;
                    } else {
                        m_key_cache.emplace_back(key_cache);
                        m_value_cache.emplace_back(value_cache);
                    }

                    update_request_tensor(decoder_layer_id);
                }
            } else {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                    ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                    ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                    ov::element::Type key_precision = get_key_cache_precision(decoder_layer_id);
                    ov::element::Type value_precision = get_value_cache_precision(decoder_layer_id);

                    ov::Tensor key_cache(key_precision, key_cache_shape);
                    ov::Tensor value_cache(value_precision, value_cache_shape);

                    auto key_cache_roi_end = static_cast<unsigned char*>(key_cache.data());
                    auto value_cache_roi_end = static_cast<unsigned char*>(value_cache.data());
                    size_t key_roi_size_byte = 0;
                    size_t value_roi_size_byte = 0;

                    if (m_key_cache.size() > decoder_layer_id && m_key_cache[decoder_layer_id]) {
                        ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                        ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();
                        
                        // copy current cache data
                        if (key_precision == ov::element::u4) {
                            size_t key_stride = std::accumulate(end_key.begin(), end_key.end(), 1, std::multiplies<size_t>());
                            size_t key_roi_size_byte = key_stride + (key_stride & 1) / sub_byte_data_type_multiplier(key_precision);
                            std::memcpy(reinterpret_cast<uint8_t*>(key_cache.data()), reinterpret_cast<uint8_t*>(m_key_cache[decoder_layer_id].data()), key_roi_size_byte);
                        } else {
                            key_roi_size_byte = m_key_cache[decoder_layer_id].get_byte_size();
                            ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                            key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + key_roi_size_byte;
                            m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                        }

                        if (value_precision == ov::element::u4) {
                            size_t value_stride = std::accumulate(end_value.begin(), end_value.end(), 1, std::multiplies<size_t>());
                            size_t value_roi_size_byte = value_stride + (value_stride & 1) / sub_byte_data_type_multiplier(value_precision);
                            std::memcpy(reinterpret_cast<uint8_t*>(value_cache.data()), reinterpret_cast<uint8_t*>(m_value_cache[decoder_layer_id].data()), value_roi_size_byte);
                        } else {
                            value_roi_size_byte = m_value_cache[decoder_layer_id].get_byte_size();
                            value_cache_roi_end = static_cast<unsigned char*>(value_cache.data()) + value_roi_size_byte;
                            ov::Tensor dst_value_roi(value_cache, start_value, end_value);
                            m_value_cache[decoder_layer_id].copy_to(dst_value_roi);
                        }
                    }

                    // set new cache tensors
                    if (m_key_cache.size() > decoder_layer_id) {
                        m_key_cache[decoder_layer_id] = key_cache;
                        m_value_cache[decoder_layer_id] = value_cache;
                    } else {
                        m_key_cache.emplace_back(key_cache);
                        m_value_cache.emplace_back(value_cache);
                    }

                    update_request_tensor(decoder_layer_id);
                }
            }
        }
        catch (ov::Exception& e) {
            if (std::string(e.what()).find("bad allocation") != std::string::npos) {
                OPENVINO_THROW("Requested KV-cache size is larger than available memory size on the system.");
            } else {
                throw;
            }
        }
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    size_t get_v_head_size(size_t layer_id) const {
        return m_value_shapes[layer_id][3].get_length();
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                    ov::Shape key_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Shape value_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Coordinate key_src_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_src_end_roi = key_shape;
                    ov::Coordinate key_dst_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_dst_end_roi = key_shape;

                    ov::Coordinate value_src_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_src_end_roi = value_shape;
                    ov::Coordinate value_dst_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_dst_end_roi = value_shape;
                    key_src_end_roi[0] = (key_src_start_roi[0] = src_block_id) + 1;
                    value_src_end_roi[0] = (value_src_start_roi[0] = src_block_id) + 1;
                    key_dst_end_roi[0] = (key_dst_start_roi[0] = dst_block_id) + 1;
                    value_dst_end_roi[0] = (value_dst_start_roi[0] = dst_block_id) + 1;

                    auto copy_one_block = [&](ov::Tensor& dst, const ov::Tensor& src, size_t src_start, size_t dst_start, size_t stride) {
                        const bool is_remote = dst.is<ov::RemoteTensor>() || src.is<ov::RemoteTensor>();
                        if (is_remote) {
                            return;
                        }
                        auto sub_byte_multipyer = sub_byte_data_type_multiplier(dst.get_element_type());
                        OPENVINO_SUPPRESS_DEPRECATED_START
                        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.data()) + src_start * stride;
                        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data()) + dst_start * stride;
                        OPENVINO_SUPPRESS_DEPRECATED_END
                        std::memcpy(dst_ptr, src_ptr, 1 * stride);
                    };

                    const auto& key_cache_prec = m_key_cache[decoder_layer_id].get_element_type();
                    if (key_cache_prec == ov::element::u4 || key_cache_prec == ov::element::i4) {
                        size_t stride = std::accumulate(std::next(key_shape.begin()), key_shape.end(), 1, std::multiplies<size_t>()) / 2;
                        copy_one_block(m_key_cache[decoder_layer_id], m_key_cache[decoder_layer_id], key_src_start_roi[0], key_dst_start_roi[0], stride);
                    } else {
                        ov::Tensor key_src_cache_roi(m_key_cache[decoder_layer_id], key_src_start_roi, key_src_end_roi);
                        ov::Tensor key_dst_cache_roi(m_key_cache[decoder_layer_id], key_dst_start_roi, key_dst_end_roi);
                        key_src_cache_roi.copy_to(key_dst_cache_roi);
                    }

                    const auto& value_cache_prec = m_value_cache[decoder_layer_id].get_element_type();
                    if (value_cache_prec == ov::element::u4 || value_cache_prec == ov::element::i4) {
                        size_t stride = std::accumulate(std::next(value_shape.begin()), value_shape.end(), 1, std::multiplies<size_t>()) / 2;
                        copy_one_block(m_value_cache[decoder_layer_id], m_value_cache[decoder_layer_id], value_src_start_roi[0], value_dst_start_roi[0], stride);
                    } else {
                        ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                        ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);
                        value_src_cache_roi.copy_to(value_dst_cache_roi);
                    }
                }
            }
        }
    }

    // Dump KV cache (all layers key+value) to `dir` as binary files.
    void dump_kv_cache_to_dir(const std::string& dir, size_t num_kv_blocks) const {
        dump_kv_cache_to_dir_impl(dir, num_kv_blocks, num_kv_blocks, false);
    }

    // Enhanced dump that saves only used blocks
    void dump_kv_cache_to_dir_optimized(const std::string& dir, size_t num_kv_blocks, size_t used_blocks) const {
        dump_kv_cache_to_dir_impl(dir, num_kv_blocks, used_blocks, true);
    }
    
    // Dump only sequence state metadata 
    void dump_sequence_state(const std::string& dir, const std::vector<int64_t>& cached_tokens, size_t sequence_length, size_t position_offset, const std::string& model_name = "unknown") const {
        std::string sequence_meta = dir + "/sequence_state.json";
        std::ofstream seq_out(sequence_meta, std::ios::out);
        if (seq_out) {
            seq_out << "{\n";
            seq_out << "  \"dump_version\": 1,\n";
            seq_out << "  \"model_name\": \"" << model_name << "\",\n";
            seq_out << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
            seq_out << "  \"sequence_length\": " << sequence_length << ",\n";
            seq_out << "  \"position_offset\": " << position_offset << ",\n";
            seq_out << "  \"num_cached_tokens\": " << cached_tokens.size() << ",\n";
            seq_out << "  \"cached_tokens\": [";
            for (size_t i = 0; i < cached_tokens.size(); ++i) {
                seq_out << cached_tokens[i];
                if (i + 1 < cached_tokens.size()) seq_out << ", ";
            }
            seq_out << "]\n";
            seq_out << "}\n";
            seq_out.close();
            if (LOG_CACHE_MANAGER_VERBOSE) {
                 std::cout << "[CacheManager] dumped sequence state with " << cached_tokens.size() << " cached tokens" << std::endl;
            }
        }
    }

    void dump_kv_cache_with_sequence_state(const std::string& dir, size_t num_kv_blocks, const std::vector<int64_t>& cached_tokens, size_t sequence_length, size_t position_offset, const std::string& model_name = "unknown") const {
        dump_kv_cache_to_dir(dir, num_kv_blocks);
        dump_sequence_state(dir, cached_tokens, sequence_length, position_offset, model_name);
    }

    // Enhanced load function that restores both KV cache and token sequence state
    struct SequenceState {
        std::vector<int64_t> cached_tokens;
        size_t sequence_length = 0;
        size_t position_offset = 0;
        std::string model_name;
        uint64_t timestamp = 0;
        int dump_version = 0;
    };

    bool load_kv_cache_with_sequence_state(const std::string& dir, size_t expected_num_kv_blocks = 0, 
                                          SequenceState* out_sequence_state = nullptr) {
        // Check if cache is already loaded for this directory to prevent duplicate loading
        static std::string last_loaded_dir = "";
        if (last_loaded_dir == dir && m_num_allocated_kv_blocks > 0) {
            if (OV_GENAI_VERBOSE_ENABLED) {
                std::cout << "[CacheManager] load_kv_cache_with_sequence_state: KV cache already loaded from " << dir 
                          << ", skipping tensor loading and proceeding to sequence state" << std::endl;
            }
        } else {
            // First load the KV cache tensors (existing functionality)
            if (!load_kv_cache_from_dir(dir, expected_num_kv_blocks)) {
                return false;
            }
            last_loaded_dir = dir; // Remember this directory to prevent duplicate loads
        }
        
        // Then load sequence state metadata if available
        std::string sequence_meta = dir + "/sequence_state.json";
        if (!std::filesystem::exists(sequence_meta)) {
            if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_with_sequence_state: no sequence state found, KV cache loaded without token information" << std::endl;
            return true; // Still successful - KV cache is loaded
        }
        
        if (!out_sequence_state) {
            if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_with_sequence_state: sequence state available but no output parameter provided" << std::endl;
            return true;
        }
        
        try {
            std::ifstream seq_in(sequence_meta);
            std::string line, content;
            while (std::getline(seq_in, line)) {
                content += line + "\n";
            }
            seq_in.close();
            
            // Simple JSON parsing for our structured format
            auto extract_number = [&](const std::string& key) -> uint64_t {
                std::string search = "\"" + key + "\": ";
                size_t pos = content.find(search);
                if (pos == std::string::npos) return 0;
                pos += search.length();
                size_t end = content.find(",", pos);
                if (end == std::string::npos) end = content.find("\n", pos);
                return std::stoull(content.substr(pos, end - pos));
            };
            
            auto extract_string = [&](const std::string& key) -> std::string {
                std::string search = "\"" + key + "\": \"";
                size_t pos = content.find(search);
                if (pos == std::string::npos) return "";
                pos += search.length();
                size_t end = content.find("\"", pos);
                return content.substr(pos, end - pos);
            };
            
            out_sequence_state->dump_version = static_cast<int>(extract_number("dump_version"));
            out_sequence_state->model_name = extract_string("model_name");
            out_sequence_state->timestamp = extract_number("timestamp");
            out_sequence_state->sequence_length = extract_number("sequence_length");
            out_sequence_state->position_offset = extract_number("position_offset");
            
            // Parse cached tokens array
            size_t tokens_start = content.find("\"cached_tokens\": [");
            size_t tokens_end = content.find("]", tokens_start);
            if (tokens_start != std::string::npos && tokens_end != std::string::npos) {
                tokens_start += std::string("\"cached_tokens\": [").length();
                std::string tokens_str = content.substr(tokens_start, tokens_end - tokens_start);
                
                std::stringstream ss(tokens_str);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    // Trim whitespace
                    token.erase(0, token.find_first_not_of(" \t\n\r"));
                    token.erase(token.find_last_not_of(" \t\n\r") + 1);
                    if (!token.empty()) {
                        out_sequence_state->cached_tokens.push_back(std::stoll(token));
                    }
                }
            }
            
            if (OV_GENAI_VERBOSE_ENABLED) {
                std::cout << "[CacheManager] loaded sequence state: " << out_sequence_state->cached_tokens.size() 
                          << " cached tokens, sequence_length=" << out_sequence_state->sequence_length 
                          << ", position_offset=" << out_sequence_state->position_offset 
                          << ", model=" << out_sequence_state->model_name << std::endl;
            }
            return true;
            
        } catch (const std::exception &e) {
            if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_with_sequence_state: exception parsing sequence state: " << e.what() << std::endl;
            return true; // Still return true since KV cache was loaded successfully
        }
    }

    bool load_kv_cache_from_dir(const std::string& dir, size_t expected_num_kv_blocks = 0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (OV_GENAI_VERBOSE_ENABLED) {
            std::cout << "[CacheManager] ⏱️ TIMING: load_kv_cache_from_dir STARTING for dir=" << dir << std::endl;
            std::cout << "[CacheManager] load_kv_cache_from_dir: m_num_decoder_layers=" << m_num_decoder_layers << std::endl;
            std::cout.flush();
        }
        
        std::lock_guard<std::mutex> lock(m_cache_mutex);

        try {
            if (!std::filesystem::exists(dir)) {
                if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_from_dir: directory does not exist: " << dir << std::endl;
                return false;
            }
            if (OV_GENAI_VERBOSE_ENABLED) {
                std::cout << "[CacheManager] load_kv_cache_from_dir: directory exists" << std::endl;
                std::cout.flush();
            }

            size_t num_kv_blocks = expected_num_kv_blocks;
            if (num_kv_blocks == 0) {
                std::string meta0 = dir + "/layer_0_key.meta";
                if (!std::filesystem::exists(meta0)) {
                    if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_from_dir: missing meta file " << meta0 << std::endl;
                    return false;
                }
                std::ifstream in(meta0);
                std::string line;
                while (std::getline(in, line)) {
                    if (line.rfind("num_blocks=", 0) == 0) {
                        num_kv_blocks = std::stoull(line.substr(std::string("num_blocks=").size()));
                        break;
                    }
                }
            }
            if (OV_GENAI_VERBOSE_ENABLED) {
                std::cout << "[CacheManager] load_kv_cache_from_dir: num_kv_blocks=" << num_kv_blocks << std::endl;
                std::cout.flush();
            }

            if (num_kv_blocks == 0) {
                if (OV_GENAI_VERBOSE_ENABLED) std::cout << "[CacheManager] load_kv_cache_from_dir: could not determine num_kv_blocks" << std::endl;
                return false;
            }

            if constexpr (USE_CONTIGUOUS_GPU_BUFFER) {
                if (m_context && !m_contiguous_buffers_allocated) {
                    if (LOG_CACHE_MANAGER_VERBOSE) std::cout << "[CacheManager] 🚀 Pre-allocating contiguous GPU buffers..." << std::endl;
                    size_t total_key_bytes = 0, total_value_bytes = 0;
                    for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
                        ov::Shape key_shape = set_kv_blocks(m_key_shapes[layer], num_kv_blocks);
                        ov::Shape val_shape = set_kv_blocks(m_value_shapes[layer], num_kv_blocks);
                        total_key_bytes += key_shape[0] * key_shape[1] * key_shape[2] * key_shape[3] * m_key_precisions[layer].size();
                        total_value_bytes += val_shape[0] * val_shape[1] * val_shape[2] * val_shape[3] * m_value_precisions[layer].size();
                    }
                    try {
                        ov::Shape key_buffer_shape = {total_key_bytes / 4}; 
                        ov::Shape val_buffer_shape = {total_value_bytes / 4};
                        m_contiguous_key_buffer = m_context.create_tensor(ov::element::f32, key_buffer_shape);
                        m_contiguous_value_buffer = m_context.create_tensor(ov::element::f32, val_buffer_shape);
                        m_contiguous_buffers_allocated = true;
                    } catch (const std::exception& e) {
                        if (LOG_CACHE_MANAGER_VERBOSE) std::cout << "[CacheManager] 🚀 Contiguous buffer allocation failed: " << e.what() << std::endl;
                        m_contiguous_buffers_allocated = false;
                    }
                }
            }
            
            allocate_cache_if_needed(num_kv_blocks);
            if (OV_GENAI_VERBOSE_ENABLED) {
                std::cout << "[CacheManager] load_kv_cache_from_dir: allocate_cache_if_needed completed, starting layer loop for " << m_num_decoder_layers << " layers" << std::endl;
                std::cout.flush();
            }

            for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
                if (OV_GENAI_VERBOSE_ENABLED) {
                    std::cout << "[CacheManager] load_kv_cache_from_dir: Starting layer " << layer << std::endl;
                    std::cout.flush();
                }
                auto layer_start_time = std::chrono::high_resolution_clock::now();
                std::string key_meta = dir + "/layer_" + std::to_string(layer) + "_key.meta";
                std::string key_bin = dir + "/layer_" + std::to_string(layer) + "_key.bin";
                std::string val_meta = dir + "/layer_" + std::to_string(layer) + "_value.meta";
                std::string val_bin = dir + "/layer_" + std::to_string(layer) + "_value.bin";

                if (!std::filesystem::exists(key_meta) || !std::filesystem::exists(key_bin) ||
                    !std::filesystem::exists(val_meta) || !std::filesystem::exists(val_bin)) {
                    if (LOG_CACHE_MANAGER_VERBOSE) std::cout << "[CacheManager] load_kv_cache_from_dir: missing files for layer " << layer << std::endl;
                    return false;
                }

                auto parse_meta = [&](const std::string &meta_path) -> std::map<std::string, std::string> {
                    std::map<std::string, std::string> out;
                    std::ifstream in(meta_path);
                    std::string line;
                    while (std::getline(in, line)) {
                        auto pos = line.find('=');
                        if (pos != std::string::npos) {
                            std::string k = line.substr(0, pos);
                            std::string v = line.substr(pos+1);
                            out[k] = v;
                        }
                    }
                    return out;
                };

                auto key_meta_map = parse_meta(key_meta);
                auto val_meta_map = parse_meta(val_meta);

                auto parse_shape = [&](const std::string &s)->ov::Shape{
                    ov::Shape shape;
                    size_t start = 0;
                    while (start < s.size()) {
                        auto comma = s.find(',', start);
                        std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
                        shape.push_back(static_cast<size_t>(std::stoull(token)));
                        if (comma == std::string::npos) break;
                        start = comma + 1;
                    }
                    return shape;
                };

                ov::Shape key_shape = parse_shape(key_meta_map["shape"]);
                ov::Shape val_shape = parse_shape(val_meta_map["shape"]);
                
                // Check for CPU/GPU format mismatch
                ov::Shape expected_key_shape = set_kv_blocks(m_key_shapes[layer], num_kv_blocks);
                if (key_shape.size() == expected_key_shape.size() && key_shape.size() >= 4) {
                    // Compare non-block dimensions (indices 1, 2, 3)
                    bool shape_mismatch = false;
                    for (size_t i = 1; i < 4; ++i) {
                        if (key_shape[i] != expected_key_shape[i]) {
                            shape_mismatch = true;
                            break;
                        }
                    }
                    if (shape_mismatch) {
                        std::cout << "[CacheManager] ⚠️ KV cache format mismatch detected!" << std::endl;
                        std::cout << "[CacheManager]   Saved shape: [" << key_shape[0] << "," << key_shape[1] << "," << key_shape[2] << "," << key_shape[3] << "]" << std::endl;
                        std::cout << "[CacheManager]   Expected shape: [" << expected_key_shape[0] << "," << expected_key_shape[1] << "," << expected_key_shape[2] << "," << expected_key_shape[3] << "]" << std::endl;
                        std::cout << "[CacheManager]   This usually means the KV cache was dumped on a different device (CPU vs GPU)." << std::endl;
                        std::cout << "[CacheManager]   KV cache dump/load must use the same device type." << std::endl;
                        std::cout << "[CacheManager]   Skipping KV cache restoration - will compute from scratch." << std::endl;
                        return false;
                    }
                }

                auto read_binary_to_tensor = [&](const std::string &binpath, const ov::element::Type &prec, const ov::Shape &shape, ov::Tensor &out_host_tensor, const std::map<std::string, std::string> &meta_map)->bool{
                    std::ifstream in(binpath, std::ios::binary | std::ios::ate);
                    if (!in) return false;
                    std::streamsize size = in.tellg();
                    in.seekg(0, std::ios::beg);
                    
                    out_host_tensor = ov::Tensor(prec, shape);
                    size_t expected_size = out_host_tensor.get_byte_size();
                    
                    bool is_optimized = (meta_map.find("optimized") != meta_map.end() && meta_map.at("optimized") == "true");
                    if (is_optimized && meta_map.find("actual_bytes") != meta_map.end()) {
                        expected_size = std::stoull(meta_map.at("actual_bytes"));
                    }
                    
                    if (static_cast<size_t>(size) != expected_size && prec != ov::element::i4 && prec != ov::element::u4) {
                        return false;
                    }
                    
                    size_t bytes_to_read = std::min(static_cast<size_t>(size), out_host_tensor.get_byte_size());
                    in.read(reinterpret_cast<char*>(out_host_tensor.data()), static_cast<std::streamsize>(bytes_to_read));
                    if (bytes_to_read < out_host_tensor.get_byte_size()) {
                        std::memset(reinterpret_cast<char*>(out_host_tensor.data()) + bytes_to_read, 0, out_host_tensor.get_byte_size() - bytes_to_read);
                    }
                    return true;
                };

                ov::Tensor host_key_tensor;
                if (!read_binary_to_tensor(key_bin, m_key_precisions[layer], key_shape, host_key_tensor, key_meta_map)) return false;

                ov::Tensor host_val_tensor;
                if (!read_binary_to_tensor(val_bin, m_value_precisions[layer], val_shape, host_val_tensor, val_meta_map)) return false;

                if (m_context) {
                    if (LOG_CACHE_MANAGER_VERBOSE) {
                        size_t key_b = host_key_tensor.get_byte_size();
                        size_t val_b = host_val_tensor.get_byte_size();
                        std::cout << "[CacheManager] 🔍 Layer " << layer << " GPU allocation: key_bytes=" << key_b << ", val_bytes=" << val_b << ", total=" << (key_b + val_b) << std::endl;
                        std::cout << "[CacheManager] ⚡ Layer " << layer << " attempting USM optimization..." << std::endl;
                    }
                    ov::Shape device_key_shape = set_kv_blocks(m_key_shapes[layer], num_kv_blocks);
                    ov::Shape device_val_shape = set_kv_blocks(m_value_shapes[layer], num_kv_blocks);
                    
                    // Debug: print shape comparison
                    if (OV_GENAI_VERBOSE_ENABLED) {
                        std::cout << "[CacheManager] 📊 Layer " << layer << " shape comparison:" << std::endl;
                        std::cout << "[CacheManager]   host_key_shape: [";
                        for (size_t i = 0; i < key_shape.size(); ++i) { std::cout << key_shape[i] << (i+1 < key_shape.size() ? "," : ""); }
                        std::cout << "], bytes=" << host_key_tensor.get_byte_size() << std::endl;
                        std::cout << "[CacheManager]   device_key_shape: [";
                        for (size_t i = 0; i < device_key_shape.size(); ++i) { std::cout << device_key_shape[i] << (i+1 < device_key_shape.size() ? "," : ""); }
                        std::cout << "], expected_bytes=" << (device_key_shape[0] * device_key_shape[1] * device_key_shape[2] * device_key_shape[3]) << std::endl;
                        std::cout.flush();
                    }
                    
                    ov::Tensor device_key, device_val;
                    bool key_created = false, val_created = false;
                    std::string usm_key_type = "", usm_val_type = "";
                    
                    if constexpr (USE_USM_BUFFERS) {
                        usm_key_type = try_create_usm_tensor(m_key_precisions[layer], device_key_shape, device_key);
                        usm_val_type = try_create_usm_tensor(m_value_precisions[layer], device_val_shape, device_val);
                        if (!usm_key_type.empty() && !usm_val_type.empty()) {
                            key_created = val_created = true;
                            if (LOG_CACHE_MANAGER_VERBOSE) {
                                std::cout << "[CacheManager] ⚡ Layer " << layer << " SUCCESS: " << usm_key_type << " tensors created (GPU-optimized)" << std::endl;
                            }
                        }
                    }
                    
                    if constexpr (USE_CONTIGUOUS_GPU_BUFFER) {
                        if (!key_created && !val_created && m_contiguous_buffers_allocated) {
                            try {
                                device_key = m_context.create_tensor(m_key_precisions[layer], device_key_shape);
                                device_val = m_context.create_tensor(m_value_precisions[layer], device_val_shape);
                                key_created = val_created = true;
                            } catch (const std::exception&) {}
                        }
                    }
                    
                    if (!key_created || !val_created) {
                        device_key = m_context.create_tensor(m_key_precisions[layer], device_key_shape);
                        device_val = m_context.create_tensor(m_value_precisions[layer], device_val_shape);
                    }
                    
                    // For GPU tensors, we need to handle copying carefully.
                    // Create a padded host tensor with device shape and copy the full buffer.
                    auto copy_host_to_device_gpu = [&](ov::Tensor& device_tensor, const ov::Tensor& host_tensor, 
                                                       const ov::Shape& device_shape, const ov::element::Type& precision,
                                                       const char* name) {
                        if (OV_GENAI_VERBOSE_ENABLED) {
                            std::cout << "[CacheManager] 📋 copy_host_to_device_gpu(" << name << "): host_bytes=" << host_tensor.get_byte_size() 
                                      << ", device_bytes=" << device_tensor.get_byte_size() << std::endl;
                            std::cout.flush();
                        }
                        
                        try {
                            // Try ROI-based copy first (works when shapes are compatible)
                            ov::Coordinate start{0,0,0,0};
                            ov::Coordinate end = host_tensor.get_shape();
                            if (OV_GENAI_VERBOSE_ENABLED) {
                                std::cout << "[CacheManager] 📋 Creating RemoteTensor ROI with start={0,0,0,0}, end=host_shape..." << std::endl;
                                std::cout.flush();
                            }
                            
                            ov::RemoteTensor dst_roi(device_tensor, start, end);
                            if (OV_GENAI_VERBOSE_ENABLED) {
                                std::cout << "[CacheManager] 📋 RemoteTensor ROI created, calling copy_from..." << std::endl;
                                std::cout.flush();
                            }
                            
                            dst_roi.copy_from(host_tensor);
                            if (OV_GENAI_VERBOSE_ENABLED) {
                                std::cout << "[CacheManager] 📋 copy_from completed successfully!" << std::endl;
                                std::cout.flush();
                            }
                        } catch (const std::exception& e) {
                            if (OV_GENAI_VERBOSE_ENABLED) {
                                std::cout << "[CacheManager] ⚠️ ROI copy failed: " << e.what() << std::endl;
                                std::cout << "[CacheManager] 📋 Falling back to padded copy..." << std::endl;
                                std::cout.flush();
                            }
                            
                            // Fallback: create padded host tensor with device shape
                            ov::Tensor padded_host(precision, device_shape);
                            std::memset(padded_host.data(), 0, padded_host.get_byte_size());
                            std::memcpy(padded_host.data(), host_tensor.data(), host_tensor.get_byte_size());
                            
                            // Copy full padded tensor to GPU
                            const_cast<ov::RemoteTensor&>(device_tensor.as<ov::RemoteTensor>()).copy_from(padded_host);
                            if (OV_GENAI_VERBOSE_ENABLED) {
                                std::cout << "[CacheManager] 📋 Padded copy completed!" << std::endl;
                                std::cout.flush();
                            }
                        }
                    };
                    
                    copy_host_to_device_gpu(device_key, host_key_tensor, device_key_shape, m_key_precisions[layer], "key");
                    copy_host_to_device_gpu(device_val, host_val_tensor, device_val_shape, m_value_precisions[layer], "val");

                    if (m_key_cache.size() > layer) {
                        m_key_cache[layer] = device_key;
                        m_value_cache[layer] = device_val;
                    } else {
                        m_key_cache.emplace_back(device_key);
                        m_value_cache.emplace_back(device_val);
                    }
                    m_total_gpu_bytes_allocated += device_key.get_byte_size() + device_val.get_byte_size();
                    update_request_tensor(layer);

                    if (LOG_CACHE_MANAGER_VERBOSE) {
                        auto layer_end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(layer_end_time - layer_start_time).count();
                        std::cout << "[CacheManager] ⏱️ Layer " << layer << " timing: TOTAL=" << duration << "ms" << std::endl;
                    }
                } else {
                    // CPU path
                    ov::Shape cpu_key_shape = set_kv_blocks(m_key_shapes[layer], num_kv_blocks);
                    ov::Shape cpu_val_shape = set_kv_blocks(m_value_shapes[layer], num_kv_blocks);
                    ov::Tensor cpu_key(m_key_precisions[layer], cpu_key_shape);
                    ov::Tensor cpu_val(m_value_precisions[layer], cpu_val_shape);

                    std::memset(cpu_key.data(), 0, cpu_key.get_byte_size());
                    std::memset(cpu_val.data(), 0, cpu_val.get_byte_size());

                    ov::Coordinate start{0,0,0,0};
                    ov::Coordinate end_key = host_key_tensor.get_shape();
                    ov::Tensor dst_key_roi(cpu_key, start, end_key);
                    host_key_tensor.copy_to(dst_key_roi);

                    ov::Coordinate end_val = host_val_tensor.get_shape();
                    ov::Tensor dst_val_roi(cpu_val, start, end_val);
                    host_val_tensor.copy_to(dst_val_roi);

                    if (m_key_cache.size() > layer) {
                        m_key_cache[layer] = cpu_key;
                        m_value_cache[layer] = cpu_val;
                    } else {
                        m_key_cache.emplace_back(cpu_key);
                        m_value_cache.emplace_back(cpu_val);
                    }
                    update_request_tensor(layer);
                }
            }
            return true;
        } catch (const std::exception& e) {
            if (LOG_CACHE_MANAGER_VERBOSE) std::cout << "[CacheManager] load_kv_cache_from_dir failed: " << e.what() << std::endl;
            return false;
        }
    }

    size_t get_num_allocated_kv_blocks() const {
        return m_num_allocated_kv_blocks;
    }

    void clear() {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
            m_key_cache[decoder_layer_id] = ov::Tensor();
            m_value_cache[decoder_layer_id] = ov::Tensor();
        }
        m_num_allocated_kv_blocks = 0;
    }
};

}
