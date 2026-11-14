// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <vector>
#include <list>
#include <string>
#include <cstdint>
#include <cstring>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
#include "continuous_batching/cache/kv_chunk_table.hpp"
#include "logger.hpp"
#include "utils.hpp"
namespace ov::genai {

class KVCacheManager : public ICacheManager {
    // Chunk sizes are rounded up to a multiple of this many blocks. Internal: it only sets the resolution of the
    // chunk table's directory (one 4-byte entry per granule), not what users can ask for.
    static constexpr size_t kDefaultGranuleBlocks = 4;
    // A growth step adds at least 1/kGrowthFloorDivisor of the blocks already allocated, which bounds the chunk count
    // to O(log capacity) at the cost of at most ~6% over-allocation.
    static constexpr size_t kGrowthFloorDivisor = 16;
    // Capacity of the chunk table, fixed for the lifetime of the manager so the table tensor is sized and bound once:
    // at most kMaxChunks chunks and kMaxGranules granules in total. The growth floor above keeps the chunk count
    // logarithmic in the capacity, and 16384 granules of 4 blocks address 65536 blocks with a 64 KiB directory.
    static constexpr size_t kMaxChunks = 128;
    static constexpr size_t kMaxGranules = size_t{1} << 14;

    size_t m_num_layers = 0;
    std::string m_device;
    size_t m_block_size = 0; // block size is per inference device 
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0, m_block_size_in_bytes = 0;
    ov::InferRequest m_request;
    ov::RemoteContext m_context;

    // When enabled, key/value cache storage is grown in chunks (each chunk a separate USM device allocation) instead of
    // one growing tensor per layer. `key_cache.N` / `value_cache.N` are bound once to the chunk-0 tensors (they carry
    // the dtype and block shape), and the global `kv_chunk_table` model input, updated in place, tells the kernels where
    // every chunk lives (see kv_chunk_table.hpp). Mutually exclusive with cache eviction/rotation (see
    // SchedulerConfig::validate()) since that addresses cache blocks via the single-tensor accessors below, which have
    // not been made chunk-aware. Prefix caching is supported.
    bool m_chunked_kv_cache = false;
    // m_granule_shift: chunk sizes are rounded up to 2^m_granule_shift blocks.
    uint32_t m_granule_shift = 0;
    std::vector<size_t> m_chunk_blocks;       // blocks in each chunk, in allocation order
    std::vector<size_t> m_chunk_first_block;  // first block id of each chunk (prefix sums of m_chunk_blocks)
    std::vector<std::vector<ov::Tensor>> m_key_chunks, m_value_chunks;  // [layer][chunk_id]
    kv_chunk_table::GlobalChunkTable m_chunk_table;

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
        pshape[0] = num_kv_blocks;
        return pshape.get_shape();
    }

    // Memory a number of blocks takes across all layers (K and V), for log messages.
    double blocks_to_mib(size_t blocks) const {
        return static_cast<double>(blocks) * static_cast<double>(m_block_size_in_bytes) / (1024.0 * 1024.0);
    }

    void update_request_tensor(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
    }

    // --- chunked KV cache helpers ---

    size_t chunk_id_of(size_t physical_block_id) const {
        // Last chunk whose first block is <= physical_block_id.
        const auto it = std::upper_bound(m_chunk_first_block.begin(), m_chunk_first_block.end(), physical_block_id);
        OPENVINO_ASSERT(it != m_chunk_first_block.begin(), "KV cache block ", physical_block_id, " is not in any chunk");
        return static_cast<size_t>(it - m_chunk_first_block.begin()) - 1;
    }

    size_t local_block_id_of(size_t physical_block_id) const {
        return physical_block_id - m_chunk_first_block[chunk_id_of(physical_block_id)];
    }

    // Calls fn(tensor, local_block_id) for the key then the value tensor of every layer holding the block.
    template <class Fn>
    void for_each_block_segment(size_t block_id, Fn&& fn) const {
        OPENVINO_ASSERT(block_id < m_num_allocated_kv_blocks, "Invalid KV cache block ID ", block_id);
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            if (m_chunked_kv_cache) {
                const size_t chunk_id = chunk_id_of(block_id), local_id = local_block_id_of(block_id);
                fn(m_key_chunks[layer][chunk_id], local_id);
                fn(m_value_chunks[layer][chunk_id], local_id);
            } else {
                fn(m_key_cache[layer], block_id);
                fn(m_value_cache[layer], block_id);
            }
        }
    }

    // Copies one block of `tensor` to (to_host) or from `host`; returns the block's byte size.
    static size_t copy_block_segment(const ov::Tensor& tensor, size_t local_id, uint8_t* host, bool to_host) {
        ov::Shape block_shape = tensor.get_shape();
        const size_t block_bytes = tensor.get_byte_size() / block_shape[0];
        if (tensor.is<ov::RemoteTensor>()) {
            ov::Coordinate begin(block_shape.size(), 0);
            ov::Coordinate end(block_shape.begin(), block_shape.end());
            begin[0] = local_id;
            end[0] = local_id + 1;
            ov::RemoteTensor roi(tensor.as<ov::RemoteTensor>(), begin, end);
            block_shape[0] = 1;
            ov::Tensor host_view(tensor.get_element_type(), block_shape, host);
            if (to_host) {
                roi.copy_to(host_view);
            } else {
                roi.copy_from(host_view);
            }
        } else {
            auto* block = static_cast<uint8_t*>(const_cast<ov::Tensor&>(tensor).data()) + local_id * block_bytes;
            if (to_host) {
                std::memcpy(host, block, block_bytes);
            } else {
                std::memcpy(block, host, block_bytes);
            }
        }
        return block_bytes;
    }

    static uint64_t usm_pointer_of(const ov::Tensor& chunk_tensor) {
        // Avoids depending on openvino/runtime/intel_gpu/ocl/ocl.hpp's USMTensor helper, which pulls in
        // the OpenCL C++ headers (CL/cl2.hpp) that aren't on this project's include path; mem_handle is
        // just a property name (see remote_properties.hpp), readable off any ov::RemoteTensor directly.
        auto remote_tensor = const_cast<ov::Tensor&>(chunk_tensor).as<ov::RemoteTensor>();
        auto params = remote_tensor.get_params();
        void* ptr = params.at(ov::intel_gpu::mem_handle.name()).as<ov::intel_gpu::gpu_handle_param>();
        OPENVINO_ASSERT(ptr != nullptr, "Chunked KV cache chunk did not provide a valid USM device pointer");
        return reinterpret_cast<uint64_t>(ptr);
    }

    void bind_chunk0_cache_tensors(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_chunks[decoder_layer_id][0]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_chunks[decoder_layer_id][0]);
    }

    // Binds the smallest legal (1-block) allocation to key_cache.N / value_cache.N so the request stops referencing
    // the chunk-0 tensors (see clear()); nothing reads them until the next allocation rebinds chunk 0.
    void bind_placeholder_cache_tensors(size_t decoder_layer_id) {
        ov::Shape key_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], 1);
        ov::Shape value_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], 1);
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id),
                             m_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_shape));
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id),
                             m_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_shape));
    }

    // Only ever appends one brand-new chunk; existing chunks (and the KV data already written into them) are never
    // reallocated, copied, or moved, and nothing is rebound: the table is updated in place. This is the actual point of
    // chunking: growth no longer costs an O(existing_cache_size) copy.
    void allocate_chunked_cache_if_needed(size_t num_kv_blocks) {
        const size_t granule_blocks = size_t{1} << m_granule_shift;
        const size_t deficit = num_kv_blocks - m_num_allocated_kv_blocks;
        // One chunk per growth step, sized from the shortfall. The floor keeps repeated small shortfalls from creating a
        // long tail of tiny chunks (each one costs 2 * layers USM allocations).
        const size_t growth_floor = m_num_allocated_kv_blocks / kGrowthFloorDivisor;
        const size_t wanted = std::max(deficit, growth_floor);
        const size_t chunk_blocks = (wanted + granule_blocks - 1) / granule_blocks * granule_blocks;
        const size_t new_total_blocks = m_num_allocated_kv_blocks + chunk_blocks;
        const size_t new_chunk_id = m_chunk_blocks.size();
        GENAI_INFO("[KV_TRACE] allocate_chunked_physical_cache requested_blocks=%zu deficit=%zu growth_floor=%zu granule_blocks=%zu | "
                   "added_blocks=%zu (%.1f MiB) physical_blocks %zu->%zu (%.1f MiB) | chunks=%zu new_chunk_id=%zu | "
                   "usm_allocations=%zu (layers=%zu x K,V) table_generation=%zu",
                   num_kv_blocks,
                   deficit,
                   growth_floor,
                   granule_blocks,
                   chunk_blocks,
                   blocks_to_mib(chunk_blocks),
                   m_num_allocated_kv_blocks,
                   new_total_blocks,
                   blocks_to_mib(new_total_blocks),
                   new_chunk_id + 1,
                   new_chunk_id,
                   2 * m_num_layers,
                   m_num_layers,
                   new_chunk_id + 1);
        // Chunk tensors must be true USM device allocations (not the plugin's default cl_mem
        // buffer): usm_pointer_of() hands their address straight to a kernel as a raw pointer, and
        // a cl_mem handle is not a dereferenceable device address.
        const ov::AnyMap usm_device_params{{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER}};
        std::vector<ov::Tensor> new_keys, new_values;
        std::vector<uint64_t> key_ptrs, value_ptrs;
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
            ov::Shape key_chunk_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], chunk_blocks);
            ov::Shape value_chunk_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], chunk_blocks);
            new_keys.push_back(m_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_chunk_shape, usm_device_params));
            new_values.push_back(m_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_chunk_shape, usm_device_params));
            key_ptrs.push_back(usm_pointer_of(new_keys.back()));
            value_ptrs.push_back(usm_pointer_of(new_values.back()));
        }
        // Commit: allocation succeeded for every layer.
        m_chunk_table.append_chunk(chunk_blocks, key_ptrs, value_ptrs);
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
            m_key_chunks[decoder_layer_id].push_back(std::move(new_keys[decoder_layer_id]));
            m_value_chunks[decoder_layer_id].push_back(std::move(new_values[decoder_layer_id]));
            if (new_chunk_id == 0) {
                bind_chunk0_cache_tensors(decoder_layer_id);
            }
        }
        m_chunk_first_block.push_back(m_num_allocated_kv_blocks);
        m_chunk_blocks.push_back(chunk_blocks);
        m_num_allocated_kv_blocks = new_total_blocks;
    }

public:
    /**
     * @brief Check whether the compiled model has KV cache inputs (key_cache.* / value_cache.*).
     * @param compiled_model The compiled model to inspect.
     * @return true if at least one key_cache and one value_cache input are found.
     */
    static bool has_cache_inputs(const ov::CompiledModel& compiled_model) {
        bool has_key = false, has_value = false;
        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find("key_cache.") == 0)
                    has_key = true;
                else if (name.find("value_cache.") == 0)
                    has_value = true;
                if (has_key && has_value)
                    return true;
            }
        }
        return false;
    }

    /**
     * @brief Check whether the compiled model was built with the global chunked KV cache input `kv_chunk_table`
     * (see ov::pass::SDPAToPagedAttention's allow_kv_chunk_table option).
     */
    static bool has_kv_chunk_table_input(const ov::CompiledModel& compiled_model) {
        for (const auto& input : compiled_model.inputs()) {
            if (input.get_names().count("kv_chunk_table") != 0)
                return true;
        }
        return false;
    }

    explicit KVCacheManager(ov::InferRequest request, bool use_chunked_kv_cache = false, size_t granule_blocks = kDefaultGranuleBlocks) :
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
        for (const auto& input : compiled_model.inputs()) {
            for (auto & name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                ov::PartialShape pshape;

                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_key_shapes.push_back(pshape);
                    m_key_precisions.push_back(cache_precision);
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
        m_num_layers = m_value_precisions.size();
        OPENVINO_ASSERT(m_num_layers == m_key_precisions.size(), "Invalid case: a different number of K and V caches in a LLM model");

        if (use_chunked_kv_cache) {
            OPENVINO_ASSERT(m_context, "SchedulerConfig use_chunked_kv_cache requires a GPU device");
            OPENVINO_ASSERT(has_kv_chunk_table_input(compiled_model),
                            "SchedulerConfig use_chunked_kv_cache requires a model converted with a `kv_chunk_table` input");
            m_chunked_kv_cache = true;
            OPENVINO_ASSERT(granule_blocks > 0 && (granule_blocks & (granule_blocks - 1)) == 0,
                            "KV cache granule must be a power of two, got ", granule_blocks);
            m_granule_shift = 0;
            while ((size_t{1} << m_granule_shift) < granule_blocks) {
                ++m_granule_shift;
            }
            m_key_chunks.resize(m_num_layers);
            m_value_chunks.resize(m_num_layers);
            // The table tensor is the request's own (USM host) input tensor: size it once for the whole capacity so it
            // can stay bound and be updated in place on every growth step.
            ov::Tensor table = m_request.get_tensor("kv_chunk_table");
            table.set_shape(ov::Shape{kv_chunk_table::GlobalChunkTable::tensor_elements(m_num_layers, kMaxChunks, kMaxGranules)});
            m_chunk_table = kv_chunk_table::GlobalChunkTable(table, m_num_layers, kMaxChunks, kMaxGranules, m_granule_shift);
        }

        GENAI_INFO("[KV_TRACE] physical_manager device=%s layers=%zu block_size=%zu block_bytes=%zu remote=%s chunked=%s "
                   "granule_blocks=%zu (%.1f MiB) growth_floor=1/%zu of physical blocks",
               m_device.c_str(),
               m_num_layers,
               m_block_size,
               m_block_size_in_bytes,
               m_context ? "true" : "false",
               m_chunked_kv_cache ? "true" : "false",
               m_chunked_kv_cache ? size_t{1} << m_granule_shift : size_t{0},
               blocks_to_mib(m_chunked_kv_cache ? size_t{1} << m_granule_shift : size_t{0}),
               kGrowthFloorDivisor);
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        return m_num_layers;
    }

    size_t get_num_cache_tensors() const override {
        return m_num_layers * 2;
    }

    std::string get_device() const override {
        return m_device;
    }

    size_t get_block_size() const override {
        return m_block_size;
    }

    size_t get_block_size_in_bytes() const override {
        return m_block_size_in_bytes;
    }

    size_t get_num_allocated_blocks() const override {
        return m_num_allocated_kv_blocks;
    }

    std::vector<size_t> get_chunk_first_blocks() const override {
        return m_chunk_first_block;
    }

    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_precisions.size());
        return m_key_precisions[decoder_layer_id];
    }

    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_precisions.size());
        return m_value_precisions[decoder_layer_id];
    }

    // --- KV-cache-specific accessors ---

    size_t sub_byte_data_type_multiplier(const ov::element::Type data_type) const {
        if (data_type == ov::element::i4 || data_type == ov::element::u4)
            return 2;
        return 1;
    }

    void allocate_cache_if_needed(size_t num_kv_blocks) override {
        if (m_num_allocated_kv_blocks >= num_kv_blocks) {
            return;
        }
        if (m_chunked_kv_cache) {
            allocate_chunked_cache_if_needed(num_kv_blocks);
            return;
        }
        GENAI_INFO("[KV_TRACE] allocate_physical_cache old_blocks=%zu new_blocks=%zu layers=%zu",
                   m_num_allocated_kv_blocks,
                   num_kv_blocks,
                   m_num_layers);
        const size_t old_num_kv_blocks = m_num_allocated_kv_blocks;
        try {
            m_num_allocated_kv_blocks = num_kv_blocks;

            ov::Coordinate start_key{0,0,0,0};
            ov::Coordinate start_value{0,0,0,0};

            if (m_context) {// Allocate KV caches
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
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
                        GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=key method=remote_tensor.copy_from bytes=%zu old_blocks=%zu new_blocks=%zu",
                                   decoder_layer_id,
                                   m_key_cache[decoder_layer_id].get_byte_size(),
                                   old_num_kv_blocks,
                                   num_kv_blocks);
                        dst_key_roi.copy_from(m_key_cache[decoder_layer_id]);
                        GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=value method=remote_tensor.copy_from bytes=%zu old_blocks=%zu new_blocks=%zu",
                                   decoder_layer_id,
                                   m_value_cache[decoder_layer_id].get_byte_size(),
                                   old_num_kv_blocks,
                                   num_kv_blocks);
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
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
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
                            GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=key method=memcpy bytes=%zu old_blocks=%zu new_blocks=%zu",
                                       decoder_layer_id,
                                       key_roi_size_byte,
                                       old_num_kv_blocks,
                                       num_kv_blocks);
                            std::memcpy(reinterpret_cast<uint8_t*>(key_cache.data()), reinterpret_cast<uint8_t*>(m_key_cache[decoder_layer_id].data()), key_roi_size_byte);
                        } else {
                            key_roi_size_byte = m_key_cache[decoder_layer_id].get_byte_size();
                            GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=key method=tensor.copy_to bytes=%zu old_blocks=%zu new_blocks=%zu",
                                       decoder_layer_id,
                                       key_roi_size_byte,
                                       old_num_kv_blocks,
                                       num_kv_blocks);
                            ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                            key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + key_roi_size_byte;
                            m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                        }

                        if (value_precision == ov::element::u4) {
                            size_t value_stride = std::accumulate(end_value.begin(), end_value.end(), 1, std::multiplies<size_t>());
                            size_t value_roi_size_byte = value_stride + (value_stride & 1) / sub_byte_data_type_multiplier(value_precision);
                            GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=value method=memcpy bytes=%zu old_blocks=%zu new_blocks=%zu",
                                       decoder_layer_id,
                                       value_roi_size_byte,
                                       old_num_kv_blocks,
                                       num_kv_blocks);
                            std::memcpy(reinterpret_cast<uint8_t*>(value_cache.data()), reinterpret_cast<uint8_t*>(m_value_cache[decoder_layer_id].data()), value_roi_size_byte);
                        } else {
                            value_roi_size_byte = m_value_cache[decoder_layer_id].get_byte_size();
                            GENAI_INFO("[KV_TRACE] cache_grow_copy type=0 layer=%zu tensor=value method=tensor.copy_to bytes=%zu old_blocks=%zu new_blocks=%zu",
                                       decoder_layer_id,
                                       value_roi_size_byte,
                                       old_num_kv_blocks,
                                       num_kv_blocks);
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

    // Byte-level access to one physical block across all layers (per layer: key bytes, then value bytes).
    // Chunk-aware; not on the inference path, used to verify cache contents.
    std::vector<uint8_t> read_block(size_t block_id) const {
        std::vector<uint8_t> data(m_block_size_in_bytes);
        size_t offset = 0;
        for_each_block_segment(block_id, [&](const ov::Tensor& tensor, size_t local_id) {
            offset += copy_block_segment(tensor, local_id, data.data() + offset, /*to_host=*/true);
        });
        OPENVINO_ASSERT(offset == data.size(), "Unexpected KV cache block byte size: ", offset, " vs ", data.size());
        return data;
    }

    void write_block(size_t block_id, const std::vector<uint8_t>& data) {
        OPENVINO_ASSERT(data.size() == m_block_size_in_bytes, "Unexpected KV cache block byte size: got ", data.size(),
                        ", expected ", m_block_size_in_bytes);
        size_t offset = 0;
        for_each_block_segment(block_id, [&](const ov::Tensor& tensor, size_t local_id) {
            offset += copy_block_segment(tensor, local_id, const_cast<uint8_t*>(data.data()) + offset, /*to_host=*/false);
        });
    }

    size_t get_v_head_size(size_t layer_id) const {
        return m_value_shapes[layer_id][3].get_length();
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        if (m_chunked_kv_cache) {
            copy_chunked_blocks(block_copy_map);
            return;
        }
        size_t copied_blocks = 0;
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
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

    void clear() override {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
            if (decoder_layer_id < m_key_cache.size()) {
                m_key_cache[decoder_layer_id] = ov::Tensor();
                m_value_cache[decoder_layer_id] = ov::Tensor();
            }
            if (m_chunked_kv_cache) {
                const bool was_allocated = !m_key_chunks[decoder_layer_id].empty();
                m_key_chunks[decoder_layer_id].clear();
                m_value_chunks[decoder_layer_id].clear();
                if (was_allocated) {
                    // The request still references the chunk-0 tensors it was bound to.
                    bind_placeholder_cache_tensors(decoder_layer_id);
                }
            }
        }
        if (m_chunked_kv_cache) {
            m_chunk_table.clear();
        }
        m_chunk_blocks.clear();
        m_chunk_first_block.clear();
        m_num_allocated_kv_blocks = 0;
    }

private:
    // Requires all four chunks to be USM device allocations (guaranteed: chunked mode only allocates
    // via m_context.create_tensor(), asserted GPU-only at construction time), so no CPU/host fallback
    // path is needed here unlike the legacy copy_blocks() above.
    void copy_chunked_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        size_t copied_blocks = 0;
        for (const auto& blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                ++copied_blocks;
                const size_t src_chunk = chunk_id_of(src_block_id), src_local = local_block_id_of(src_block_id);
                const size_t dst_chunk = chunk_id_of(dst_block_id), dst_local = local_block_id_of(dst_block_id);
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
                    ov::Tensor& key_src_chunk = m_key_chunks[decoder_layer_id][src_chunk];
                    ov::Tensor& key_dst_chunk = m_key_chunks[decoder_layer_id][dst_chunk];
                    ov::Tensor& value_src_chunk = m_value_chunks[decoder_layer_id][src_chunk];
                    ov::Tensor& value_dst_chunk = m_value_chunks[decoder_layer_id][dst_chunk];

                    ov::Shape key_shape = key_src_chunk.get_shape();
                    ov::Coordinate key_src_start(key_shape.size(), 0), key_src_end = key_shape;
                    ov::Coordinate key_dst_start(key_shape.size(), 0), key_dst_end = key_shape;
                    key_src_end[0] = (key_src_start[0] = src_local) + 1;
                    key_dst_end[0] = (key_dst_start[0] = dst_local) + 1;
                    ov::RemoteTensor key_dst_roi(key_dst_chunk, key_dst_start, key_dst_end);
                    ov::RemoteTensor(key_src_chunk, key_src_start, key_src_end).copy_to(key_dst_roi);

                    ov::Shape value_shape = value_src_chunk.get_shape();
                    ov::Coordinate value_src_start(value_shape.size(), 0), value_src_end = value_shape;
                    ov::Coordinate value_dst_start(value_shape.size(), 0), value_dst_end = value_shape;
                    value_src_end[0] = (value_src_start[0] = src_local) + 1;
                    value_dst_end[0] = (value_dst_start[0] = dst_local) + 1;
                    ov::RemoteTensor value_dst_roi(value_dst_chunk, value_dst_start, value_dst_end);
                    ov::RemoteTensor(value_src_chunk, value_src_start, value_src_end).copy_to(value_dst_roi);
                }
            }
        }
        if (copied_blocks > 0) {
            GENAI_INFO("[KV_TRACE] copy_chunked_physical_blocks count=%zu allocated_blocks=%zu",
                       copied_blocks,
                       m_num_allocated_kv_blocks);
        }
    }
};

}
