// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vision_registry.hpp"

namespace ov::genai {

VisionRegistry::VisionEntry::VisionEntry(VisionType t, ov::Tensor tensor)
    : type(t), original(std::move(tensor)), ref_count(0) {}

VisionRegistry::VisionEntry::VisionEntry(VisionEntry&& other) noexcept
    : type(other.type),
      original(std::move(other.original)),
      encoded_image(std::move(other.encoded_image)),
      encoded_video(std::move(other.encoded_video)),
      ref_count(other.ref_count.load()) {}

VisionRegistry::VisionEntry& VisionRegistry::VisionEntry::operator=(VisionEntry&& other) noexcept {
    if (this != &other) {
        type = other.type;
        original = std::move(other.original);
        encoded_image = std::move(other.encoded_image);
        encoded_video = std::move(other.encoded_video);
        ref_count.store(other.ref_count.load());
    }
    return *this;
}

namespace {

constexpr size_t MIN_SAMPLE_BYTES = 64 * 1024;
constexpr size_t MAX_SAMPLE_BYTES = 256 * 1024;
constexpr double SAMPLE_RATIO = 0.0625; // 1 / 16
constexpr size_t HASH_CHUNK_SIZE = sizeof(uint64_t);  // 8 bytes

size_t calculate_hash_stride(size_t byte_size) {
    if (byte_size <= MIN_SAMPLE_BYTES) {
        return 1;
    }

    size_t target_size = static_cast<size_t>(byte_size * SAMPLE_RATIO);
    target_size = std::clamp(target_size, MIN_SAMPLE_BYTES, MAX_SAMPLE_BYTES);
 
    size_t stride = byte_size / target_size;
    
    // Align to 8-byte chunks for uint64_t access
    stride = stride & ~(HASH_CHUNK_SIZE - 1);
    return std::max(HASH_CHUNK_SIZE, stride);
}

} // namespace

// Hash tensor using FNV-1a algorithm.
// See: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
VisionID VisionRegistry::compute_hash(const ov::Tensor& tensor) {
    // FNV-1a parameters (64-bit)
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325;
    constexpr uint64_t FNV_PRIME = 0x100000001b3;
    
    uint64_t hash = FNV_OFFSET_BASIS;

    const auto& shape = tensor.get_shape();

    // Hash shape
    for (const auto dim : shape) {
        hash ^= dim;
        hash *= FNV_PRIME;
    }
    
    // Hash element type
    hash ^= static_cast<uint64_t>(tensor.get_element_type().hash());
    hash *= FNV_PRIME;
    
    // Hash tensor content
    const uint8_t* data = tensor.data<uint8_t>();
    const size_t byte_size = tensor.get_byte_size();
    const uint64_t* data64 = reinterpret_cast<const uint64_t*>(data);
    
    const size_t num_frames = shape.size() == 4 ? shape[0] : 1;
    const size_t frame_size_bytes = byte_size / num_frames;
    const size_t frame_chunks = frame_size_bytes / HASH_CHUNK_SIZE;
    
    const size_t frame_stride = calculate_hash_stride(frame_size_bytes);
    
    for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const uint64_t* frame_data = data64 + (frame_idx * frame_chunks);
        
        for (size_t i = 0; i < frame_chunks; i += frame_stride) {
            hash ^= frame_data[i];
            hash *= FNV_PRIME;
        }
        
        // Hash last chunk if strided loop didn't process it
        if (frame_stride > 1 && frame_chunks > 0 && (frame_chunks - 1) % frame_stride != 0) {
            hash ^= frame_data[frame_chunks - 1];
            hash *= FNV_PRIME;
        }
        
        // Hash remaining bytes
        const size_t frame_offset = frame_idx * frame_size_bytes;
        const size_t remaining_start = frame_offset + frame_chunks * HASH_CHUNK_SIZE;
        const size_t remaining_end = frame_offset + frame_size_bytes;
        for (size_t i = remaining_start; i < remaining_end; ++i) {
            hash ^= data[i];
            hash *= FNV_PRIME;
        }
    }
    
    return hash;
}

VisionID VisionRegistry::register_vision(const ov::Tensor& tensor, VisionType type) {
    VisionID id = compute_hash(tensor);
    
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    if (it == m_entries.end()) {
        ov::Tensor owned_tensor(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(owned_tensor);
        m_entries.emplace(id, VisionEntry(type, std::move(owned_tensor)));
    }
    m_entries.at(id).ref_count++;
    return id;
}

VisionID VisionRegistry::register_image(const ov::Tensor& image) {
    return register_vision(image, VisionType::IMAGE);
}

VisionID VisionRegistry::register_video(const ov::Tensor& video) {
    return register_vision(video, VisionType::VIDEO);
}

std::vector<VisionID> VisionRegistry::register_images(const std::vector<ov::Tensor>& images) {
    std::vector<VisionID> ids;
    ids.reserve(images.size());
    for (const auto& img : images) {
        ids.push_back(register_image(img));
    }
    return ids;
}

std::vector<VisionID> VisionRegistry::register_videos(const std::vector<ov::Tensor>& videos) {
    std::vector<VisionID> ids;
    ids.reserve(videos.size());
    for (const auto& vid : videos) {
        ids.push_back(register_video(vid));
    }
    return ids;
}

void VisionRegistry::add_ref(const VisionID& id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    OPENVINO_ASSERT(it != m_entries.end(), "Vision ID not found in VisionRegistry: ", id);
    it->second.ref_count++;
}

void VisionRegistry::release_ref(const VisionID& id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    OPENVINO_ASSERT(it != m_entries.end(), "Vision ID not found in VisionRegistry: ", id);
    if (--it->second.ref_count == 0) {
        m_entries.erase(it);
    }
}

size_t VisionRegistry::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

bool VisionRegistry::contains(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.find(id) != m_entries.end();
}

VisionType VisionRegistry::get_type(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.at(id).type;
}

const ov::Tensor& VisionRegistry::get_original(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.at(id).original;
}

void VisionRegistry::set_encoded_image(const VisionID& id, EncodedImage encoded) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == VisionType::IMAGE, 
                    "Cannot set encoded image for video entry");
    entry.encoded_image = std::move(encoded);
}

bool VisionRegistry::has_encoded_image(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    return it != m_entries.end() && it->second.encoded_image.has_value();
}

const EncodedImage& VisionRegistry::get_encoded_image(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == VisionType::IMAGE,
                    "Cannot get encoded image for video entry");
    OPENVINO_ASSERT(entry.encoded_image.has_value(),
                    "Encoded image not available for id: ", id);
    return *entry.encoded_image;
}

void VisionRegistry::set_encoded_video(const VisionID& id, EncodedVideo encoded) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == VisionType::VIDEO,
                    "Cannot set encoded video for image entry");
    entry.encoded_video = std::move(encoded);
}

bool VisionRegistry::has_encoded_video(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    return it != m_entries.end() && it->second.encoded_video.has_value();
}

const EncodedVideo& VisionRegistry::get_encoded_video(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == VisionType::VIDEO,
                    "Cannot get encoded video for image entry");
    OPENVINO_ASSERT(entry.encoded_video.has_value(),
                    "Encoded video not available for id: ", id);
    return *entry.encoded_video;
}

} // namespace ov::genai
