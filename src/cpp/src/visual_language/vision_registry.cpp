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

// Hash tensor using FNV-1a algorithm.
// See: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
VisionID VisionRegistry::compute_hash(const ov::Tensor& tensor) {
    // FNV-1a parameters (64-bit)
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325;
    constexpr uint64_t FNV_PRIME = 0x100000001b3;
    
    uint64_t hash = FNV_OFFSET_BASIS;
    
    // Hash shape
    for (auto dim : tensor.get_shape()) {
        hash ^= dim;
        hash *= FNV_PRIME;
    }
    
    // Hash element type
    hash ^= static_cast<uint64_t>(tensor.get_element_type().hash());
    hash *= FNV_PRIME;
    
    // Hash tensor content
    const uint8_t* data = tensor.data<uint8_t>();
    const size_t byte_size = tensor.get_byte_size();
    for (size_t i = 0; i < byte_size; ++i) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }
    
    return std::to_string(hash);
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
    if (it != m_entries.end()) {
        it->second.ref_count++;
    }
}

void VisionRegistry::release_ref(const VisionID& id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    if (it != m_entries.end()) {
        if (--it->second.ref_count == 0) {
            m_entries.erase(it);
        }
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
