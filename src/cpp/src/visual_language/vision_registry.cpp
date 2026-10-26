// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vision_registry.hpp"
#include "logger.hpp"

#include <chrono>
#include <cstring>

namespace ov::genai {

VisionRegistry::VisionEntry::VisionEntry(ModalityType t, ov::Tensor tensor)
    : type(t), original(std::move(tensor)), ref_count(0) {}

VisionRegistry::VisionEntry::VisionEntry(VisionEntry&& other) noexcept
    : type(other.type),
      original(std::move(other.original)),
      encoded_image(std::move(other.encoded_image)),
      encoded_video(std::move(other.encoded_video)),
      encoded_audio(std::move(other.encoded_audio)),
    ref_count(other.ref_count.load()),
    metadata(std::move(other.metadata)),
    content_hash(other.content_hash),
    retained_bytes(other.retained_bytes),
    idle_position(other.idle_position) {}

VisionRegistry::VisionEntry& VisionRegistry::VisionEntry::operator=(VisionEntry&& other) noexcept {
    if (this != &other) {
        type = other.type;
        original = std::move(other.original);
        encoded_image = std::move(other.encoded_image);
        encoded_video = std::move(other.encoded_video);
        encoded_audio = std::move(other.encoded_audio);
        ref_count.store(other.ref_count.load());
        metadata = std::move(other.metadata);
        content_hash = other.content_hash;
        retained_bytes = other.retained_bytes;
        idle_position = other.idle_position;
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

// memcpy instead of a uint64_t* cast: user and ROI tensors are not guaranteed to be 8-byte aligned.
uint64_t load_u64(const uint8_t* ptr) {
    uint64_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

bool same_content(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    return lhs.get_element_type() == rhs.get_element_type() && lhs.get_shape() == rhs.get_shape() &&
           (lhs.get_byte_size() == 0 || std::memcmp(lhs.data(), rhs.data(), lhs.get_byte_size()) == 0);
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
    
    // Untyped pointer because data<uint8_t>() throws for f32 audio. The element type is already
    // in the hash, so raw bytes stay unambiguous.
    const auto* data = static_cast<const uint8_t*>(tensor.data());
    const size_t byte_size = tensor.get_byte_size();

    const size_t num_frames = shape.size() == 4 ? shape[0] : 1;
    const size_t frame_size_bytes = byte_size / num_frames;
    const size_t frame_chunks = frame_size_bytes / HASH_CHUNK_SIZE;
    
    const size_t frame_stride = calculate_hash_stride(frame_size_bytes);
    
    for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const auto* frame_data = data + frame_idx * frame_size_bytes;

        for (size_t i = 0; i < frame_chunks; i += frame_stride) {
            hash ^= load_u64(frame_data + i * HASH_CHUNK_SIZE);
            hash *= FNV_PRIME;
        }
        
        // Hash last chunk if strided loop didn't process it
        if (frame_stride > 1 && frame_chunks > 0 && (frame_chunks - 1) % frame_stride != 0) {
            hash ^= load_u64(frame_data + (frame_chunks - 1) * HASH_CHUNK_SIZE);
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

VisionID VisionRegistry::register_vision(const ov::Tensor& tensor, ModalityType type, const VideoMetadata& metadata) {
    const auto pack = [&tensor]() {
        ov::Tensor packed(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(packed);
        return packed;
    };
    // Hashing and comparison read raw bytes, which for a strided ROI are not its logical content.
    const bool is_contiguous = tensor.is_continuous();
    ov::Tensor content = is_contiguous ? tensor : pack();
    auto hash = compute_hash(content);
    if (type == ModalityType::VIDEO) {
        hash ^= std::hash<float>{}(metadata.fps);
        for (size_t index : metadata.frames_indices) {
            hash = (hash ^ index) * 0x100000001b3;
        }
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    auto& bucket = m_hash_buckets[hash];
    for (VisionID id : bucket) {
        auto& entry = m_entries.at(id);
        if (entry.type == type && entry.metadata.fps == metadata.fps &&
            entry.metadata.frames_indices == metadata.frames_indices && same_content(entry.original, content)) {
            pin(entry);
            return id;
        }
    }
    // A packed ROI is already a private copy; a contiguous tensor may be caller memory and needs one.
    ov::Tensor owned_tensor = is_contiguous ? pack() : std::move(content);
    const VisionID id = m_next_id++;
    auto& entry = m_entries.emplace(id, VisionEntry(type, std::move(owned_tensor))).first->second;
    entry.metadata = metadata;
    entry.content_hash = hash;
    entry.ref_count++;
    bucket.push_back(id);
    return id;
}

VisionID VisionRegistry::register_image(const ov::Tensor& image) {
    return register_vision(image, ModalityType::IMAGE);
}

VisionID VisionRegistry::register_video(const ov::Tensor& video, const VideoMetadata& metadata) {
    return register_vision(video, ModalityType::VIDEO, metadata);
}

VisionID VisionRegistry::register_audio(const ov::Tensor& audio) {
    return register_vision(audio, ModalityType::AUDIO);
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

std::vector<VisionID> VisionRegistry::register_audios(const std::vector<ov::Tensor>& audios) {
    std::vector<VisionID> ids;
    ids.reserve(audios.size());
    for (const auto& audio : audios) {
        ids.push_back(register_audio(audio));
    }
    return ids;
}

void VisionRegistry::add_ref(const VisionID& id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    OPENVINO_ASSERT(it != m_entries.end(), "Vision ID not found in VisionRegistry: ", id);
    pin(it->second);
}

void VisionRegistry::release_ref(const VisionID& id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    OPENVINO_ASSERT(it != m_entries.end(), "Vision ID not found in VisionRegistry: ", id);
    OPENVINO_ASSERT(it->second.ref_count > 0, "Vision entry has no references: ", id);
    if (--it->second.ref_count == 0) {
        auto& entry = it->second;
        if (m_capacity_bytes == 0 || (!entry.encoded_image && !entry.encoded_video)) {
            erase_entry(id);
            return;
        }
        entry.retained_bytes = entry_bytes(entry);
        m_idle_bytes += entry.retained_bytes;
        m_idle_entries.push_back(id);
        entry.idle_position = std::prev(m_idle_entries.end());
        evict_idle_entries();
    }
}

void VisionRegistry::set_capacity(size_t capacity_bytes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_capacity_bytes = capacity_bytes;
    evict_idle_entries();
}

EncodedImage VisionRegistry::encode_image(const ov::Tensor& image,
                                          const std::function<EncodedImage(const ov::Tensor&)>& encode) {
    const auto lookup_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_encoding_mutex);
    const VisionID id = register_image(image);
    const ScopedReference reference{*this, id};
    const bool cache_hit = has_encoded_image(id);
    const auto encode_start = std::chrono::steady_clock::now();
    if (!cache_hit) {
        set_encoded_image(id, encode(get_original(id)));
    }
    GENAI_DEBUG("Image embedding cache: id=%llu hit=%s lookup_ms=%.3f encoder_ms=%.3f",
                static_cast<unsigned long long>(id), cache_hit ? "true" : "false",
                std::chrono::duration<double, std::milli>(encode_start - lookup_start).count(),
                cache_hit ? 0.0 : std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - encode_start).count());
    return get_encoded_image(id);
}

EncodedVideo VisionRegistry::encode_video(
    const ov::Tensor& video,
    const VideoMetadata& metadata,
    const std::function<EncodedVideo(const ov::Tensor&, const VideoMetadata&)>& encode) {
    std::lock_guard<std::mutex> lock(m_encoding_mutex);
    const VisionID id = register_video(video, metadata);
    const ScopedReference reference{*this, id};
    if (!has_encoded_video(id)) {
        set_encoded_video(id, encode(get_original(id), metadata));
    }
    return get_encoded_video(id);
}

void VisionRegistry::pin(VisionEntry& entry) {
    if (entry.idle_position) {
        m_idle_entries.erase(*entry.idle_position);
        entry.idle_position.reset();
        m_idle_bytes -= entry.retained_bytes;
    }
    entry.ref_count++;
}

void VisionRegistry::erase_entry(VisionID id) {
    auto& entry = m_entries.at(id);
    if (entry.idle_position) {
        m_idle_entries.erase(*entry.idle_position);
        m_idle_bytes -= entry.retained_bytes;
    }
    auto bucket = m_hash_buckets.find(entry.content_hash);
    auto& ids = bucket->second;
    ids.erase(std::find(ids.begin(), ids.end(), id));
    if (ids.empty()) {
        m_hash_buckets.erase(bucket);
    }
    m_entries.erase(id);
}

void VisionRegistry::evict_idle_entries() {
    while (!m_idle_entries.empty() && (m_capacity_bytes == 0 || m_idle_bytes > m_capacity_bytes)) {
        erase_entry(m_idle_entries.front());
    }
}

size_t VisionRegistry::entry_bytes(const VisionEntry& entry) {
    size_t bytes = 0;
    std::unordered_map<const void*, size_t> buffers;
    const auto account = [&](const ov::Tensor& tensor) {
        if (tensor) {
            auto& accounted = buffers[tensor.data()];
            const size_t size = tensor.get_byte_size();
            if (size > accounted) {
                bytes += size - accounted;
                accounted = size;
            }
        }
    };
    account(entry.original);
    if (entry.encoded_image) {
        const auto& image = *entry.encoded_image;
        account(image.resized_source);
        account(image.images_features_projection);
        account(image.resampled_image.resampled_source);
        for (const auto& row : image.resampled_image.vision_embed_tensors) {
            for (const auto& tensor : row) {
                account(tensor);
            }
        }
    }
    if (entry.encoded_video) {
        account(entry.encoded_video->video_features);
    }
    return bytes;
}

size_t VisionRegistry::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

bool VisionRegistry::contains(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.find(id) != m_entries.end();
}

ModalityType VisionRegistry::get_type(const VisionID& id) const {
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
    OPENVINO_ASSERT(entry.type == ModalityType::IMAGE, "Cannot set encoded image for a non-image entry");
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
    OPENVINO_ASSERT(entry.type == ModalityType::IMAGE, "Cannot get encoded image for a non-image entry");
    OPENVINO_ASSERT(entry.encoded_image.has_value(),
                    "Encoded image not available for id: ", id);
    return *entry.encoded_image;
}

void VisionRegistry::set_encoded_video(const VisionID& id, EncodedVideo encoded) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == ModalityType::VIDEO, "Cannot set encoded video for a non-video entry");
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
    OPENVINO_ASSERT(entry.type == ModalityType::VIDEO, "Cannot get encoded video for a non-video entry");
    OPENVINO_ASSERT(entry.encoded_video.has_value(),
                    "Encoded video not available for id: ", id);
    return *entry.encoded_video;
}

void VisionRegistry::set_encoded_audio(const VisionID& id, EncodedAudio encoded) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == ModalityType::AUDIO, "Cannot set encoded audio for a non-audio entry");
    entry.encoded_audio = std::move(encoded);
}

bool VisionRegistry::has_encoded_audio(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_entries.find(id);
    return it != m_entries.end() && it->second.encoded_audio.has_value();
}

const EncodedAudio& VisionRegistry::get_encoded_audio(const VisionID& id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto& entry = m_entries.at(id);
    OPENVINO_ASSERT(entry.type == ModalityType::AUDIO, "Cannot get encoded audio for a non-audio entry");
    OPENVINO_ASSERT(entry.encoded_audio.has_value(), "Encoded audio not available for id: ", id);
    return *entry.encoded_audio;
}

} // namespace ov::genai
