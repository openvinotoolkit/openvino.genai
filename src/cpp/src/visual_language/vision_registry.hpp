// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vision_encoder.hpp"
#include <optional>

namespace ov::genai {

using VisionID = uint64_t;

class VisionRegistry {
public:
    VisionRegistry() = default;

    VisionRegistry(const VisionRegistry&) = delete;
    VisionRegistry& operator=(const VisionRegistry&) = delete;
    VisionRegistry(VisionRegistry&&) = delete;
    VisionRegistry& operator=(VisionRegistry&&) = delete;
    
    ~VisionRegistry() = default;

    VisionID register_image(const ov::Tensor& image);
    VisionID register_video(const ov::Tensor& video);
    VisionID register_audio(const ov::Tensor& audio);

    std::vector<VisionID> register_images(const std::vector<ov::Tensor>& images);
    std::vector<VisionID> register_videos(const std::vector<ov::Tensor>& videos);
    std::vector<VisionID> register_audios(const std::vector<ov::Tensor>& audios);

    void add_ref(const VisionID& id);
    void release_ref(const VisionID& id);

    size_t size() const;
    bool contains(const VisionID& id) const;
    ModalityType get_type(const VisionID& id) const;

    const ov::Tensor& get_original(const VisionID& id) const;

    void set_encoded_image(const VisionID& id, EncodedImage encoded);
    bool has_encoded_image(const VisionID& id) const;
    const EncodedImage& get_encoded_image(const VisionID& id) const;

    void set_encoded_video(const VisionID& id, EncodedVideo encoded);
    bool has_encoded_video(const VisionID& id) const;
    const EncodedVideo& get_encoded_video(const VisionID& id) const;

    void set_encoded_audio(const VisionID& id, EncodedAudio encoded);
    bool has_encoded_audio(const VisionID& id) const;
    const EncodedAudio& get_encoded_audio(const VisionID& id) const;

private:
    struct VisionEntry {
        ModalityType type;
        ov::Tensor original;
        std::optional<EncodedImage> encoded_image;
        std::optional<EncodedVideo> encoded_video;
        std::optional<EncodedAudio> encoded_audio;
        std::atomic<size_t> ref_count{0};
        
        VisionEntry(ModalityType t, ov::Tensor tensor);
        VisionEntry(VisionEntry&& other) noexcept;
        VisionEntry& operator=(VisionEntry&& other) noexcept;
        
        VisionEntry(const VisionEntry&) = delete;
        VisionEntry& operator=(const VisionEntry&) = delete;

        ~VisionEntry() = default;
    };

    std::unordered_map<VisionID, VisionEntry> m_entries;

    mutable std::mutex m_mutex;

    VisionID register_vision(const ov::Tensor& tensor, ModalityType type);

    static VisionID compute_hash(const ov::Tensor& tensor);
};

} // namespace ov::genai
