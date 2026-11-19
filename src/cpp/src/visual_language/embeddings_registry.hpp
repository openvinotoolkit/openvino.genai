// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/vision_encoder.hpp"
#include <optional>

namespace ov::genai {

using EmbeddingID = uint64_t;

class EmbeddingsRegistry {
public:
    EmbeddingsRegistry() = default;

    EmbeddingsRegistry(const EmbeddingsRegistry&) = delete;
    EmbeddingsRegistry& operator=(const EmbeddingsRegistry&) = delete;
    EmbeddingsRegistry(EmbeddingsRegistry&&) = delete;
    EmbeddingsRegistry& operator=(EmbeddingsRegistry&&) = delete;
    
    ~EmbeddingsRegistry() = default;

    EmbeddingID register_image(const ov::Tensor& image);
    EmbeddingID register_video(const ov::Tensor& video);
    EmbeddingID register_audio(const ov::Tensor& audio);

    std::vector<EmbeddingID> register_images(const std::vector<ov::Tensor>& images);
    std::vector<EmbeddingID> register_videos(const std::vector<ov::Tensor>& videos);
    std::vector<EmbeddingID> register_audios(const std::vector<ov::Tensor>& audios);

    void add_ref(const EmbeddingID& id);
    void release_ref(const EmbeddingID& id);

    size_t size() const;
    bool contains(const EmbeddingID& id) const;
    ModalityType get_type(const EmbeddingID& id) const;

    const ov::Tensor& get_original(const EmbeddingID& id) const;

    void set_encoded_image(const EmbeddingID& id, EncodedImage encoded);
    bool has_encoded_image(const EmbeddingID& id) const;
    const EncodedImage& get_encoded_image(const EmbeddingID& id) const;

    void set_encoded_video(const EmbeddingID& id, EncodedVideo encoded);
    bool has_encoded_video(const EmbeddingID& id) const;
    const EncodedVideo& get_encoded_video(const EmbeddingID& id) const;

    void set_encoded_audio(const EmbeddingID& id, EncodedAudio encoded);
    bool has_encoded_audio(const EmbeddingID& id) const;
    const EncodedAudio& get_encoded_audio(const EmbeddingID& id) const;

private:
    struct EmbeddingEntry {
        ModalityType type;
        ov::Tensor original;
        std::optional<EncodedImage> encoded_image;
        std::optional<EncodedVideo> encoded_video;
        std::optional<EncodedAudio> encoded_audio;
        std::atomic<size_t> ref_count{0};
        
        EmbeddingEntry(ModalityType t, ov::Tensor tensor);
        EmbeddingEntry(EmbeddingEntry&& other) noexcept;
        EmbeddingEntry& operator=(EmbeddingEntry&& other) noexcept;
        
        EmbeddingEntry(const EmbeddingEntry&) = delete;
        EmbeddingEntry& operator=(const EmbeddingEntry&) = delete;

        ~EmbeddingEntry() = default;
    };

    std::unordered_map<EmbeddingID, EmbeddingEntry> m_entries;

    mutable std::mutex m_mutex;

    EmbeddingID register_media(const ov::Tensor& tensor, ModalityType type);

    static EmbeddingID compute_hash(const ov::Tensor& tensor);
};

} // namespace ov::genai
