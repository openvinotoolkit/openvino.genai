// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "visual_language/vision_registry.hpp"

using ov::genai::ModalityType;
using ov::genai::VisionRegistry;

namespace {

// 10 s of 16 kHz audio. At 640000 bytes the hash reads every 8th 8-byte chunk, so chunk 1 is skipped.
constexpr size_t LONG_AUDIO_SAMPLES = 160000;
constexpr size_t UNSAMPLED_SAMPLE = 3;  // bytes 12..15, inside chunk 1

ov::Tensor make_audio(size_t samples, float fill = 0.5f) {
    ov::Tensor audio{ov::element::f32, {samples}};
    std::fill_n(audio.data<float>(), samples, fill);
    return audio;
}

ov::Tensor make_frames(size_t frames) {
    ov::Tensor video{ov::element::u8, {frames, 8, 8, 3}};
    std::fill_n(video.data<uint8_t>(), video.get_size(), uint8_t{42});
    return video;
}

}  // namespace

TEST(VisionRegistry, SameContentSharesEntry) {
    VisionRegistry registry;
    const auto first = registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    const auto second = registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    EXPECT_EQ(first, second);
    EXPECT_EQ(registry.size(), 1);
}

// Before the fix the second audio hit the first entry and reused its embeddings.
TEST(VisionRegistry, DifferenceInUnsampledBytesGetsOwnEntry) {
    VisionRegistry registry;
    auto changed = make_audio(LONG_AUDIO_SAMPLES);
    changed.data<float>()[UNSAMPLED_SAMPLE] = -1.0f;

    const auto first = registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    const auto second = registry.register_audio(changed);

    EXPECT_NE(first, second);
    EXPECT_EQ(registry.size(), 2);
    EXPECT_EQ(registry.get_original(second).data<const float>()[UNSAMPLED_SAMPLE], -1.0f);
    EXPECT_EQ(registry.get_original(first).data<const float>()[UNSAMPLED_SAMPLE], 0.5f);
}

TEST(VisionRegistry, CollidingEntryIsFoundAgainAfterProbe) {
    VisionRegistry registry;
    auto changed = make_audio(LONG_AUDIO_SAMPLES);
    changed.data<float>()[UNSAMPLED_SAMPLE] = -1.0f;

    registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    const auto second = registry.register_audio(changed);
    EXPECT_EQ(registry.register_audio(changed), second);
    EXPECT_EQ(registry.size(), 2);
}

// Releasing the head of a probe chain may cost a duplicate entry, never a wrong reuse.
TEST(VisionRegistry, ReleasedChainHeadNeverReturnsWrongEntry) {
    VisionRegistry registry;
    auto changed = make_audio(LONG_AUDIO_SAMPLES);
    changed.data<float>()[UNSAMPLED_SAMPLE] = -1.0f;

    const auto first = registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    registry.register_audio(changed);
    registry.release_ref(first);

    const auto again = registry.register_audio(changed);
    EXPECT_EQ(registry.get_original(again).data<const float>()[UNSAMPLED_SAMPLE], -1.0f);
}

// A one-frame video and an image built from the same tensor used to share one entry, and
// set_encoded_video() then failed on the image entry.
TEST(VisionRegistry, SameTensorAsImageAndVideoGetsSeparateEntries) {
    VisionRegistry registry;
    const auto frames = make_frames(1);
    const auto image = registry.register_image(frames);
    const auto video = registry.register_video(frames);

    EXPECT_NE(image, video);
    EXPECT_EQ(registry.get_type(image), ModalityType::IMAGE);
    EXPECT_EQ(registry.get_type(video), ModalityType::VIDEO);
    EXPECT_NO_THROW(registry.set_encoded_video(video, {}));
}

TEST(VisionRegistry, EmptyAudioSharesEntry) {
    VisionRegistry registry;
    const auto first = registry.register_audio(make_audio(0));
    const auto second = registry.register_audio(make_audio(0));
    EXPECT_EQ(first, second);
    EXPECT_EQ(registry.size(), 1);
}

// Tensors wrapping caller memory are not guaranteed to be 8-byte aligned.
TEST(VisionRegistry, MisalignedTensorMatchesAlignedCopy) {
    VisionRegistry registry;
    constexpr size_t samples = 1001;
    std::vector<uint64_t> storage(samples / 2 + 2);
    auto* misaligned = reinterpret_cast<uint8_t*>(storage.data()) + sizeof(float);
    ov::Tensor view{ov::element::f32, {samples}, misaligned};
    std::fill_n(view.data<float>(), samples, 0.25f);

    const auto from_view = registry.register_audio(view);
    const auto from_copy = registry.register_audio(make_audio(samples, 0.25f));
    EXPECT_EQ(from_view, from_copy);
    EXPECT_EQ(registry.size(), 1);
}
