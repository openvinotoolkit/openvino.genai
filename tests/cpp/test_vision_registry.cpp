// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <future>
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

TEST(VisionRegistry, CompletedImageSurvivesLastRelease) {
    VisionRegistry registry(1024);
    const auto image = make_frames(1);
    const auto first = registry.register_image(image);
    registry.set_encoded_image(first, {});
    registry.release_ref(first);
    EXPECT_EQ(registry.register_image(image), first);
    EXPECT_TRUE(registry.has_encoded_image(first));
}

TEST(VisionRegistry, DefaultCapacityReleasesCompletedImage) {
    VisionRegistry registry;
    const auto id = registry.register_image(make_frames(1));
    registry.set_encoded_image(id, {});
    registry.release_ref(id);
    EXPECT_EQ(registry.size(), 0);
}

TEST(VisionRegistry, IdleBudgetEvictsLeastRecentlyUsedImage) {
    const auto image = make_frames(1);
    VisionRegistry registry(image.get_byte_size() * 2);
    const auto first = registry.register_image(image);
    registry.set_encoded_image(first, {});
    registry.release_ref(first);
    auto changed = make_frames(1);
    changed.data<uint8_t>()[0]++;
    const auto second = registry.register_image(changed);
    registry.set_encoded_image(second, {});
    registry.release_ref(second);
    EXPECT_EQ(registry.register_image(image), first);
    registry.release_ref(first);
    changed.data<uint8_t>()[0]++;
    const auto third = registry.register_image(changed);
    registry.set_encoded_image(third, {});
    registry.release_ref(third);
    EXPECT_TRUE(registry.contains(first));
    EXPECT_FALSE(registry.contains(second));
    EXPECT_TRUE(registry.contains(third));
}

TEST(VisionRegistry, PinnedEntrySurvivesCapacityReduction) {
    VisionRegistry registry(1024);
    const auto id = registry.register_image(make_frames(1));
    registry.set_encoded_image(id, {});
    registry.set_capacity(0);
    EXPECT_TRUE(registry.contains(id));
    registry.release_ref(id);
    EXPECT_FALSE(registry.contains(id));
}

TEST(VisionRegistry, VideoMetadataDistinguishesEntries) {
    VisionRegistry registry;
    const auto frames = make_frames(4);
    const auto first = registry.register_video(frames, {24.0f, {0, 1}});
    EXPECT_EQ(registry.register_video(frames, {24.0f, {0, 1}}), first);
    EXPECT_NE(registry.register_video(frames, {30.0f, {0, 1}}), first);
    EXPECT_NE(registry.register_video(frames, {24.0f, {1, 0}}), first);
}

TEST(VisionRegistry, RepeatedImageSkipsEncodingAndPreservesOutput) {
    VisionRegistry registry(1024);
    size_t calls = 0;
    const auto encode = [&](const ov::Tensor&) {
        ++calls;
        ov::genai::EncodedImage image;
        image.resized_source = make_audio(4, static_cast<float>(calls));
        image.num_image_tokens = 4;
        return image;
    };
    const auto first = registry.encode_image(make_frames(1), encode);
    const auto second = registry.encode_image(make_frames(1), encode);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(second.num_image_tokens, 4);
    registry.set_capacity(0);
    EXPECT_EQ(registry.size(), 0);
    EXPECT_EQ(first.resized_source.data<const float>()[0], 1.0f);
    EXPECT_EQ(second.resized_source.data<const float>()[0], 1.0f);
    registry.encode_image(make_frames(1), encode);
    EXPECT_EQ(calls, 2);
}

TEST(VisionRegistry, VideoMetadataChangeRequiresEncoding) {
    VisionRegistry registry(4096);
    size_t calls = 0;
    const auto encode = [&](const ov::Tensor&, const ov::genai::VideoMetadata& metadata) {
        ++calls;
        ov::genai::EncodedVideo video;
        video.metadata = metadata;
        return video;
    };
    const auto frames = make_frames(4);
    registry.encode_video(frames, {24.0f, {0, 1}}, encode);
    registry.encode_video(frames, {24.0f, {0, 1}}, encode);
    EXPECT_EQ(calls, 1);
    const auto changed = registry.encode_video(frames, {30.0f, {0, 1}}, encode);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(changed.metadata.fps, 30.0f);
    registry.encode_video(frames, {24.0f, {1, 0}}, encode);
    EXPECT_EQ(calls, 3);
}

TEST(VisionRegistry, FailedEncodingReleasesTransientEntry) {
    VisionRegistry registry(1024);
    EXPECT_ANY_THROW(registry.encode_image(make_frames(1), [](const ov::Tensor&) -> ov::genai::EncodedImage {
        OPENVINO_THROW("Expected encoder failure");
    }));
    EXPECT_EQ(registry.size(), 0);
}

TEST(VisionRegistry, PayloadBudgetIncludesEncodedTensors) {
    const auto image = make_frames(1);
    VisionRegistry registry(image.get_byte_size());
    size_t calls = 0;
    const auto encode = [&](const ov::Tensor&) {
        ++calls;
        ov::genai::EncodedImage encoded;
        encoded.resized_source = make_audio(1);
        return encoded;
    };
    registry.encode_image(image, encode);
    EXPECT_EQ(registry.size(), 0);
    registry.encode_image(image, encode);
    EXPECT_EQ(calls, 2);
}

TEST(VisionRegistry, InputMutationRequiresEncoding) {
    VisionRegistry registry(1024);
    size_t calls = 0;
    const auto encode = [&](const ov::Tensor&) {
        ++calls;
        return ov::genai::EncodedImage{};
    };
    auto image = make_frames(1);
    registry.encode_image(image, encode);
    image.data<uint8_t>()[0]++;
    registry.encode_image(image, encode);
    registry.encode_image(make_frames(1), encode);
    EXPECT_EQ(calls, 2);
}

TEST(VisionRegistry, ConcurrentMissesEncodeOnce) {
    VisionRegistry registry(1024);
    size_t calls = 0;
    const auto encode = [&](const ov::Tensor&) {
        ++calls;
        return ov::genai::EncodedImage{};
    };
    auto first = std::async(std::launch::async, [&] { return registry.encode_image(make_frames(1), encode); });
    auto second = std::async(std::launch::async, [&] { return registry.encode_image(make_frames(1), encode); });
    EXPECT_EQ(first.get().num_image_tokens, 0);
    EXPECT_EQ(second.get().num_image_tokens, 0);
    EXPECT_EQ(calls, 1);
}

TEST(VisionRegistry, AudioIsNotRetainedByVisionBudget) {
    VisionRegistry registry(1024);
    const auto id = registry.register_audio(make_audio(4));
    registry.set_encoded_audio(id, {});
    registry.release_ref(id);
    EXPECT_EQ(registry.size(), 0);
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

TEST(VisionRegistry, ReleasedChainHeadNeverReturnsWrongEntry) {
    VisionRegistry registry;
    auto changed = make_audio(LONG_AUDIO_SAMPLES);
    changed.data<float>()[UNSAMPLED_SAMPLE] = -1.0f;

    const auto first = registry.register_audio(make_audio(LONG_AUDIO_SAMPLES));
    const auto second = registry.register_audio(changed);
    registry.release_ref(first);

    const auto again = registry.register_audio(changed);
    EXPECT_EQ(again, second);
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

namespace {

constexpr size_t PARENT_HEIGHT = 4;
constexpr size_t PARENT_WIDTH = 8;

// Every row has its own fill, so an ROI spanning two rows differs from any single parent row.
ov::Tensor make_striped_image() {
    ov::Tensor image{ov::element::u8, {PARENT_HEIGHT, PARENT_WIDTH, 3}};
    auto* data = image.data<uint8_t>();
    for (size_t row = 0; row < PARENT_HEIGHT; ++row) {
        std::fill_n(data + row * PARENT_WIDTH * 3, PARENT_WIDTH * 3, static_cast<uint8_t>(row + 1));
    }
    return image;
}

// Left half of the top two rows: strided, since each row skips the right half of the parent.
ov::Tensor top_left_roi(const ov::Tensor& parent) {
    return ov::Tensor{parent, ov::Coordinate{0, 0, 0}, ov::Coordinate{2, PARENT_WIDTH / 2, 3}};
}

ov::Tensor packed_copy(const ov::Tensor& tensor) {
    ov::Tensor packed{tensor.get_element_type(), tensor.get_shape()};
    tensor.copy_to(packed);
    return packed;
}

}  // namespace

TEST(VisionRegistry, RoiTensorMatchesPackedCopy) {
    VisionRegistry registry;
    const auto parent = make_striped_image();
    const auto roi = top_left_roi(parent);
    ASSERT_FALSE(roi.is_continuous());

    const auto from_roi = registry.register_image(roi);
    const auto from_copy = registry.register_image(packed_copy(roi));
    EXPECT_EQ(from_roi, from_copy);
    EXPECT_EQ(registry.size(), 1);
}

TEST(VisionRegistry, RoiTensorDoesNotMatchItsRawSpan) {
    VisionRegistry registry;
    const auto parent = make_striped_image();
    const auto roi = top_left_roi(parent);
    // Same shape as the ROI, holding the bytes the ROI's data pointer runs over: all of parent row 0.
    ov::Tensor raw_span{ov::element::u8, roi.get_shape()};
    std::fill_n(raw_span.data<uint8_t>(), raw_span.get_size(), uint8_t{1});

    const auto span_id = registry.register_image(raw_span);
    const auto roi_id = registry.register_image(roi);
    EXPECT_NE(span_id, roi_id);
    EXPECT_EQ(registry.size(), 2);

    const auto& stored = registry.get_original(roi_id);
    const auto expected = packed_copy(roi);
    ASSERT_EQ(stored.get_byte_size(), expected.get_byte_size());
    EXPECT_EQ(std::memcmp(stored.data(), expected.data(), expected.get_byte_size()), 0);
}
