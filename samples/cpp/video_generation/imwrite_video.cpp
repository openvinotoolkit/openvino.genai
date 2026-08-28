// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <fstream>

#include "imwrite_video.hpp"

namespace {

// AVI headers, written to the file as-is. AVI is a little-endian format, so this assumes a little-endian host.
#pragma pack(push, 1)
struct MainAviHeader {
    uint32_t microsec_per_frame;
    uint32_t max_bytes_per_sec = 0;
    uint32_t padding_granularity = 0;
    uint32_t flags = 0x110;  // AVIF_HASINDEX | AVIF_ISINTERLEAVED
    uint32_t total_frames;
    uint32_t initial_frames = 0;
    uint32_t streams = 2;
    uint32_t suggested_buffer_size;
    uint32_t width;
    uint32_t height;
    uint32_t reserved[4] = {};
};

struct AviStreamHeader {
    uint32_t fcc_type;
    uint32_t fcc_handler = 0;
    uint32_t flags = 0;
    uint16_t priority = 0;
    uint16_t language = 0;
    uint32_t initial_frames = 0;
    uint32_t scale;
    uint32_t rate;  // rate / scale == samples per second
    uint32_t start = 0;
    uint32_t length;
    uint32_t suggested_buffer_size;
    uint32_t quality = 0xFFFFFFFF;
    uint32_t sample_size = 0;
    uint16_t frame[4] = {};
};

struct BitmapInfoHeader {
    uint32_t size = sizeof(BitmapInfoHeader);
    int32_t width;
    int32_t height;
    uint16_t planes = 1;
    uint16_t bit_count = 24;
    uint32_t compression;
    uint32_t size_image;
    int32_t x_pels_per_meter = 0;
    int32_t y_pels_per_meter = 0;
    uint32_t clr_used = 0;
    uint32_t clr_important = 0;
};

struct PcmWaveFormat {
    uint16_t format_tag = 1;  // WAVE_FORMAT_PCM
    uint16_t channels;
    uint32_t samples_per_sec;
    uint32_t avg_bytes_per_sec;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
};

struct AviIndexEntry {
    uint32_t chunk_id;
    uint32_t flags = 0x10;  // AVIIF_KEYFRAME
    uint32_t offset;
    uint32_t size;
};
#pragma pack(pop)

uint32_t make_fourcc(const char (&id)[5]) {
    uint32_t fourcc;
    std::memcpy(&fourcc, id, 4);
    return fourcc;
}

void append(std::vector<uint8_t>& buf, const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), bytes, bytes + size);
}

template <typename T>
void append_struct(std::vector<uint8_t>& buf, const T& value) {
    append(buf, &value, sizeof(T));
}

void append_chunk_header(std::vector<uint8_t>& buf, const char (&id)[5], uint32_t size) {
    append(buf, id, 4);
    append_struct(buf, size);
}

void append_list_header(std::vector<uint8_t>& buf, const char (&type)[5], uint32_t content_size) {
    append_chunk_header(buf, "LIST", content_size);
    append(buf, type, 4);
}

std::string batch_filename(const std::string& filename, size_t batch_size, size_t batch_idx, const char* default_ext) {
    if (batch_size == 1)
        return filename;
    std::filesystem::path p(filename);
    std::string ext = p.has_extension() ? p.extension().string() : default_ext;
    return (p.parent_path() / (p.stem().string() + "_b" + std::to_string(batch_idx) + ext)).string();
}

}  // namespace

void save_video(const std::string& filename,
                const ov::Tensor& video_tensor,  // [B, F, H, W, C], u8
                float fps) {
    const ov::Shape shape = video_tensor.get_shape();

    if (shape.empty() || video_tensor.get_size() == 0) {
        throw std::runtime_error("save_video(): input tensor is empty, skip saving: " + filename);
    }

    const size_t B = shape[0], F = shape[1], H = shape[2], W = shape[3], C = shape[4];
    const uint8_t* video_data = video_tensor.data<const uint8_t>();

    for (size_t b = 0; b < B; ++b) {
        std::string out = batch_filename(filename, B, b, ".avi");

        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::VideoWriter writer(out, cv::CAP_OPENCV_MJPEG, fourcc, static_cast<double>(fps), cv::Size(W, H), true);
        if (!writer.isOpened())
            throw std::runtime_error("VideoWriter failed to open: " + out);

        const size_t frame_bytes = H * W * C;
        const size_t batch_stride = F * frame_bytes;
        const uint8_t* batch_ptr = video_data + b * batch_stride;

        for (size_t f = 0; f < F; ++f) {
            const uint8_t* frame_ptr = batch_ptr + f * frame_bytes;

            cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t*>(frame_ptr));
            cv::Mat bgr;
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);

            writer.write(bgr);
        }
    }
}

void save_video_with_audio(const std::string& filename,
                           const ov::Tensor& video_tensor,  // [B, F, H, W, C], u8
                           const ov::Tensor& audio_tensor,  // [B, C, S], f32
                           float fps,
                           uint32_t sample_rate) {
    const ov::Shape video_shape = video_tensor.get_shape();
    const ov::Shape audio_shape = audio_tensor.get_shape();

    if (video_shape.size() != 5 || video_tensor.get_size() == 0)
        throw std::runtime_error("save_video_with_audio(): video tensor is empty, skip saving: " + filename);
    if (audio_shape.size() != 3 || audio_tensor.get_size() == 0)
        throw std::runtime_error("save_video_with_audio(): audio tensor is empty, skip saving: " + filename);
    if (video_shape[0] != audio_shape[0])
        throw std::runtime_error("save_video_with_audio(): video and audio batch sizes differ");

    const size_t B = video_shape[0], F = video_shape[1], H = video_shape[2], W = video_shape[3], C = video_shape[4];
    const size_t AC = audio_shape[1], S = audio_shape[2];
    const uint8_t* video_data = video_tensor.data<const uint8_t>();
    const float* audio_data = audio_tensor.data<const float>();

    const uint16_t block_align = static_cast<uint16_t>(AC * sizeof(int16_t));

    for (size_t b = 0; b < B; ++b) {
        std::string out = batch_filename(filename, B, b, ".avi");

        const size_t frame_bytes = H * W * C;
        const uint8_t* batch_ptr = video_data + b * F * frame_bytes;

        const float* audio_ptr = audio_data + b * AC * S;
        std::vector<int16_t> pcm(S * AC);
        for (size_t s = 0; s < S; ++s)
            for (size_t c = 0; c < AC; ++c)
                pcm[s * AC + c] = static_cast<int16_t>(std::clamp(audio_ptr[c * S + s], -1.0f, 1.0f) * 32767.0f);

        // Interleave one video chunk ("00dc") and one audio chunk ("01wb") per frame.
        std::vector<uint8_t> movi;
        std::vector<AviIndexEntry> index;
        uint32_t max_video_chunk = 0;
        uint32_t max_audio_chunk = 0;

        auto append_movi_chunk = [&](const char (&id)[5], const uint8_t* data, uint32_t size) {
            AviIndexEntry entry{};
            entry.chunk_id = make_fourcc(id);
            entry.offset = static_cast<uint32_t>(movi.size()) + 4;
            entry.size = size;
            index.push_back(entry);

            append_chunk_header(movi, id, size);
            append(movi, data, size);
            if (size % 2 != 0)
                movi.push_back(0);
        };

        cv::Mat bgr;
        std::vector<uint8_t> jpeg;
        for (size_t f = 0; f < F; ++f) {
            cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t*>(batch_ptr + f * frame_bytes));
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
            if (!cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 95}))
                throw std::runtime_error("Failed to encode video frame for: " + out);
            max_video_chunk = std::max<uint32_t>(max_video_chunk, jpeg.size());
            append_movi_chunk("00dc", jpeg.data(), static_cast<uint32_t>(jpeg.size()));

            const size_t sample_begin = f * S / F, sample_end = (f + 1) * S / F;
            const uint32_t audio_bytes = static_cast<uint32_t>((sample_end - sample_begin) * block_align);
            append_movi_chunk("01wb", reinterpret_cast<const uint8_t*>(pcm.data() + sample_begin * AC), audio_bytes);
            max_audio_chunk = std::max(max_audio_chunk, audio_bytes);
        }

        MainAviHeader avih{};
        avih.microsec_per_frame = static_cast<uint32_t>(1e6f / fps + 0.5f);
        avih.total_frames = static_cast<uint32_t>(F);
        avih.suggested_buffer_size = max_video_chunk;
        avih.width = static_cast<uint32_t>(W);
        avih.height = static_cast<uint32_t>(H);

        AviStreamHeader video_strh{};
        video_strh.fcc_type = make_fourcc("vids");
        video_strh.fcc_handler = make_fourcc("MJPG");
        video_strh.scale = 1000;
        video_strh.rate = static_cast<uint32_t>(fps * 1000 + 0.5f);
        video_strh.length = static_cast<uint32_t>(F);
        video_strh.suggested_buffer_size = max_video_chunk;
        video_strh.frame[2] = static_cast<uint16_t>(W);
        video_strh.frame[3] = static_cast<uint16_t>(H);

        BitmapInfoHeader video_strf{};
        video_strf.width = static_cast<int32_t>(W);
        video_strf.height = static_cast<int32_t>(H);
        video_strf.compression = make_fourcc("MJPG");
        video_strf.size_image = static_cast<uint32_t>(W * H * 3);

        AviStreamHeader audio_strh{};
        audio_strh.fcc_type = make_fourcc("auds");
        audio_strh.scale = block_align;
        audio_strh.rate = sample_rate * block_align;
        audio_strh.length = static_cast<uint32_t>(S);
        audio_strh.suggested_buffer_size = max_audio_chunk;
        audio_strh.sample_size = block_align;

        PcmWaveFormat audio_strf{};
        audio_strf.channels = static_cast<uint16_t>(AC);
        audio_strf.samples_per_sec = sample_rate;
        audio_strf.avg_bytes_per_sec = sample_rate * block_align;
        audio_strf.block_align = block_align;

        const uint32_t video_strl_size = 4 + 8 + sizeof(video_strh) + 8 + sizeof(video_strf);
        const uint32_t audio_strl_size = 4 + 8 + sizeof(audio_strh) + 8 + sizeof(audio_strf);
        const uint32_t hdrl_size = 4 + 8 + sizeof(avih) + 8 + video_strl_size + 8 + audio_strl_size;
        const uint32_t movi_size = 4 + static_cast<uint32_t>(movi.size());
        const uint32_t idx1_size = static_cast<uint32_t>(index.size() * sizeof(AviIndexEntry));
        const uint32_t riff_size = 4 + 8 + hdrl_size + 8 + movi_size + 8 + idx1_size;

        std::vector<uint8_t> header;
        append_chunk_header(header, "RIFF", riff_size);
        append(header, "AVI ", 4);
        append_list_header(header, "hdrl", hdrl_size);
        append_chunk_header(header, "avih", sizeof(avih));
        append_struct(header, avih);
        append_list_header(header, "strl", video_strl_size);
        append_chunk_header(header, "strh", sizeof(video_strh));
        append_struct(header, video_strh);
        append_chunk_header(header, "strf", sizeof(video_strf));
        append_struct(header, video_strf);
        append_list_header(header, "strl", audio_strl_size);
        append_chunk_header(header, "strh", sizeof(audio_strh));
        append_struct(header, audio_strh);
        append_chunk_header(header, "strf", sizeof(audio_strf));
        append_struct(header, audio_strf);
        append_list_header(header, "movi", movi_size);

        std::vector<uint8_t> idx1;
        append_chunk_header(idx1, "idx1", idx1_size);
        for (const AviIndexEntry& entry : index)
            append_struct(idx1, entry);

        std::ofstream file(out, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open AVI file for writing: " + out);
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        file.write(reinterpret_cast<const char*>(movi.data()), movi.size());
        file.write(reinterpret_cast<const char*>(idx1.data()), idx1.size());
        if (!file)
            throw std::runtime_error("Failed to write AVI file: " + out);
    }
}
