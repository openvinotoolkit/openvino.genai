// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
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

std::string batch_filename(const std::string& filename, size_t batch_size, size_t batch_idx, const char* default_ext) {
    if (batch_size == 1)
        return filename;
    std::filesystem::path p(filename);
    std::string ext = p.has_extension() ? p.extension().string() : default_ext;
    return (p.parent_path() / (p.stem().string() + "_b" + std::to_string(batch_idx) + ext)).string();
}

void append_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>(v >> 8));
}

void append_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void append_fourcc(std::vector<uint8_t>& buf, const char* fourcc) {
    buf.insert(buf.end(), fourcc, fourcc + 4);
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

        std::vector<std::vector<uint8_t>> jpeg_frames(F);
        uint32_t max_video_chunk = 0;
        for (size_t f = 0; f < F; ++f) {
            cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t*>(batch_ptr + f * frame_bytes));
            cv::Mat bgr;
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
            if (!cv::imencode(".jpg", bgr, jpeg_frames[f], {cv::IMWRITE_JPEG_QUALITY, 95}))
                throw std::runtime_error("Failed to encode video frame for: " + out);
            max_video_chunk = std::max<uint32_t>(max_video_chunk, jpeg_frames[f].size());
        }

        const float* audio_ptr = audio_data + b * AC * S;
        std::vector<int16_t> pcm(S * AC);
        for (size_t s = 0; s < S; ++s)
            for (size_t c = 0; c < AC; ++c)
                pcm[s * AC + c] = static_cast<int16_t>(std::clamp(audio_ptr[c * S + s], -1.0f, 1.0f) * 32767.0f);

        // Interleave one video chunk ("00dc") and one audio chunk ("01wb") per frame.
        std::vector<uint8_t> movi;
        struct IndexEntry { const char* fourcc; uint32_t offset, size; };
        std::vector<IndexEntry> index;
        uint32_t max_audio_chunk = 0;

        auto append_chunk = [&](const char* fourcc, const uint8_t* data, uint32_t size) {
            index.push_back({fourcc, static_cast<uint32_t>(movi.size()) + 4, size});
            append_fourcc(movi, fourcc);
            append_u32(movi, size);
            movi.insert(movi.end(), data, data + size);
            if (size % 2 != 0)
                movi.push_back(0);
        };

        for (size_t f = 0; f < F; ++f) {
            append_chunk("00dc", jpeg_frames[f].data(), static_cast<uint32_t>(jpeg_frames[f].size()));

            const size_t sample_begin = f * S / F, sample_end = (f + 1) * S / F;
            const uint32_t audio_bytes = static_cast<uint32_t>((sample_end - sample_begin) * block_align);
            append_chunk("01wb", reinterpret_cast<const uint8_t*>(pcm.data() + sample_begin * AC), audio_bytes);
            max_audio_chunk = std::max(max_audio_chunk, audio_bytes);
        }

        const uint32_t avih_size = 56, strh_size = 56, video_strf_size = 40, audio_strf_size = 16;
        const uint32_t video_strl_size = 4 + 8 + strh_size + 8 + video_strf_size;
        const uint32_t audio_strl_size = 4 + 8 + strh_size + 8 + audio_strf_size;
        const uint32_t hdrl_size = 4 + 8 + avih_size + 8 + video_strl_size + 8 + audio_strl_size;
        const uint32_t movi_size = 4 + static_cast<uint32_t>(movi.size());
        const uint32_t idx1_size = static_cast<uint32_t>(index.size()) * 16;
        const uint32_t riff_size = 4 + 8 + hdrl_size + 8 + movi_size + 8 + idx1_size;

        std::vector<uint8_t> header;
        append_fourcc(header, "RIFF");
        append_u32(header, riff_size);
        append_fourcc(header, "AVI ");

        append_fourcc(header, "LIST");
        append_u32(header, hdrl_size);
        append_fourcc(header, "hdrl");

        append_fourcc(header, "avih");
        append_u32(header, avih_size);
        append_u32(header, static_cast<uint32_t>(1e6f / fps + 0.5f));  // dwMicroSecPerFrame
        append_u32(header, 0);                                         // dwMaxBytesPerSec
        append_u32(header, 0);                                         // dwPaddingGranularity
        append_u32(header, 0x110);                                     // AVIF_HASINDEX | AVIF_ISINTERLEAVED
        append_u32(header, static_cast<uint32_t>(F));                  // dwTotalFrames
        append_u32(header, 0);                                         // dwInitialFrames
        append_u32(header, 2);                                         // dwStreams
        append_u32(header, max_video_chunk);                           // dwSuggestedBufferSize
        append_u32(header, static_cast<uint32_t>(W));
        append_u32(header, static_cast<uint32_t>(H));
        for (int i = 0; i < 4; ++i)
            append_u32(header, 0);  // dwReserved

        append_fourcc(header, "LIST");
        append_u32(header, video_strl_size);
        append_fourcc(header, "strl");
        append_fourcc(header, "strh");
        append_u32(header, strh_size);
        append_fourcc(header, "vids");
        append_fourcc(header, "MJPG");
        append_u32(header, 0);  // dwFlags
        append_u16(header, 0);  // wPriority
        append_u16(header, 0);  // wLanguage
        append_u32(header, 0);  // dwInitialFrames
        append_u32(header, 1000);                                       // dwScale
        append_u32(header, static_cast<uint32_t>(fps * 1000 + 0.5f));   // dwRate
        append_u32(header, 0);                                          // dwStart
        append_u32(header, static_cast<uint32_t>(F));                   // dwLength
        append_u32(header, max_video_chunk);                            // dwSuggestedBufferSize
        append_u32(header, 0xFFFFFFFF);                                 // dwQuality
        append_u32(header, 0);                                          // dwSampleSize
        append_u16(header, 0);  // rcFrame
        append_u16(header, 0);
        append_u16(header, static_cast<uint16_t>(W));
        append_u16(header, static_cast<uint16_t>(H));
        append_fourcc(header, "strf");
        append_u32(header, video_strf_size);  // BITMAPINFOHEADER
        append_u32(header, video_strf_size);  // biSize
        append_u32(header, static_cast<uint32_t>(W));
        append_u32(header, static_cast<uint32_t>(H));
        append_u16(header, 1);   // biPlanes
        append_u16(header, 24);  // biBitCount
        append_fourcc(header, "MJPG");                       // biCompression
        append_u32(header, static_cast<uint32_t>(W * H * 3));  // biSizeImage
        for (int i = 0; i < 4; ++i)
            append_u32(header, 0);  // biXPelsPerMeter .. biClrImportant

        append_fourcc(header, "LIST");
        append_u32(header, audio_strl_size);
        append_fourcc(header, "strl");
        append_fourcc(header, "strh");
        append_u32(header, strh_size);
        append_fourcc(header, "auds");
        append_u32(header, 0);  // fccHandler
        append_u32(header, 0);  // dwFlags
        append_u16(header, 0);  // wPriority
        append_u16(header, 0);  // wLanguage
        append_u32(header, 0);  // dwInitialFrames
        append_u32(header, block_align);                // dwScale
        append_u32(header, sample_rate * block_align);  // dwRate
        append_u32(header, 0);                          // dwStart
        append_u32(header, static_cast<uint32_t>(S));   // dwLength
        append_u32(header, max_audio_chunk);            // dwSuggestedBufferSize
        append_u32(header, 0xFFFFFFFF);                 // dwQuality
        append_u32(header, block_align);                // dwSampleSize
        for (int i = 0; i < 4; ++i)
            append_u16(header, 0);  // rcFrame
        append_fourcc(header, "strf");
        append_u32(header, audio_strf_size);  // PCMWAVEFORMAT
        append_u16(header, 1);                // WAVE_FORMAT_PCM
        append_u16(header, static_cast<uint16_t>(AC));
        append_u32(header, sample_rate);
        append_u32(header, sample_rate * block_align);
        append_u16(header, block_align);
        append_u16(header, 16);  // wBitsPerSample

        append_fourcc(header, "LIST");
        append_u32(header, movi_size);
        append_fourcc(header, "movi");

        std::vector<uint8_t> idx1;
        append_fourcc(idx1, "idx1");
        append_u32(idx1, idx1_size);
        for (const IndexEntry& entry : index) {
            append_fourcc(idx1, entry.fourcc);
            append_u32(idx1, 0x10);  // AVIIF_KEYFRAME
            append_u32(idx1, entry.offset);
            append_u32(idx1, entry.size);
        }

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
