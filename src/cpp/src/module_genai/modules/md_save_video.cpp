// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_save_video.hpp"

#include "module_genai/module_factory.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <vector>
#include <stdexcept>

// stb_image_write for JPEG encoding
// Include in anonymous namespace to avoid symbol conflicts
namespace {
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "../utils/stb_image_write.h"
}  // anonymous namespace

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(SaveVideoModule);

void SaveVideoModule::print_static_config() {
    std::cout << R"(
  save_video:          # Module Name
    type: "SaveVideoModule"
    description: "Save video tensor to AVI file. Input: [B, F, H, W, C] tensor. Supported DataType: [OVTensor]"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "saved_video"
        type: "String"
      - name: "saved_videos"
        type: "VecString"
    params:
      filename_prefix: "String"       # Default: "output"
      output_folder: "String"         # Default: "./output"
      fps: "Int"                      # Default: 25
      quality: "Int"                  # JPEG quality 0-100, Default: 85
      convert_bgr2rgb: "Bool"         # Default: false
    )" << std::endl;
}

namespace {

// ------------------------------
// Small helpers for little-endian I/O
// ------------------------------
struct LEWriter {
    std::ofstream out;

    explicit LEWriter(const std::string& path) {
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open output file: " + path);
    }

    ~LEWriter() {
        if (out.is_open()) {
            out.close();
        }
    }

    uint32_t tell() { return static_cast<uint32_t>(out.tellp()); }

    void seek(uint32_t pos) {
        out.seekp(pos, std::ios::beg);
        if (!out) throw std::runtime_error("seek failed");
    }

    void write_bytes(const void* data, size_t n) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
        if (!out) throw std::runtime_error("write failed");
    }

    void write_u16(uint16_t v) {
        uint8_t b[2] = {
            static_cast<uint8_t>((v >> 0) & 0xFF),
            static_cast<uint8_t>((v >> 8) & 0xFF)
        };
        write_bytes(b, 2);
    }

    void write_u32(uint32_t v) {
        uint8_t b[4] = {
            static_cast<uint8_t>((v >> 0) & 0xFF),
            static_cast<uint8_t>((v >> 8) & 0xFF),
            static_cast<uint8_t>((v >> 16) & 0xFF),
            static_cast<uint8_t>((v >> 24) & 0xFF)
        };
        write_bytes(b, 4);
    }

    void write_i32(int32_t v) { write_u32(static_cast<uint32_t>(v)); }

    void write_fourcc(const char tag[4]) { write_bytes(tag, 4); }

    void patch_u32(uint32_t pos, uint32_t v) {
        auto cur = tell();
        seek(pos);
        write_u32(v);
        seek(cur);
    }

    bool good() const { return out.good(); }
};

// ------------------------------
// AVI Index Entry
// ------------------------------
struct AVIIndexEntry {
    uint32_t offset;  // relative to start of 'movi' data
    uint32_t size;    // data size (excluding padding)
};

// ------------------------------
// AVI MJPEG Writer
// ------------------------------
class AVIMJPEGWriter {
public:
    AVIMJPEGWriter(const std::string& path, uint32_t width, uint32_t height, uint32_t fps)
        : w(width), h(height), m_fps(fps), writer(path) {
        begin_file();
    }

    ~AVIMJPEGWriter() {
        try {
            if (!finalized) finish();
        } catch (...) {
            // avoid throwing in dtor
        }
    }

    // Adds a single JPEG-compressed frame as an MJPEG chunk.
    // 'rgb' must be w*h*3 (RGB24).
    void add_rgb_frame_as_jpeg(const uint8_t* rgb, int quality = 85) {
        // Encode to JPEG in-memory using stb
        std::vector<uint8_t> jpeg;
        jpeg.reserve(w * h / 2);  // heuristic
        stbi_write_jpg_to_func(&stb_callback, &jpeg, static_cast<int>(w), static_cast<int>(h),
                               3, rgb, quality);

        // Write '00dc' chunk into 'movi' list
        const uint32_t chunk_start = writer.tell();
        writer.write_fourcc("00dc");
        writer.write_u32(static_cast<uint32_t>(jpeg.size()));
        writer.write_bytes(jpeg.data(), jpeg.size());
        if (jpeg.size() & 1) {
            const uint8_t pad = 0;
            writer.write_bytes(&pad, 1);  // word-align
        }

        // Record for idx1
        AVIIndexEntry e;
        e.offset = chunk_start - movi_data_start;
        e.size = static_cast<uint32_t>(jpeg.size());
        index.push_back(e);

        ++frame_count;
        if (jpeg.size() > max_chunk) max_chunk = static_cast<uint32_t>(jpeg.size());
    }

    void finish() {
        if (finalized) return;

        // Close 'movi' list
        const uint32_t movi_end = writer.tell();
        const uint32_t movi_size = movi_end - (movi_list_start + 8);
        writer.patch_u32(movi_list_size_pos, movi_size);

        // Write 'idx1'
        writer.write_fourcc("idx1");
        writer.write_u32(static_cast<uint32_t>(index.size()) * 16);
        for (const auto& e : index) {
            writer.write_fourcc("00dc");
            writer.write_u32(0x00000010);  // AVIIF_KEYFRAME
            writer.write_u32(e.offset);
            writer.write_u32(e.size);
        }

        const uint32_t file_end = writer.tell();

        // Patch 'hdrl' list size
        const uint32_t hdrl_size = hdrl_end - (hdrl_list_start + 8);
        writer.patch_u32(hdrl_list_size_pos, hdrl_size);

        // Patch 'RIFF' size
        const uint32_t riff_size = file_end - (riff_start + 8);
        writer.patch_u32(riff_size_pos, riff_size);

        // Patch avih
        writer.patch_u32(avih_data_pos + 16, frame_count);
        writer.patch_u32(avih_data_pos + 28, max_chunk);
        writer.patch_u32(avih_data_pos + 4, m_fps * max_chunk);

        // Patch strh
        writer.patch_u32(strh_data_pos + 32, frame_count);
        writer.patch_u32(strh_data_pos + 36, max_chunk);

        finalized = true;
    }

    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    uint32_t frame_rate() const { return m_fps; }

private:
    uint32_t w, h, m_fps;
    LEWriter writer;

    // Offsets for patching
    uint32_t riff_start = 0;
    uint32_t riff_size_pos = 0;
    uint32_t hdrl_list_start = 0;
    uint32_t hdrl_list_size_pos = 0;
    uint32_t hdrl_end = 0;
    uint32_t avih_data_pos = 0;
    uint32_t strh_data_pos = 0;
    uint32_t movi_list_start = 0;
    uint32_t movi_list_size_pos = 0;
    uint32_t movi_data_start = 0;

    std::vector<AVIIndexEntry> index;
    uint32_t frame_count = 0;
    uint32_t max_chunk = 0;
    bool finalized = false;

    static void stb_callback(void* ctx, void* data, int size) {
        auto* vec = reinterpret_cast<std::vector<uint8_t>*>(ctx);
        auto* p = reinterpret_cast<const uint8_t*>(data);
        vec->insert(vec->end(), p, p + size);
    }

    void begin_file() {
        // RIFF header
        riff_start = writer.tell();
        writer.write_fourcc("RIFF");
        riff_size_pos = writer.tell();
        writer.write_u32(0);  // placeholder
        writer.write_fourcc("AVI ");

        // LIST 'hdrl'
        hdrl_list_start = writer.tell();
        writer.write_fourcc("LIST");
        hdrl_list_size_pos = writer.tell();
        writer.write_u32(0);  // placeholder
        writer.write_fourcc("hdrl");

        // 'avih' (MainAVIHeader) 56 bytes
        writer.write_fourcc("avih");
        writer.write_u32(56);
        avih_data_pos = writer.tell();

        const uint32_t usec_per_frame = 1000000u / m_fps;
        writer.write_u32(usec_per_frame);
        writer.write_u32(0);           // dwMaxBytesPerSec (patch later)
        writer.write_u32(0);           // dwPaddingGranularity
        writer.write_u32(0x00000010);  // dwFlags (AVIF_HASINDEX)
        writer.write_u32(0);           // dwTotalFrames (patch later)
        writer.write_u32(0);           // dwInitialFrames
        writer.write_u32(1);           // dwStreams
        writer.write_u32(0);           // dwSuggestedBufferSize (patch later)
        writer.write_u32(w);           // dwWidth
        writer.write_u32(h);           // dwHeight
        writer.write_u32(0); writer.write_u32(0); writer.write_u32(0); writer.write_u32(0);  // dwReserved[4]

        // LIST 'strl'
        const uint32_t strl_list_start = writer.tell();
        writer.write_fourcc("LIST");
        const uint32_t strl_list_size_pos = writer.tell();
        writer.write_u32(0);  // placeholder
        writer.write_fourcc("strl");

        // 'strh' (AVISTREAMHEADER) 56 bytes
        writer.write_fourcc("strh");
        writer.write_u32(56);
        strh_data_pos = writer.tell();

        writer.write_fourcc("vids");
        writer.write_fourcc("MJPG");
        writer.write_u32(0);           // dwFlags
        writer.write_u16(0);           // wPriority
        writer.write_u16(0);           // wLanguage
        writer.write_u32(0);           // dwInitialFrames
        writer.write_u32(1);           // dwScale
        writer.write_u32(m_fps);       // dwRate
        writer.write_u32(0);           // dwStart
        writer.write_u32(0);           // dwLength (patch later)
        writer.write_u32(0);           // dwSuggestedBufferSize (patch later)
        writer.write_u32(0xFFFFFFFFu); // dwQuality
        writer.write_u32(0);           // dwSampleSize

        // rcFrame
        writer.write_u16(0);
        writer.write_u16(0);
        writer.write_u16(static_cast<uint16_t>(w));
        writer.write_u16(static_cast<uint16_t>(h));

        // 'strf' (BITMAPINFOHEADER) 40 bytes
        writer.write_fourcc("strf");
        writer.write_u32(40);
        writer.write_u32(40);          // biSize
        writer.write_i32(static_cast<int32_t>(w));
        writer.write_i32(static_cast<int32_t>(h));
        writer.write_u16(1);           // biPlanes
        writer.write_u16(24);          // biBitCount
        writer.write_fourcc("MJPG");   // biCompression
        writer.write_u32(0);           // biSizeImage
        writer.write_i32(0);           // biXPelsPerMeter
        writer.write_i32(0);           // biYPelsPerMeter
        writer.write_u32(0);           // biClrUsed
        writer.write_u32(0);           // biClrImportant

        // Patch strl LIST size
        const uint32_t strl_end = writer.tell();
        const uint32_t strl_size = strl_end - (strl_list_start + 8);
        writer.patch_u32(strl_list_size_pos, strl_size);

        hdrl_end = writer.tell();

        // LIST 'movi'
        movi_list_start = writer.tell();
        writer.write_fourcc("LIST");
        movi_list_size_pos = writer.tell();
        writer.write_u32(0);  // placeholder
        writer.write_fourcc("movi");
        movi_data_start = writer.tell();
    }
};

// Helper to generate batch filename
static inline std::string batch_name(const std::string& base, size_t b) {
    const auto dot = base.find_last_of('.');
    if (dot == std::string::npos) return base + "_b" + std::to_string(b) + ".avi";
    return base.substr(0, dot) + "_b" + std::to_string(b) + base.substr(dot);
}

// Convert frame data to RGB u8 format
static inline void pack_to_rgb_u8(const uint8_t* src, size_t H, size_t W, size_t C,
                                  bool bgr2rgb, std::vector<uint8_t>& out_rgb) {
    const size_t HW = H * W;
    out_rgb.resize(HW * 3);

    if (C == 3 && !bgr2rgb) {
        std::memcpy(out_rgb.data(), src, HW * 3);
        return;
    }

    uint8_t* dst = out_rgb.data();
    if (C == 1) {  // gray to RGB
        for (size_t i = 0; i < HW; ++i) {
            const uint8_t g = src[i];
            *dst++ = g; *dst++ = g; *dst++ = g;
        }
        return;
    }

    // C >= 3: treat as RGBX/BGRX
    const uint8_t* s = src;
    if (!bgr2rgb) {
        for (size_t i = 0; i < HW; ++i) { *dst++ = s[0]; *dst++ = s[1]; *dst++ = s[2]; s += C; }
    } else {
        for (size_t i = 0; i < HW; ++i) { *dst++ = s[2]; *dst++ = s[1]; *dst++ = s[0]; s += C; }
    }
}

}  // anonymous namespace


SaveVideoModule::SaveVideoModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize SaveVideoModule");
    }
}

SaveVideoModule::~SaveVideoModule() {}

bool SaveVideoModule::initialize() {
    const auto& params = module_desc->params;

    // Get filename_prefix parameter
    auto it_prefix = params.find("filename_prefix");
    if (it_prefix != params.end()) {
        m_filename_prefix = it_prefix->second;
    } else {
        m_filename_prefix = "output";
    }

    // Get output_folder parameter
    auto it_folder = params.find("output_folder");
    if (it_folder != params.end()) {
        m_output_folder = it_folder->second;
    } else {
        m_output_folder = "./output";
    }

    // Get fps parameter
    auto it_fps = params.find("fps");
    if (it_fps != params.end()) {
        m_fps = static_cast<uint32_t>(std::stoi(it_fps->second));
    } else {
        m_fps = 25;
    }

    // Get quality parameter
    auto it_quality = params.find("quality");
    if (it_quality != params.end()) {
        m_quality = std::stoi(it_quality->second);
        m_quality = std::max(0, std::min(100, m_quality));
    } else {
        m_quality = 85;
    }

    // Get convert_bgr2rgb parameter
    auto it_bgr2rgb = params.find("convert_bgr2rgb");
    if (it_bgr2rgb != params.end()) {
        m_convert_bgr2rgb = (it_bgr2rgb->second == "true" || it_bgr2rgb->second == "1");
    } else {
        m_convert_bgr2rgb = false;
    }

    // Create output folder if it doesn't exist
    std::filesystem::path output_path(m_output_folder);
    if (!std::filesystem::exists(output_path)) {
        try {
            std::filesystem::create_directories(output_path);
        } catch (const std::exception& e) {
            GENAI_ERR("SaveVideoModule[" + module_desc->name + "]: Failed to create output folder: " + e.what());
            return false;
        }
    }

    return true;
}

std::string SaveVideoModule::generate_filename() {
    size_t seq_num = m_sequence_number.fetch_add(1);

    // Format: prefix_YYYYMMDD_HHMMSS_seqnum.avi
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", &tm_now);

    std::ostringstream filename;
    filename << m_filename_prefix << "_" << time_buffer << "_" << std::setfill('0') << std::setw(5) << seq_num << ".avi";

    std::filesystem::path full_path = std::filesystem::path(m_output_folder) / filename.str();
    return full_path.string();
}

std::vector<std::string> SaveVideoModule::save_tensor_as_video(const ov::Tensor& tensor, const std::string& filepath) {
    std::vector<std::string> saved_paths;
    auto shape = tensor.get_shape();
    auto element_type = tensor.get_element_type();

    // Expect tensor shape to be [B, F, H, W, C] (5D video tensor)
    if (shape.size() != 5) {
        GENAI_ERR("SaveVideoModule: Expected 5D tensor [B, F, H, W, C], got " + std::to_string(shape.size()) + "D");
        return saved_paths;
    }

    const size_t B = shape[0];  // Batch
    const size_t F = shape[1];  // Frames
    const size_t H = shape[2];  // Height
    const size_t W = shape[3];  // Width
    const size_t C = shape[4];  // Channels

    if (!(C == 1 || C == 3 || C == 4)) {
        GENAI_ERR("SaveVideoModule: Unsupported number of channels: " + std::to_string(C) + ". Expected 1, 3, or 4.");
        return saved_paths;
    }

    // Currently only support uint8 tensor
    if (element_type != ov::element::u8) {
        GENAI_ERR("SaveVideoModule: Unsupported tensor element type: " + element_type.get_type_name() + ". Expected u8.");
        return saved_paths;
    }

    const size_t elems_per_frame = H * W * C;
    const uint8_t* base = tensor.data<uint8_t>();

    std::vector<uint8_t> rgb;
    for (size_t b = 0; b < B; ++b) {
        std::string current_filepath = filepath;
        if (B > 1) {
            current_filepath = batch_name(filepath, b);
        }

        try {
            AVIMJPEGWriter avi(current_filepath, static_cast<uint32_t>(W), static_cast<uint32_t>(H), m_fps);

            const uint8_t* batch_ptr = base + b * F * elems_per_frame;
            for (size_t f = 0; f < F; ++f) {
                const uint8_t* frame_ptr = batch_ptr + f * elems_per_frame;
                pack_to_rgb_u8(frame_ptr, H, W, C, m_convert_bgr2rgb, rgb);
                avi.add_rgb_frame_as_jpeg(rgb.data(), m_quality);
            }

            avi.finish();
            saved_paths.push_back(current_filepath);
            GENAI_INFO("SaveVideoModule: Saved video to: " + current_filepath + " (" +
                       std::to_string(F) + " frames, " + std::to_string(W) + "x" + std::to_string(H) +
                       " @ " + std::to_string(m_fps) + " fps)");
        } catch (const std::exception& e) {
            GENAI_ERR("SaveVideoModule: Failed to save video to: " + current_filepath + " - " + e.what());
            return saved_paths;
        }
    }

    return saved_paths;
}

void SaveVideoModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    if (this->inputs.find("raw_data") == this->inputs.end()) {
        GENAI_ERR("SaveVideoModule[" + module_desc->name + "]: 'raw_data' input not found");
        return;
    }

    auto& raw_data = this->inputs["raw_data"].data;
    std::vector<std::string> saved_filepaths;

    if (raw_data.is<ov::Tensor>()) {
        ov::Tensor tensor = raw_data.as<ov::Tensor>();
        std::string filepath = generate_filename();

        auto paths = save_tensor_as_video(tensor, filepath);
        if (paths.empty()) {
            GENAI_ERR("SaveVideoModule[" + module_desc->name + "]: Failed to save video");
        } else {
            saved_filepaths.insert(saved_filepaths.end(), paths.begin(), paths.end());
        }
    } else if (raw_data.is<std::vector<ov::Tensor>>()) {
        auto tensors = raw_data.as<std::vector<ov::Tensor>>();
        for (size_t i = 0; i < tensors.size(); i++) {
            std::string filepath = generate_filename();
            auto paths = save_tensor_as_video(tensors[i], filepath);
            if (paths.empty()) {
                GENAI_ERR("SaveVideoModule[" + module_desc->name + "]: Failed to save video " + std::to_string(i));
            } else {
                saved_filepaths.insert(saved_filepaths.end(), paths.begin(), paths.end());
            }
        }
    } else {
        GENAI_ERR("SaveVideoModule[" + module_desc->name + "]: Unsupported data type. Expected OVTensor or VecOVTensor.");
    }

    // Set outputs
    if (!saved_filepaths.empty()) {
        this->outputs["saved_video"].data = saved_filepaths[0];
        this->outputs["saved_videos"].data = saved_filepaths;

        GENAI_INFO("SaveVideoModule[" + module_desc->name + "]: Output 'saved_video' = " + saved_filepaths[0]);
        if (saved_filepaths.size() > 1) {
            GENAI_INFO("SaveVideoModule[" + module_desc->name + "]: Output 'saved_videos' contains " +
                       std::to_string(saved_filepaths.size()) + " files");
        }
    }
}

} // namespace module
} // namespace genai
} // namespace ov
