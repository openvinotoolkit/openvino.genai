// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/rag/embedding_pipeline.hpp>
#include <openvino/genai/visual_language/video_metadata.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

float cosine_similarity(const float* a, const float* b, size_t size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f)
        return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

ov::Tensor load_image(const fs::path& image_path) {
    cv::Mat bgr = cv::imread(image_path.string());
    if (bgr.empty()) {
        OPENVINO_THROW("Failed to load image: " + image_path.string());
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    ov::Tensor tensor(ov::element::u8, {size_t(rgb.rows), size_t(rgb.cols), 3});
    std::memcpy(tensor.data(), rgb.data, rgb.total() * 3);
    return tensor;
}

std::vector<size_t> sample_frame_indices(size_t total_frames, size_t num_frames) {
    std::vector<size_t> indices(num_frames);
    if (num_frames == 0) {
        return indices;
    }
    if (num_frames == 1) {
        indices[0] = 0;
        return indices;
    }

    const double step = static_cast<double>(total_frames - 1) / static_cast<double>(num_frames - 1);
    for (size_t i = 0; i < num_frames; ++i) {
        indices[i] = std::min(static_cast<size_t>(std::floor(i * step)), total_frames - 1);
    }
    return indices;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string run_command(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        OPENVINO_THROW("Failed to run command: " + command);
    }
    while (const size_t bytes_read = std::fread(buffer.data(), 1, buffer.size(), pipe)) {
        output.append(buffer.data(), bytes_read);
    }
    const int status = pclose(pipe);
    if (status != 0) {
        OPENVINO_THROW("Command failed: " + command);
    }
    return output;
}

double parse_fps(const std::string& fps_text) {
    const size_t slash_pos = fps_text.find('/');
    if (slash_pos == std::string::npos) {
        return std::stod(fps_text);
    }
    const double numerator = std::stod(fps_text.substr(0, slash_pos));
    const double denominator = std::stod(fps_text.substr(slash_pos + 1));
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

struct VideoInfo {
    size_t width = 0;
    size_t height = 0;
    double fps = 0.0;
    std::optional<size_t> frame_count;
};

VideoInfo probe_video_with_ffprobe(const fs::path& video_path) {
    const std::string command =
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,avg_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1:nokey=1 " +
        shell_quote(video_path.string());
    std::istringstream output(run_command(command));

    std::string width;
    std::string height;
    std::string fps;
    std::string frame_count;
    std::getline(output, width);
    std::getline(output, height);
    std::getline(output, fps);
    std::getline(output, frame_count);

    VideoInfo info;
    info.width = static_cast<size_t>(std::stoul(width));
    info.height = static_cast<size_t>(std::stoul(height));
    info.fps = parse_fps(fps);
    if (!frame_count.empty() && frame_count != "N/A") {
        info.frame_count = static_cast<size_t>(std::stoul(frame_count));
    }
    return info;
}

std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video_with_ffmpeg(const fs::path& video_path, size_t num_frames) {
    const VideoInfo info = probe_video_with_ffprobe(video_path);
    OPENVINO_ASSERT(info.width > 0 && info.height > 0, "Failed to read video dimensions: ", video_path.string());

    const std::string command = "ffmpeg -v error -i " + shell_quote(video_path.string()) + " -f rawvideo -pix_fmt rgb24 -";
    const std::string raw_frames = run_command(command);
    const size_t frame_size = info.width * info.height * 3;
    OPENVINO_ASSERT(frame_size > 0 && raw_frames.size() % frame_size == 0,
                    "Unexpected raw frame data size from ffmpeg for video: ",
                    video_path.string());
    const size_t total_frames = raw_frames.size() / frame_size;
    OPENVINO_ASSERT(total_frames > 0, "Failed to read frames from video: ", video_path.string());
    if (info.frame_count.has_value()) {
        OPENVINO_ASSERT(*info.frame_count == total_frames,
                        "ffprobe frame count (",
                        *info.frame_count,
                        ") does not match decoded frame count (",
                        total_frames,
                        ")");
    }

    ov::genai::VideoMetadata metadata;
    metadata.fps = info.fps;
    metadata.frames_indices = sample_frame_indices(total_frames, std::min(num_frames, total_frames));

    ov::Tensor video_tensor(ov::element::u8, {total_frames, info.height, info.width, 3});
    std::memcpy(video_tensor.data(), raw_frames.data(), raw_frames.size());
    return {video_tensor, metadata};
}

std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video_with_opencv(cv::VideoCapture& cap,
                                                                       const fs::path& video_path,
                                                                       size_t num_frames) {
    const size_t total_frames = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps          = cap.get(cv::CAP_PROP_FPS);

    if (total_frames == 0) {
        OPENVINO_THROW("Failed to read frames from video: " + video_path.string());
    }

    const auto sampled_indices = sample_frame_indices(total_frames, std::min(num_frames, total_frames));
    ov::genai::VideoMetadata metadata;
    metadata.fps = fps;
    metadata.frames_indices = sampled_indices;

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        frames.push_back(rgb.clone());
    }

    if (frames.empty()) {
        OPENVINO_THROW("No sampled frames collected from video: " + video_path.string());
    }

    ov::Tensor video_tensor(ov::element::u8,
                            {frames.size(),
                             static_cast<size_t>(frames.front().rows),
                             static_cast<size_t>(frames.front().cols),
                             3});
    uint8_t* dst = video_tensor.data<uint8_t>();
    for (const auto& f : frames) {
        std::memcpy(dst, f.data, f.total() * 3);
        dst += f.total() * 3;
    }
    return {video_tensor, metadata};
}

std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video(const fs::path& video_path, size_t num_frames = 8) {
    cv::VideoCapture cap(video_path.string());
    if (cap.isOpened()) {
        return load_video_with_opencv(cap, video_path, num_frames);
    }
    return load_video_with_ffmpeg(video_path, num_frames);
}

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
            " <MODEL_DIR> --query <QUERY> --images <img1> [img2 ...] --videos <vid1> [vid2 ...] "
            "[--num-video-frames 8] [--device <DEVICE>]");
    }

    const std::string model_dir = argv[1];

    std::string device = "CPU";
    std::optional<std::string> query;
    std::vector<std::string> image_paths;
    std::vector<std::string> video_paths;
    size_t num_video_frames = 8;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--images") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                image_paths.push_back(argv[++i]);
            }
        } else if (arg == "--videos") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                video_paths.push_back(argv[++i]);
            }
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--num-video-frames" && i + 1 < argc) {
            num_video_frames = std::stoul(argv[++i]);
        } else if (arg == "--query" && i + 1 < argc) {
            query = argv[++i];
        } else {
            throw std::runtime_error("Unexpected or incomplete argument: " + arg);
        }
    }

    if (!query.has_value()) {
        throw std::runtime_error("--query is required");
    }

    if (image_paths.empty() && video_paths.empty()) {
        throw std::runtime_error("At least one input must be provided via --images or --videos");
    }

    ov::genai::EmbeddingPipeline pipeline(model_dir, device);

    // Embed the text query
    ov::genai::EmbedResult query_result = pipeline.embed(*query);
    const ov::Tensor& query_tensor = query_result.embeddings;
    const size_t embed_dim = query_tensor.get_shape().at(1);
    const float* query_vec = query_tensor.data<const float>();

    struct Entry {
        float score;
        std::string type;
        std::string path;
    };
    std::vector<Entry> results;

    for (const auto& path : image_paths) {
        ov::Tensor image = load_image(path);
        ov::genai::EmbedResult result = pipeline.embed(std::string{}, {image});
        const float* emb = result.embeddings.data<const float>();
        results.push_back({cosine_similarity(query_vec, emb, embed_dim), "image", path});
    }

    for (const auto& path : video_paths) {
        auto [video_tensor, metadata] = load_video(path, num_video_frames);
        ov::genai::EmbedResult result = pipeline.embed(std::string{}, {}, {video_tensor}, {metadata});
        const float* emb = result.embeddings.data<const float>();
        results.push_back({cosine_similarity(query_vec, emb, embed_dim), "video", path});
    }

    std::sort(results.begin(), results.end(), [](const Entry& a, const Entry& b) {
        return a.score > b.score;
    });

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Query: " << *query << "\nRanked inputs by cosine similarity:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << (i + 1) << ". " << results[i].type << ": "
                  << fs::absolute(results[i].path).string()
                  << " similarity=" << results[i].score << "\n";
    }
    const auto& best = results.front();
    std::cout << "Most similar input: " << best.type << " "
              << fs::absolute(best.path).string()
              << " similarity=" << best.score << "\n";

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
