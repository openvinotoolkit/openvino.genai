// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/rag/embedding_pipeline.hpp>
#include "../visual_language_chat/load_image.hpp"
#include "../visual_language_chat/load_video.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
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

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
            " <MODEL_DIR> --query <QUERY> --images <IMAGE_FILE OR DIR_WITH_IMAGES> "
            "--videos <VIDEO_FILE OR DIR_WITH_VIDEOS> "
            "[--num-video-frames 8] [--device <DEVICE>]");
    }

    const std::string model_dir = argv[1];

    std::string device = "CPU";
    std::optional<std::string> query;
    std::optional<fs::path> images_path;
    std::optional<fs::path> videos_path;
    size_t num_video_frames = 8;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--images" && i + 1 < argc) {
            images_path = argv[++i];
        } else if (arg == "--videos" && i + 1 < argc) {
            videos_path = argv[++i];
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

    if (!images_path.has_value() && !videos_path.has_value()) {
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

    const std::vector<ov::Tensor> images = images_path.has_value()
        ? utils::load_images(*images_path)
        : std::vector<ov::Tensor>{};
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
        const std::string path = images_path->string() + "#" + std::to_string(image_idx);
        const ov::Tensor& image = images[image_idx];
        ov::genai::EmbedResult result = pipeline.embed(std::string{}, {image});
        const float* emb = result.embeddings.data<const float>();
        results.push_back({cosine_similarity(query_vec, emb, embed_dim), "image", path});
    }

    const auto [videos, videos_metadata] = videos_path.has_value()
        ? utils::load_videos(*videos_path, num_video_frames)
        : std::pair<std::vector<ov::Tensor>, std::vector<ov::genai::VideoMetadata>>{};
    for (size_t video_idx = 0; video_idx < videos.size(); ++video_idx) {
        const std::string path = videos_path->string() + "#" + std::to_string(video_idx);
        ov::genai::EmbedResult result =
            pipeline.embed(std::string{}, {}, {videos[video_idx]}, {videos_metadata[video_idx]});
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
