// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "audio_utils.hpp"
#include "load_image.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace {
constexpr uint32_t SPEECH_SAMPLE_RATE = 24000;  // Qwen3-Omni speech output is 24kHz mono PCM.
constexpr size_t DEFAULT_VIDEO_FRAMES = 8;

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

// Pick num_frames evenly spaced indices across a video of total_frames.
std::vector<size_t> sample_frame_indices(size_t total_frames, size_t num_frames) {
    std::vector<size_t> indices;
    indices.reserve(num_frames);
    const float step = static_cast<float>(total_frames) / num_frames;
    for (size_t i = 0; i < num_frames; ++i) {
        indices.push_back(std::min(static_cast<size_t>(i * step), total_frames - 1));
    }
    return indices;
}

// Decode a video into an [N, H, W, 3] uint8 tensor and attach sampling metadata.
std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video(const std::filesystem::path& video_path,
                                                           size_t num_frames = DEFAULT_VIDEO_FRAMES) {
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open the video file: " + video_path.string());
    }

    const size_t total_num_frames = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    ov::genai::VideoMetadata video_metadata;
    video_metadata.fps = cap.get(cv::CAP_PROP_FPS);
    // Passing frame indices selects those frames within the pipeline and skips model-specific sampling.
    // Leave frames_indices empty to apply model-specific sampling (e.g. for Qwen3-VL).
    video_metadata.frames_indices = sample_frame_indices(total_num_frames, num_frames);

    const size_t width = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const size_t height = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    ov::Tensor video_tensor(ov::element::u8, ov::Shape{total_num_frames, height, width, 3});
    uint8_t* video_tensor_data = video_tensor.data<uint8_t>();

    cv::Mat frame;
    size_t frame_idx = 0;
    while (cap.read(frame)) {
        OPENVINO_ASSERT(static_cast<size_t>(frame.cols) == width && static_cast<size_t>(frame.rows) == height &&
                            frame.channels() == 3,
                        "Unexpected frame geometry while decoding video");
        std::memcpy(video_tensor_data, frame.data, frame.total() * 3 * sizeof(uint8_t));
        video_tensor_data += frame.total() * 3;
        ++frame_idx;
    }
    OPENVINO_ASSERT(frame_idx == total_num_frames,
                    "Frame count mismatch: expected " + std::to_string(total_num_frames) + ", got " +
                        std::to_string(frame_idx));

    return {std::move(video_tensor), std::move(video_metadata)};
}

// Save the first waveform of a result to a WAV file. Speech output is optional (talker mode),
// so callers pass results whose waveforms may be empty.
void save_speech(const ov::genai::OmniDecodedResults& results, const std::string& file_name) {
    if (results.speech_result.waveforms.empty()) {
        return;
    }
    const ov::Tensor& waveform = results.speech_result.waveforms[0];
    utils::audio::save_to_wav(waveform.data<const float>(),
                              waveform.get_size(),
                              file_name,
                              waveform.get_element_type().bitwidth(),
                              SPEECH_SAMPLE_RATE);
    std::cout << "\n[Speech output saved to \"" << file_name << "\"]" << std::endl;
}
}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE> [VIDEO_FILE]");
    }

    const std::filesystem::path models_path = argv[1];

    // Two configs: text_config drives the thinker text decode, talker_speech_config drives
    // the talker + speech output. Speech output is hardcoded on here to show the multimodal
    // path. Set talker_speech_config.return_audio = false to get text-only responses.
    ov::genai::GenerationConfig text_config;
    text_config.max_new_tokens = 256;

    ov::genai::OmniTalkerSpeechConfig talker_speech_config(models_path);
    talker_speech_config.return_audio = true;
    // Leaving talker_speech_config.speaker empty selects the model's default voice. Available
    // voices vary by checkpoint.

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    ov::Tensor audio_tensor = utils::audio::read_wav_as_tensor(argv[3]);
    std::vector<ov::Tensor> audios = {std::move(audio_tensor)};

    // Compose OmniPipeline from separate VLM and Talker components.
    // This allows independent device selection (e.g., VLM on GPU, Talker on CPU),
    // sharing a VLM base across pipelines, or injecting custom TalkerBase implementations.
    // For simpler use cases: ov::genai::OmniPipeline pipe(models_path, "CPU");

    auto vlm = std::make_shared<ov::genai::VLMPipeline>(models_path, "CPU");
    auto talker = std::make_shared<ov::genai::Talker>(models_path, "CPU");
    ov::genai::OmniPipeline pipe(vlm, talker);

    // Speaker API demo: list available speakers and demonstrate voice blending.
    std::vector<std::string> speakers = talker->list_speakers();
    std::cout << "\n=== Available Speakers ===\n";
    if (speakers.empty()) {
        std::cout << "No named speakers (model has single default voice)\n";
    } else {
        std::cout << "Found " << speakers.size() << " speakers:\n";
        for (const auto& name : speakers) {
            std::cout << "  - " << name << "\n";
        }
        std::cout << "\n";

        if (speakers.size() >= 2) {
            std::cout << "=== Voice Blending Demo ===\n";
            std::cout << "Blending " << speakers[0] << " + " << speakers[1] << " (50/50 mix)\n";
            ov::Tensor emb1 = talker->get_speaker_embedding(speakers[0]);
            ov::Tensor emb2 = talker->get_speaker_embedding(speakers[1]);
            ov::Tensor blended(emb1.get_element_type(), emb1.get_shape());
            const float* data1 = emb1.data<float>();
            const float* data2 = emb2.data<float>();
            float* blended_data = blended.data<float>();
            for (size_t i = 0; i < emb1.get_size(); ++i) {
                blended_data[i] = 0.5f * data1[i] + 0.5f * data2[i];
            }
            talker_speech_config.speaker = blended;
        }
    }

    ov::genai::ChatHistory history;
    std::vector<ov::Tensor> videos;
    std::vector<ov::genai::VideoMetadata> videos_metadata;
    if (argc == 5) {
        auto [video, video_metadata] = load_video(argv[4]);
        videos.push_back(std::move(video));
        videos_metadata.push_back(std::move(video_metadata));
    }

    std::string prompt;
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    size_t turn = 0;
    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::OmniDecodedResults decoded_results = pipe.generate(history,
                                                                   rgbs,
                                                                   videos,
                                                                   videos_metadata,
                                                                   audios,
                                                                   text_config,
                                                                   talker_speech_config,
                                                                   print_subword);
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    save_speech(decoded_results, "output_audio_" + std::to_string(turn) + ".wav");

    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        ++turn;
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images, videos and audio can be passed at each turn; here we rely on the info from turn 1.
        std::vector<ov::Tensor> turn_images, turn_videos, turn_audios;
        std::vector<ov::genai::VideoMetadata> turn_videos_metadata;
        decoded_results = pipe.generate(history,
                                        turn_images,
                                        turn_videos,
                                        turn_videos_metadata,
                                        turn_audios,
                                        text_config,
                                        talker_speech_config,
                                        print_subword);
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        save_speech(decoded_results, "output_audio_" + std::to_string(turn) + ".wav");

        std::cout << "\n----------\n"
                     "question:\n";
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
