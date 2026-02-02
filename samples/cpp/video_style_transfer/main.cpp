// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * @brief Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
 * to enhance the visual clarity of the output frame.
 */
void enhance_quality(cv::Mat& img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes);
    
    // Apply CLAHE to the L-channel
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    
    cv::merge(lab_planes, lab);
    cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);
}

int main(int argc, char* argv[]) {
    try {
        // Basic argument parsing
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <models_path> <input_video_path> [device]" << std::endl;
            return 1;
        }

        const std::string models_path = argv[1];
        const std::string input_video = argv[2];
        const std::string device = (argc > 3) ? argv[3] : "GPU";
        const std::string output_video = "output_style_transfer.mp4";

        // Generation Parameters
        const std::string prompt = "masterpiece, best quality, cyberpunk anime, vibrant neon lights, sharp focus, 8k";
        const int size = 512;        // Model resolution
        const int steps = 15;        // Inference steps per frame
        const float strength = 0.6f; // Style strength (0.0 - 1.0)
        
        // Frame skipping optimization
        const int skip_factor = 3; 

        std::cout << " OpenVINO GenAI Video Style Transfer Sample " << std::endl;
        std::cout << "Model: " << models_path << std::endl;
        std::cout << "Device: " << device << std::endl;

        // Initialize Pipeline
        ov::genai::Image2ImagePipeline pipe(models_path, device);
        std::cout << "Pipeline initialized successfully." << std::endl;

        // Open Video Source
        cv::VideoCapture cap(input_video);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << input_video << std::endl;
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;

        cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(size, size));

        // Processing buffers
        cv::Mat raw_frame, resized_frame, rgb_frame;
        cv::Mat previous_result;     // For temporal blending
        cv::Mat stabilized_input;    // Blended input
        cv::Mat last_generated_frame;// Cache for skipped frames
        
        int frame_id = 0;

        std::cout << "Starting processing loop..." << std::endl;

        while (true) {
            cap >> raw_frame;
            if (raw_frame.empty()) break;

            // Process only keyframes to maintain throughput
            if (frame_id % skip_factor == 0) {
                
                cv::resize(raw_frame, resized_frame, cv::Size(size, size));

                // Temporal Stabilization
                // Blend current frame with previous result to reduce flickering
                if (previous_result.empty()) {
                    stabilized_input = resized_frame.clone();
                } else {
                    cv::addWeighted(resized_frame, 0.6, previous_result, 0.4, 0.0, stabilized_input);
                }

                // Prepare Input Tensor
                cv::cvtColor(stabilized_input, rgb_frame, cv::COLOR_BGR2RGB);
                ov::Tensor input_tensor(ov::element::u8, {1, (size_t)size, (size_t)size, 3}, rgb_frame.data);

                // Run Inference
                ov::Tensor out_tensor = pipe.generate(
                    prompt, 
                    input_tensor, 
                    ov::genai::strength(strength), 
                    ov::genai::num_inference_steps(steps)
                );

                // Post-process
                uint8_t* out_data = out_tensor.data<uint8_t>();
                cv::Mat ai_result(size, size, CV_8UC3, out_data);
                cv::Mat final_bgr;
                cv::cvtColor(ai_result, final_bgr, cv::COLOR_RGB2BGR);
                
                enhance_quality(final_bgr);

                // Update cache
                previous_result = final_bgr.clone();
                last_generated_frame = final_bgr.clone();
                
                std::cout << "\rProcessed Frame: " << frame_id << std::flush;
            } 
            else {
                // Skip frame (Animation-on-threes effect)
            }

            // Write output (repeats last frame during skip phase)
            if (!last_generated_frame.empty()) {
                writer.write(last_generated_frame);
            }
            frame_id++;
        }

        std::cout << "\nProcessing complete. Output saved to: " << output_video << std::endl;
        cap.release();
        writer.release();

    } catch (const std::exception& e) {
        std::cerr << "\nException caught: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}