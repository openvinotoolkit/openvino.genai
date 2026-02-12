#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

// CLAHE Enhancement to pop colors
void apply_clahe(cv::Mat& img) {
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes;
    cv::split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab);
    cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);
}

int main(int argc, char* argv[]) {
    try {
        // Updated Usage to include Prompt and Device
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <models_path> <input_video_path> [device] [prompt]" << std::endl;
            return 1;
        }

        const std::string models_path = argv[1];
        const std::string input_video = argv[2];

        // Fix #1: Default to CPU for compatibility
        const std::string device = (argc > 3) ? argv[3] : "CPU";

        // Fix #2: Allow custom prompt via CLI
        std::string prompt = "masterpiece, best quality, cyberpunk anime, vibrant neon lights, sharp focus, 8k";
        if (argc > 4) {
            prompt = argv[4];
        }

        const std::string output_video = "output_style_transfer.mp4";

        const int size = 512;
        const int steps = 15;
        const float strength = 0.6f;
        const int skip_factor = 3;

        // Initialize Pipeline
        std::cout << "Initializing pipeline on " << device << "..." << std::endl;
        ov::genai::Image2ImagePipeline pipe(models_path, device);

        // Open Video Source
        cv::VideoCapture cap(input_video);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << input_video << std::endl;
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0)
            fps = 30;

        // Validate Video Writer
        cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(size, size));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot open video writer for file: " << output_video
                      << " Please check codec availability." << std::endl;
            return 1;
        }

        cv::Mat raw_frame, resized_frame, stabilized_input, rgb_frame;
        cv::Mat previous_result;  // For temporal blending

        int frame_id = 0;
        std::cout << "Starting processing... (Press Ctrl+C to stop)" << std::endl;

        while (cap.read(raw_frame)) {
            // Process every Nth frame (Skip & Blend strategy)
            if (frame_id % skip_factor == 0) {
                cv::resize(raw_frame, resized_frame, cv::Size(size, size));

                // Temporal Stabilization: Weighted Blend
                if (previous_result.empty()) {
                    stabilized_input = resized_frame.clone();
                } else {
                    cv::addWeighted(resized_frame, 0.6, previous_result, 0.4, 0.0, stabilized_input);
                }

                // Apply CLAHE for better color definition
                apply_clahe(stabilized_input);

                // Safer Tensor Ownership
                // Convert to RGB
                cv::cvtColor(stabilized_input, rgb_frame, cv::COLOR_BGR2RGB);

                // Create a tensor that COPIES data to ensure safety
                ov::Tensor input_tensor(ov::element::u8, {1, (size_t)size, (size_t)size, 3});
                std::memcpy(input_tensor.data(), rgb_frame.data, input_tensor.get_byte_size());

                // Run Inference
                ov::Tensor generated = pipe.generate(prompt,
                                                     input_tensor,
                                                     ov::genai::strength(strength),
                                                     ov::genai::num_inference_steps(steps));

                // Process Output
                uint8_t* out_data = generated.data<uint8_t>();
                cv::Mat out_frame(size, size, CV_8UC3, out_data);

                // Convert back to BGR for OpenCV
                cv::cvtColor(out_frame, out_frame, cv::COLOR_RGB2BGR);

                // Store for next loop's blending
                previous_result = out_frame.clone();

                writer.write(out_frame);
                std::cout << "Processed Frame: " << frame_id << "\r" << std::flush;
            }
            frame_id++;
        }

        cap.release();
        writer.release();
        std::cout << "\nDone! Output saved to " << output_video << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}