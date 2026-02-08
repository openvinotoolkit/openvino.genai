// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/visual_language/pipeline.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

/**
 * @brief Live VLM Chat Sample
 * * A multi-threaded sample demonstrating real-time interaction with Visual Language Models.
 * - Thread 1 (Main): Handles OpenCV video feed and user input.
 * - Thread 2 (Worker): Handles OpenVINO inference to prevent UI freezing.
 */

//  Shared State
struct SharedState {
    std::mutex mutex;
    std::condition_variable cv;
    cv::Mat current_frame;
    std::string prompt_text;
    std::atomic<bool> request_pending{false};
    std::atomic<bool> is_processing{false};
    std::atomic<bool> should_exit{false};
};

SharedState state;


// Streamer to print tokens as they arrive
ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

// Convert OpenCV Mat (BGR) to OpenVINO Tensor (RGB)
ov::Tensor mat_to_tensor(const cv::Mat& frame) {
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    ov::Tensor tensor(ov::element::u8, ov::Shape{1, (size_t)rgb_frame.rows, (size_t)rgb_frame.cols, 3});
    std::memcpy(tensor.data<uint8_t>(), rgb_frame.data, rgb_frame.total() * rgb_frame.elemSize());
    return tensor;
}

// AI Worker Thread 
void ai_worker_thread(std::string model_path, std::string device) {
    try {
        std::cout << "[System] Loading VLM Model on " << device << " (Please Wait)..." << std::endl;
        ov::genai::VLMPipeline pipe(model_path, device);
        
        ov::genai::GenerationConfig config;
        config.max_new_tokens = 120; // Slightly increased for better descriptions

        std::cout << "System Model Loaded. Ready." << std::endl;

        while (!state.should_exit) {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [] { return state.request_pending.load() || state.should_exit.load(); });
            
            if (state.should_exit) break;

            // Safe Copy of Data
            cv::Mat frame_copy = state.current_frame.clone();
            std::string prompt_copy = state.prompt_text;
            state.request_pending = false; 
            state.is_processing = true;
            lock.unlock();

            // Safety Check: Empty frames cause crashes
            if (frame_copy.empty()) {
                std::cerr << "Warning - Dropped empty frame." << std::endl;
                state.is_processing = false;
                continue;
            }

            // Inference
            std::cout << "\n >> [Openvino VLM]: ";
            try {
                ov::Tensor image_tensor = mat_to_tensor(frame_copy);
                
                // Explicit cast for MSVC compatibility (The Windows Fix)
                pipe.generate(prompt_copy, 
                    ov::genai::images(std::vector<ov::Tensor>{image_tensor}), 
                    ov::genai::generation_config(config), 
                    ov::genai::streamer(print_subword));
                    
            } catch (const std::exception& e) {
                std::cerr << "\n[Error During Inference]: " << e.what() << std::endl;
            }
            std::cout << "\n\n(Press SPACE or ENTER to ask again)\n" << std::flush;
            
            state.is_processing = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal AI Error: " << e.what() << std::endl;
        state.should_exit = true;
    }
}

// Main Thread
int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <MODEL_DIR> [DEVICE]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string device = (argc == 3) ? argv[2] : "CPU";

    // Launch AI Thread
    std::thread worker(ai_worker_thread, model_path, device);
    worker.detach();

    // Setup Camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        state.should_exit = true;
        return 1;
    }

    std::cout << " Live Visual Agent Started \n";
    std::cout << "Controls:\n  [SPACE/ENTER] Take Snapshot & Chat\n  [ESC] Quit\n";
    
    cv::Mat frame;
    std::string user_input;

    while (!state.should_exit) {
        cap >> frame;
        if (frame.empty()) continue;

        // UI Overlay
        if (state.is_processing) {
            cv::putText(frame, "Thinking...", cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        } else {
            cv::putText(frame, "Ready", cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Live VLM Agent", frame);
        
        // Input Handling
        int key = cv::waitKey(30); 
        if (key == 27) { // ESC key to exit
            state.should_exit = true;
            state.cv.notify_all();
            break;
        }
        
        // Trigger Chat (Space=32, Enter=13)
        if ((key == 32 || key == 13) && !state.is_processing) {
            std::cout << ">> Snapshot taken! Type question (Enter for default): ";
            
            // Note: std::getline blocks the UI thread. In a production app, 
            // this would be async or handled via a non-blocking UI. For simplicity, we block here.
            std::getline(std::cin, user_input);
            if (user_input.empty()) user_input = "Describe this image in  detail.";

            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.current_frame = frame.clone();
                state.prompt_text = user_input;
                state.request_pending = true;
            }
            state.cv.notify_one();
        }
    }
    return 0;
}