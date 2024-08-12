#include "app/worker.hpp"
#include "app/gui.hpp"

ov::Tensor postprocess_image(ov::Tensor decoded_image) {
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());
    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

ImageToImagePipeline::ImageToImagePipeline(std::string& model, std::string& device) {
    pipe = new StableDiffusionControlnetPipeline(model, device);
}

void ImageToImagePipeline::Run(std::string& prompt,
    std::string& negative_prompt,
    std::string& input_image_path,
    int steps,
    uint32_t seed) {
    StableDiffusionControlnetPipelineParam param = {
        prompt,
        negative_prompt,
        input_image_path,
        steps,
        seed,
    };
    auto decoded_image = pipe->Run(param);
    auto image = postprocess_image(decoded_image);
    // TODO: to wxBitmap?
}

wxDECLARE_EVENT(wxEVT_COMMAND_IMAGE_GEN_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_IMAGE_GEN_COMPLETED, wxThreadEvent);

WorkerThread::WorkerThread(AppFrame* handler)
    : wxThread(wxTHREAD_DETACHED),
      frame(handler),
      shouldRun(false),
      shouldExit(false) {}

void WorkerThread::RequestRun() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        shouldRun = true;
    }
    cond.notify_one();
}

wxThread::ExitCode WorkerThread::Entry() {
    while (!shouldExit) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] {
            return shouldRun || shouldExit;
        });

        if (shouldExit)
            break;

        if (shouldRun) {
            shouldRun = false;

            std::string prompt, negative_prompt, input_image_path;
            int steps;
            uint32_t seed;

            {
                // std::lock_guard<std::mutex> pipelineLock(frame->GetImageToImagePipeline()->mutex);

            }
            frame->GetImageToImagePipeline()->Run(prompt, negative_prompt, input_image_path, steps, seed);

            wxQueueEvent(frame, new wxThreadEvent(wxEVT_COMMAND_IMAGE_GEN_COMPLETED));
        }
        // lock will released here
    }

    return (wxThread::ExitCode)0;
}