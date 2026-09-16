#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "audio_utils.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

int main(int argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 4, "Usage: ", argv[0], " <MODEL_DIR> <DEVICE> <WAV_FILE> [WAV_FILE ...]");

    const std::filesystem::path model_dir = argv[1];
    const std::string device = argv[2];
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 256;
    ov::genai::ContinuousBatchingPipeline pipeline(model_dir, scheduler_config, device);
    auto tokenizer = pipeline.get_tokenizer();
    auto generation_config = pipeline.get_config();
    generation_config.max_new_tokens = 256;
    generation_config.do_sample = false;
    generation_config.apply_chat_template = false;

    const std::string prompt = "<|im_start|>system\n<|im_end|>\n"
                               "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                               "<|im_start|>assistant\n";

    std::vector<ov::genai::GenerationHandle> handles;
    for (int argument_index = 3; argument_index < argc; ++argument_index) {
        auto waveform = utils::audio::read_wav(argv[argument_index]);
        const ov::Tensor audio(ov::element::f32, {waveform.size()}, waveform.data());
        handles.push_back(pipeline.add_request(handles.size(),
                                               prompt,
                                               ov::genai::audios(std::vector<ov::Tensor>{audio}),
                                               ov::genai::generation_config(generation_config)));
    }

    while (pipeline.has_non_finished_requests()) {
        pipeline.step();
    }

    for (size_t request_id = 0; request_id < handles.size(); ++request_id) {
        OPENVINO_ASSERT(handles[request_id]->get_status() == ov::genai::GenerationStatus::FINISHED,
                        "Audio request ", request_id, " did not finish successfully");
        const auto outputs = handles[request_id]->read_all();
        OPENVINO_ASSERT(outputs.size() == 1, "Expected one transcription for audio request ", request_id);
        std::cout << "Request " << request_id << ": " << tokenizer.decode(outputs.front().generated_ids) << '\n';
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}

// results
// Request 0: language Chinese<asr_text>审核制。
// Request 1: language English<asr_text>How are you doing today?
// Request 2: language English<asr_text>You know if you watch the show, you're aware that I spend most of my time right over there, wandering the news forest for you, felling all the biggestand hardiest white story oaks, cutting and shaping them into the newsiest most topical cleats, clamps and planks, keeping them in a constant angle, gradually creating a shell-shaped shallow ball hole using the fire bending technique instead of steam bending, obviously. Then I lay out all the keel blocks to carefully set up the stem stern and garboard, attach the bill phurocks to the timber, and lovingly craft a flat transom stern out of naturally curved quarter circles. Then secure all the planks with trunnels handmade from the finest locust wood and finally adorn it with a proud bow, sprit forepeak and custom gilded fingerhead to present to you the Dutch Golden Age Spiegel yacht that is my monologue. But sometimes, sometimes folks, gotta hydrate after that. Spiegel yacht. But sometimes I awaken from a meat sweat induced fever strapped to a basket on the wonder wheel at Coney Island, stumble across the garbage flecked beach to the sound of a terrifying ragged bellow I realize is coming from my own lungs,
