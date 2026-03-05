#include "openvino/genai/c/video_generation_config.h"
#include "types_c.h"

ov_status_e ov_genai_video_generation_config_create(ov_genai_video_generation_config** config) {
    if (!config) return ov_status_e::INVALID_C_PARAM;
    try {
        auto _config = std::make_unique<ov_genai_video_generation_config>();
        _config->object = std::make_shared<ov::genai::VideoGenerationConfig>();
        *config = _config.release();
    } catch (...) { return ov_status_e::UNKNOW_EXCEPTION; }
    return ov_status_e::OK;
}

void ov_genai_video_generation_config_free(ov_genai_video_generation_config* config) {
    if (config) delete config;
}

ov_status_e ov_genai_video_generation_config_set_width(ov_genai_video_generation_config* config, size_t width) {
    if (!config || !config->object) return ov_status_e::INVALID_C_PARAM;
    try { config->object->width = width; } catch (...) { return ov_status_e::UNKNOW_EXCEPTION; }
    return ov_status_e::OK;
}

ov_status_e ov_genai_video_generation_config_set_height(ov_genai_video_generation_config* config, size_t height) {
    if (!config || !config->object) return ov_status_e::INVALID_C_PARAM;
    try { config->object->height = height; } catch (...) { return ov_status_e::UNKNOW_EXCEPTION; }
    return ov_status_e::OK;
}

ov_status_e ov_genai_video_generation_config_set_num_frames(ov_genai_video_generation_config* config, size_t num_frames) {
    if (!config || !config->object) return ov_status_e::INVALID_C_PARAM;
    try { config->object->num_frames = num_frames; } catch (...) { return ov_status_e::UNKNOW_EXCEPTION; }
    return ov_status_e::OK;
}

ov_status_e ov_genai_video_generation_config_set_num_inference_steps(ov_genai_video_generation_config* config, size_t num_inference_steps) {
    if (!config || !config->object) return ov_status_e::INVALID_C_PARAM;
    try { config->object->num_inference_steps = num_inference_steps; } catch (...) { return ov_status_e::UNKNOW_EXCEPTION; }
    return ov_status_e::OK;
}
