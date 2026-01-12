// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module.hpp"
#include "modules/md_img_preprocess.hpp"
#include "modules/md_io.hpp"
#include "modules/md_text_encoder.hpp"
#include "modules/md_vision_encoder.hpp"
#include "modules/md_text_embedding.hpp"
#include "modules/md_embedding_merger.hpp"
#include "modules/md_llm_inference.hpp"
#include "modules/md_zimage_denoiser_loop.hpp"
#include "modules/md_vae_decoder_tiling.hpp"
#include "modules/md_vae_decoder.hpp"
#include "modules/md_clip_text_encoder.hpp"
#include "modules/md_save_image.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {
namespace module {

void module_connect(PipelineModuleInstance& pipeline_instance) {
    std::unordered_map<std::string, IBaseModule::PTR> module_map;
    for (const auto& module_ptr : pipeline_instance) {
        // Process inputs
        for (auto& input : module_ptr->module_desc->inputs) {
            auto it = std::find_if(std::begin(pipeline_instance),
                                   std::end(pipeline_instance),
                                   [&](const IBaseModule::PTR& ptr) {
                                       return ptr->get_module_name() == input.source_module_name;
                                   });
            OPENVINO_ASSERT(it != std::end(pipeline_instance), "Can't find module[" + input.source_module_name + "], please check config yaml.");

            module_ptr->inputs[input.name].module_ptr = *it;

            // IBaseModule::OutputModule outp_module = {module_ptr};
            auto& module_ptrs = (*it)->outputs[input.source_module_out_name].module_ptrs;  
            if (std::find(module_ptrs.begin(), 
                          module_ptrs.end(),
                          module_ptr) == module_ptrs.end()) {
                module_ptrs.push_back(module_ptr);
            }
        }
    }
}

void construct_pipeline(const PipelineModulesDesc& pipeline_modules_desc, PipelineModuleInstance& pipeline_instance, const PipelineDesc::PTR& pipeline_desc) {
    for (auto& module_desc : pipeline_modules_desc) {
        IBaseModule::PTR module_ptr = nullptr;
        switch (module_desc.second->type) {
#define GENAI_MODULE_TYPE_CASE(module_type_enum, module_class) \
    case ModuleType::module_type_enum:                         \
        module_ptr = module_class::create(module_desc.second, pipeline_desc); \
        break;

        GENAI_MODULE_TYPE_CASE(ParameterModule, ParameterModule);
        GENAI_MODULE_TYPE_CASE(ResultModule, ResultModule);
        GENAI_MODULE_TYPE_CASE(ImagePreprocessModule, ImagePreprocessModule);
        GENAI_MODULE_TYPE_CASE(TextEncoderModule, TextEncoderModule);
        GENAI_MODULE_TYPE_CASE(VisionEncoderModule, VisionEncoderModule);
        GENAI_MODULE_TYPE_CASE(TextEmbeddingModule, TextEmbeddingModule);
        GENAI_MODULE_TYPE_CASE(EmbeddingMergerModule, EmbeddingMergerModule);
        GENAI_MODULE_TYPE_CASE(LLMInferenceModule, LLMInferenceModule);
        GENAI_MODULE_TYPE_CASE(ClipTextEncoderModule, ClipTextEncoderModule);
        GENAI_MODULE_TYPE_CASE(ZImageDenoiserLoopModule, ZImageDenoiserLoopModule);
        GENAI_MODULE_TYPE_CASE(VAEDecoderTilingModule, VAEDecoderTilingModule);
        GENAI_MODULE_TYPE_CASE(VAEDecoderModule, VAEDecoderModule);
        GENAI_MODULE_TYPE_CASE(SaveImageModule, SaveImageModule);

#undef GENAI_MODULE_TYPE_CASES
        default:
            break;
        }
        OPENVINO_ASSERT(module_ptr, "No implementation for type: " + ModuleTypeConverter::toString(module_desc.second->type));
        pipeline_instance.push_back(module_ptr);
    }
    module_connect(pipeline_instance);
}

}  // namespace module
}  // namespace genai
}  // namespace ov