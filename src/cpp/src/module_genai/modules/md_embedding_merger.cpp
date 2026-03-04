#include "md_embedding_merger.hpp"

#include "module_genai/module_factory.hpp"
#include "utils.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(EmbeddingMergerModule);

void EmbeddingMergerModule::print_static_config() {
    std::cout << R"(
  embedding_merger:
    type: "EmbeddingMergerModule"
    device: "GPU"
    inputs:
      - name: "input_ids"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.input_ids"
      - name: "input_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.input_embedding"
      - name: "image_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.image_embedding"
      - name: "video_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.video_embedding"
    outputs:
      - name: "merged_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
    params:
      model_path: "model"
    )" << std::endl;
}

EmbeddingMergerModule::EmbeddingMergerModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    VLMModelType model_type = to_vlm_model_type(desc->model_type);
    if (model_type != VLMModelType::QWEN2_VL && model_type != VLMModelType::QWEN2_5_VL) {
        GENAI_ERR("EmbeddingMergerModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate EmbeddingMergerModule");
    }
}

EmbeddingMergerModule::~EmbeddingMergerModule() {}

bool EmbeddingMergerModule::initialize() {
    const auto &params = module_desc->params;
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("EmbeddingMergerModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }
    std::filesystem::path model_path = it_path->second;

    m_tokenizer = Tokenizer(model_path, {});
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_path, "config.json");
    encode_vision_placeholder_tokens();
    return true;
}

void EmbeddingMergerModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();
    if (this->inputs.find("input_ids") == this->inputs.end()) {
        GENAI_ERR("EmbeddingMergerModule[" + module_desc->name + "]: 'input_ids' input not found");
        return;
    }
    if (this->inputs.find("input_embedding") == this->inputs.end()) {
        GENAI_ERR("EmbeddingMergerModule[" + module_desc->name + "]: 'input_embedding' input not found");
        return;
    }
    if (this->inputs.find("image_embedding") == this->inputs.end() &&  this->inputs.find("video_embedding") == this->inputs.end()) {
        this->outputs["merged_embedding"].data = this->inputs["input_embedding"].data.as<ov::Tensor>();
        return;
    }

    ov::Tensor input_ids = this->inputs["input_ids"].data.as<ov::Tensor>();
    ov::Tensor input_embedding = this->inputs["input_embedding"].data.as<ov::Tensor>();
    ov::Tensor image_embedding = this->inputs["image_embedding"].data.as<ov::Tensor>();
    ov::Tensor video_embedding = this->inputs["video_embedding"].data.as<ov::Tensor>();
    int64_t image_pad_token_id = m_vision_token_ids["image_pad"];
    int64_t video_pad_token_id = m_vision_token_ids["video_pad"];
    
    ov::Tensor merged_embeds = merge_text_and_video_image_embeddings(
        input_ids,
        input_embedding, 
        image_embedding,
        video_embedding,
        image_pad_token_id,
        video_pad_token_id
    );
    
    this->outputs["merged_embedding"].data = merged_embeds;
}

void EmbeddingMergerModule::encode_vision_placeholder_tokens() {
    auto encoded_vision_tokens = m_tokenizer.encode(
        m_vlm_config.vision_start_token + m_vlm_config.image_pad_token + m_vlm_config.video_pad_token,
        ov::genai::add_special_tokens(false));
    m_vision_token_ids["vision_start"] = encoded_vision_tokens.input_ids.data<int64_t>()[0];
    m_vision_token_ids["image_pad"] = encoded_vision_tokens.input_ids.data<int64_t>()[1];
    m_vision_token_ids["video_pad"] = encoded_vision_tokens.input_ids.data<int64_t>()[2];
}

ov::Tensor EmbeddingMergerModule::merge_text_and_video_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const ov::Tensor& processed_image_embeds,
    const ov::Tensor& processed_video_embeds,
    const int64_t image_pad_token_id,
    const int64_t video_pad_token_id
) {
    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();
    const float* image_embeds_data = processed_image_embeds.data<const float>();
    const float* video_embeds_data = processed_video_embeds.data<const float>();

    size_t image_embed_idx = 0;
    size_t video_embed_idx = 0;
    const int64_t img_token = image_pad_token_id;
    const int64_t vid_token = video_pad_token_id;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            size_t flat_idx = batch_idx * seq_length + seq_idx;
            if (input_ids_data[flat_idx] == vid_token) {
                std::copy_n(video_embeds_data + video_embed_idx * hidden_size,
                            hidden_size,
                            merged_embeds_data + flat_idx * hidden_size);
                ++video_embed_idx;
            } else if (input_ids_data[flat_idx] == img_token) {
                std::copy_n(image_embeds_data + image_embed_idx * hidden_size,
                            hidden_size,
                            merged_embeds_data + flat_idx * hidden_size);
                ++image_embed_idx;
            }
        }
    }
    return merged_embeds;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
