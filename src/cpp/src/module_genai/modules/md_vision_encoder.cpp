#include "md_vision_encoder.hpp"

#include "module_genai/module_factory.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vector>
#include "circular_buffer_queue.hpp"
#include "json_utils.hpp"
#include "module_genai/module_base.hpp"
#include "nlohmann/json.hpp"
#include "utils.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vl_sdpa_transformations.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(VisionEncoderModule);

void VisionEncoderModule::print_static_config() {
    std::cout << R"(
  vision_encoder:
    type: "VisionEncoderModule"
    device: "GPU"
    inputs:
      - name: "preprocessed_image"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "source_size"                                # Used by Qwen 2.5-VL
        type: "VecInt"                                     # Support DataType: [VecInt]
        source: "ParentModuleName.OutputPortName"
      - name: "images_sequence"                            # Used by Qwen 2.5-VL
        type: "VecInt"                                     # Support DataType: [VecInt]
        source: "ParentModuleName.OutputPortName"
      - name: "input_ids"                                  # Required for Qwen 3.5. Optional for other models when position-related outputs are needed.
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "grid_thw"                                   # Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "pos_embeds"                                 # Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "rotary_cos"                                 # Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "rotary_sin"                                 # Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
      - name: "attention_mask"                             # Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "image_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "video_embedding"
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "position_ids"                               # [Optional], depends on input_ids
        type: "OVTensor"                                   # Support DataType: [OVTensor]
      - name: "rope_delta"                                 # [Optional], depends on input_ids
        type: "Int | OVTensor"                             # Support DataType: [Int (Qwen 2.5-VL) | OVTensor (Qwen 3.5)]
      - name: "visual_pos_mask"                            # [Optional], depends on input_ids, Used by Qwen 3.5
        type: "OVTensor"                                   # Support DataType: [OVTensor]
    params:
      model_path: "model"
      vision_start_token_id: 100001
    )" << std::endl;
}

VisionEncoderModule::VisionEncoderModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    VLMModelType model_type = to_vlm_model_type(desc->model_type);
    if (model_type != VLMModelType::QWEN2_VL && model_type != VLMModelType::QWEN2_5_VL &&
        model_type != VLMModelType::QWEN3_5) {
        GENAI_ERR("VisionEncoderModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }
    if (!initialize()) {
        GENAI_ERR("Failed to initiate VisionEncoderModule");
    }
}

VisionEncoderModule::~VisionEncoderModule() {}

bool VisionEncoderModule::initialize() {
    const auto &params = module_desc->params;
    VLMModelType model_type = to_vlm_model_type(module_desc->model_type);
    auto it_path = params.find("model_path");
    if (it_path == params.end()) {
        GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'model_path' not found in params");
        return false;
    }

    auto it_vision_start_token_id = params.find("vision_start_token_id");;
    if (it_vision_start_token_id != params.end()) {
        m_vision_start_token_id = std::stoll(it_vision_start_token_id->second);
    } else {
        GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'vision_start_token_id' not found in params, using default 0");
    }

    std::filesystem::path model_path = it_path->second;

    std::shared_ptr<ov::Model> model;
    if (model_path.extension() == ".xml") {
        if (!std::filesystem::exists(model_path)) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: model file not found at " + 
                model_path.string());
            return false;
        }
        model = utils::singleton_core().read_model(model_path);
        model_path = model_path.parent_path();
    } else {
        auto model_file_path = model_path / "openvino_vision_embeddings_merger_model.xml";
        if (model_type == VLMModelType::QWEN3_5) {
            model_file_path = model_path / "qwen3_5_vision.xml";
        }
        if (!std::filesystem::exists(model_file_path)) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: model file not found at " + 
                model_file_path.string());
            return false;
        }
        model = utils::singleton_core().read_model(model_file_path);
    }

    if (model_type == VLMModelType::QWEN2_VL || model_type == VLMModelType::QWEN2_5_VL) {
        utils::request_vl_sdpa_transformations(model);
    }

    auto compiled_model = utils::singleton_core().compile_model(
        model, 
        module_desc->device.empty() ? "CPU" : module_desc->device, {});

    if (model_type == VLMModelType::QWEN2_5_VL || model_type == VLMModelType::QWEN3_5) {
        m_with_cu_seqlens_input = utils::check_vl_sdpa_transformations(compiled_model);
        ov::genai::utils::print_compiled_model_properties(compiled_model,
            m_with_cu_seqlens_input ? "VLM vision embeddings merger model with VLSDPA optimization ENABLED" :
            "VLM vision embeddings merger model with VLSDPA optimization DISABLED");
    }
    
    m_request_queue = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }
    );
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_path, "config.json");
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_path, "preprocessor_config.json");
    m_merge_length = std::pow(m_processor_config.merge_size, 2);

    {
        const auto config_path = model_path / "config.json";
        if (std::filesystem::exists(config_path)) {
            std::ifstream f(config_path);
            if (f.is_open()) {
                nlohmann::json data;
                f >> data;
                using ov::genai::utils::read_json_param;
                read_json_param(data, "image_token_id", m_image_pad_token_id);
                read_json_param(data, "video_token_id", m_video_pad_token_id);
            }
        }
    }

    return true;
}

void VisionEncoderModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    VLMModelType model_type = to_vlm_model_type(module_desc->model_type);

    if (model_type == VLMModelType::QWEN2_VL || model_type == VLMModelType::QWEN2_5_VL) {
        if (this->inputs.find("preprocessed_image") == this->inputs.end() || this->inputs["preprocessed_image"].data == nullptr) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'preprocessed_image' input not found");
            return;
        }
        if (this->inputs.find("source_size") == this->inputs.end() || this->inputs["source_size"].data.as<std::vector<int>>().empty()) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'source_size' input not found");
            return;
        }
        if (this->inputs.find("images_sequence") == this->inputs.end() || this->inputs["images_sequence"].data.as<std::vector<int>>().empty()) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'images_sequence' input not found");
            return;
        }

        ov::Tensor input_ids;
        if (this->inputs.find("input_ids") == this->inputs.end() || this->inputs["input_ids"].data == nullptr) {
            GENAI_WARN("VisionEncoderModule[" + module_desc->name +
                    "]: 'input_ids' input not found, position_ids output will be skipped");
        } else {
            input_ids = this->inputs["input_ids"].data.as<ov::Tensor>();
        }

        ov::Tensor image_embedding;
        ov::Tensor video_embedding;
        EncodedImage encoded;
        encoded.resized_source_size.height = this->inputs["source_size"].data.as<std::vector<int>>()[0];
        encoded.resized_source_size.width = this->inputs["source_size"].data.as<std::vector<int>>()[1];
        encoded.resized_source = this->inputs["preprocessed_image"].data.as<ov::Tensor>();
        std::vector<int> images_sequence = this->inputs["images_sequence"].data.as<std::vector<int>>();
        std::tie(video_embedding, image_embedding) = embed(encoded, images_sequence, input_ids);
        
        this->outputs["image_embedding"].data = image_embedding;
        this->outputs["video_embedding"].data = video_embedding;
        if (input_ids) {
            this->outputs["position_ids"].data = m_position_ids;
            int position_ids_max_element = static_cast<int>(*std::max_element(m_position_ids.data<int64_t>(), m_position_ids.data<int64_t>() + m_position_ids.get_size()));
            this->outputs["rope_delta"].data = position_ids_max_element + 1 - static_cast<int>(input_ids.get_shape().at(1));
        }
    } else if (model_type == VLMModelType::QWEN3_5) {
        if (!exists_input("preprocessed_image")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'preprocessed_image' input not found");
            return;
        }
        ov::Tensor preprocessed_image = get_input("preprocessed_image").as<ov::Tensor>();
        if (!exists_input("grid_thw")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'grid_thw' input not found");
            return;
        }
        ov::Tensor grid_thw = get_input("grid_thw").as<ov::Tensor>();
        if (!exists_input("pos_embeds")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'pos_embeds' input not found");
            return;
        }
        ov::Tensor pos_embeds = get_input("pos_embeds").as<ov::Tensor>();
        if (!exists_input("rotary_cos")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'rotary_cos' input not found");
            return;
        }
        ov::Tensor rotary_cos = get_input("rotary_cos").as<ov::Tensor>();
        if (!exists_input("rotary_sin")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'rotary_sin' input not found");
            return;
        }
        ov::Tensor rotary_sin = get_input("rotary_sin").as<ov::Tensor>();
        if (!exists_input("input_ids")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'input_ids' input not found");
            return;
        }
        ov::Tensor input_ids = get_input("input_ids").as<ov::Tensor>();
        if (!exists_input("attention_mask")) {
            GENAI_ERR("VisionEncoderModule[" + module_desc->name + "]: 'attention_mask' input not found");
            return;
        }
        ov::Tensor attention_mask = get_input("attention_mask").as<ov::Tensor>();

        Qwen3_5VisionEmbeddingResult result = embed(
            preprocessed_image, grid_thw, pos_embeds, rotary_cos, rotary_sin, input_ids, attention_mask);

        this->outputs["image_embedding"].data   = result.visual_embeds;
        this->outputs["visual_pos_mask"].data = result.visual_pos_mask;
        this->outputs["position_ids"].data    = result.position_ids;
        this->outputs["rope_delta"].data     = result.rope_deltas;
        return;
    } else {
        OPENVINO_THROW("Unsupported model: " + module_desc->model_type);
    }
}


std::pair<ov::Tensor, ov::Tensor> VisionEncoderModule::embed(const EncodedImage &image, const std::vector<int>& images_sequence, const ov::Tensor& input_ids) {
    OPENVINO_ASSERT(m_request_queue, "VisionEncoderModule is not initialized. Call initialize() first.");

    std::vector<size_t> vec_images_sequence(images_sequence.begin(), images_sequence.end());
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw({image}, vec_images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] = qwen2_vl_utils::reorder_video_embeds_and_grid_thw({}, {});

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> reordered_vision_grid_thw;
    reordered_vision_grid_thw.reserve(reordered_videos_grid_thw.size() + reordered_images_grid_thw.size());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_vision_grid_thw);

    auto [window_index, cu_window_seqlens] = qwen2_5_vl_utils::get_window_index(
        reordered_vision_grid_thw,
        m_processor_config,
        m_vlm_config
    );

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_request_queue.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    if (m_with_cu_seqlens_input) {
        ov::Tensor cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw);
        ov::Tensor t_cu_window_seqlens = qwen2_5_vl_utils::get_cu_window_seqlens(cu_window_seqlens);
        vision_embeddings_merger.set_tensor("cu_seq_lens", cu_seq_lens);
        vision_embeddings_merger.set_tensor("cu_window_seqlens", t_cu_window_seqlens);
    }
    else {
        ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw);
        size_t hidden_states_size = attention_mask.get_shape().at(1);
        ov::Tensor window_attention_mask = qwen2_5_vl_utils::get_window_attention_mask(hidden_states_size, cu_window_seqlens);
        vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
        vision_embeddings_merger.set_tensor("window_attention_mask", window_attention_mask);
    }
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.set_tensor("window_index", window_index);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    auto out_vision_shape = processed_vision_embeds.get_shape();

    // Split Video and Image's features.
    auto video_fea_num = calc_vec_tokens_num(reordered_videos_grid_thw);
    auto image_fea_num = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t video_fea_count = 0;
    if ((video_fea_num + image_fea_num) != 0) {
        video_fea_count = out_vision_shape.at(0) * video_fea_num / (video_fea_num + image_fea_num);
    }

    ov::Shape video_fea_shape = ov::Shape({video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_video = ov::Tensor(processed_vision_embeds.get_element_type(), video_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() >= res_video.get_byte_size(), "Vision embeds size should >= video embeds size.");
    std::memcpy(res_video.data(), processed_vision_embeds.data(), res_video.get_byte_size());

    ov::Shape image_fea_shape({out_vision_shape.at(0) - video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_image(processed_vision_embeds.get_element_type(), image_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() == res_image.get_byte_size() + res_video.get_byte_size(),
                    "Vision embeds size should == image + video embeds size.");
    std::memcpy(res_image.data(),
                reinterpret_cast<uint8_t*>(processed_vision_embeds.data()) + res_video.get_byte_size(),
                res_image.get_byte_size());

    std::vector<size_t> videos_sequence(reordered_video_embeds.size());
    std::iota(videos_sequence.begin(), videos_sequence.end(), 0);

    if (input_ids) {
        m_position_ids = create_position_ids(input_ids,
                                             reordered_images_grid_thw,
                                             vec_images_sequence,
                                             0,
                                             reordered_videos_grid_thw,
                                             videos_sequence,
                                             0,
                                             m_vision_start_token_id,
                                             {});
    }

    return {res_video, res_image};
}

Qwen3_5VisionEmbeddingResult VisionEncoderModule::embed(
        const ov::Tensor &pixel_values,
        const ov::Tensor &grid_thw,
        const ov::Tensor &pos_embeds,
        const ov::Tensor &rotary_cos,
        const ov::Tensor &rotary_sin,
        const ov::Tensor &input_ids,
        const ov::Tensor &attention_mask) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_request_queue.get());
    ov::InferRequest& vision_embed_request = infer_request_guard.get();
    vision_embed_request.set_tensor("pixel_values", pixel_values);
    vision_embed_request.set_tensor("grid_thw", grid_thw);
    vision_embed_request.set_tensor("pos_embeds", pos_embeds);
    vision_embed_request.set_tensor("rotary_cos", rotary_cos);
    vision_embed_request.set_tensor("rotary_sin", rotary_sin);
    vision_embed_request.infer();
    ov::Tensor vision_embeds = vision_embed_request.get_tensor("visual_embeds");

    const auto &ids_shape = input_ids.get_shape();
    const size_t batch   = ids_shape[0];
    const size_t seq_len = ids_shape[1];
    const int64_t* ids   = input_ids.data<const int64_t>();

    ov::Tensor visual_pos_mask(ov::element::boolean, ids_shape);
    for (size_t idx = 0; idx < batch * seq_len; ++idx) {
        bool active = (ids[idx] == m_image_pad_token_id || ids[idx] == m_video_pad_token_id);
        if (attention_mask && attention_mask.get_size() > 0) {
            // attention_mask is i64 [B, S]; 0 means masked out
            active = active && (attention_mask.data<const int64_t>()[idx] != 0);
        }
        static_cast<bool*>(visual_pos_mask.data())[idx] = active;
    }

    const int32_t spatial_merge_size = m_processor_config.merge_size;
    const int64_t* image_grid = grid_thw.data<const int64_t>();
    const size_t   image_grid_rows = grid_thw.get_shape().at(0);

    ov::Tensor position_ids(ov::element::i64, {3, batch, seq_len});
    std::memset(position_ids.data(), 0, position_ids.get_byte_size());
    ov::Tensor rope_deltas(ov::element::i64, {batch, 1});
    std::memset(rope_deltas.data(), 0, rope_deltas.get_byte_size());

    int64_t* pos_data   = position_ids.data<int64_t>();
    int64_t* delta_data = rope_deltas.data<int64_t>();

    size_t image_grid_index = 0;

    for (size_t b = 0; b < batch; ++b) {
        std::vector<int64_t> tokens;
        std::vector<size_t>  active_indices;
        tokens.reserve(seq_len);
        active_indices.reserve(seq_len);
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t idx = b * seq_len + s;
            if (attention_mask && attention_mask.get_size() > 0 &&
                attention_mask.data<const int64_t>()[idx] == 0) {
                continue;
            }
            tokens.push_back(ids[idx]);
            active_indices.push_back(s);
        }

        if (tokens.empty()) {
            delta_data[b] = 0;
            continue;
        }

        std::vector<int64_t> pos_t, pos_h, pos_w;
        pos_t.reserve(tokens.size());
        pos_h.reserve(tokens.size());
        pos_w.reserve(tokens.size());

        int64_t last_max = -1;

        auto append_text = [&](size_t length) {
            if (length == 0) return;
            const int64_t base = last_max + 1;
            for (size_t i = 0; i < length; ++i) {
                const int64_t v = base + static_cast<int64_t>(i);
                pos_t.push_back(v);
                pos_h.push_back(v);
                pos_w.push_back(v);
            }
            last_max = base + static_cast<int64_t>(length) - 1;
        };

        auto append_visual = [&](int64_t t, int64_t h, int64_t w) {
            const int64_t llm_grid_h = h / spatial_merge_size;
            const int64_t llm_grid_w = w / spatial_merge_size;
            const int64_t base = last_max + 1;
            int64_t max_dim = 0;
            for (int64_t tt = 0; tt < t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        pos_t.push_back(base + tt);
                        pos_h.push_back(base + hh);
                        pos_w.push_back(base + ww);
                        max_dim = std::max(max_dim, std::max(tt, std::max(hh, ww)));
                    }
                }
            }
            last_max = base + max_dim;
        };

        size_t local_grid_index = image_grid_index;
        std::vector<std::pair<size_t, bool>> visual_starts;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            if (tokens[i] != m_vision_start_token_id) continue;
            const int64_t next = tokens[i + 1];
            if (next == m_image_pad_token_id) {
                visual_starts.emplace_back(i + 1, true);
            } else if (next == m_video_pad_token_id) {
                visual_starts.emplace_back(i + 1, false);
            }
        }

        size_t st = 0;
        for (const auto& [ed, is_image] : visual_starts) {
            if (ed < st) continue;
            append_text(ed - st);

            int64_t t = 0, h = 0, w = 0;
            if (is_image) {
                OPENVINO_ASSERT(local_grid_index < image_grid_rows,
                    "VisionEncoderModule::embed: image_grid_thw has fewer entries than image placeholders");
                t = image_grid[local_grid_index * 3 + 0];
                h = image_grid[local_grid_index * 3 + 1];
                w = image_grid[local_grid_index * 3 + 2];
                local_grid_index++;
            }

            append_visual(t, h, w);

            const int64_t llm_grid_h = h / spatial_merge_size;
            const int64_t llm_grid_w = w / spatial_merge_size;
            const int64_t visual_len = t * llm_grid_h * llm_grid_w;
            st = ed + static_cast<size_t>(visual_len);
        }

        if (st < tokens.size()) {
            append_text(tokens.size() - st);
        }

        int64_t max_pos = pos_t.empty() ? 0 : pos_t[0];
        for (size_t i = 0; i < pos_t.size(); ++i) {
            max_pos = std::max({max_pos, pos_t[i], pos_h[i], pos_w[i]});
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            const size_t s    = active_indices[i];
            const size_t base = b * seq_len + s;
            pos_data[0 * batch * seq_len + base] = pos_t[i];
            pos_data[1 * batch * seq_len + base] = pos_h[i];
            pos_data[2 * batch * seq_len + base] = pos_w[i];
        }

        if (attention_mask && attention_mask.get_size() > 0) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t idx = b * seq_len + s;
                if (attention_mask.data<const int64_t>()[idx] != 0) continue;
                pos_data[0 * batch * seq_len + idx] = 1;
                pos_data[1 * batch * seq_len + idx] = 1;
                pos_data[2 * batch * seq_len + idx] = 1;
            }
        }

        delta_data[b] = max_pos + 1 - static_cast<int64_t>(seq_len);
        image_grid_index = local_grid_index;
    }

    const auto &embeds_shape = vision_embeds.get_shape();
    const size_t hidden     = embeds_shape[1];
    const size_t elem_size  = vision_embeds.get_element_type().size();
    const size_t row_bytes  = hidden * elem_size;

    ov::Tensor visual_embeds_scattered(vision_embeds.get_element_type(), {batch, seq_len, hidden});
    std::memset(visual_embeds_scattered.data(), 0, visual_embeds_scattered.get_byte_size());

    const char* src = static_cast<const char*>(vision_embeds.data());
    char*       dst = static_cast<char*>(visual_embeds_scattered.data());
    const bool* mask_ptr = static_cast<const bool*>(visual_pos_mask.data());

    size_t visual_idx = 0;
    const size_t total = batch * seq_len;
    for (size_t idx = 0; idx < total; ++idx) {
        if (!mask_ptr[idx]) continue;
        OPENVINO_ASSERT(visual_idx < embeds_shape[0],
            "VisionEncoderModule::embed: visual_embeds shorter than visual_pos_mask count");
        std::memcpy(dst + idx * row_bytes, src + visual_idx * row_bytes, row_bytes);
        visual_idx++;
    }
    OPENVINO_ASSERT(visual_idx == embeds_shape[0],
        "VisionEncoderModule::embed: visual_embeds length does not match visual_pos_mask count");

    return {position_ids, visual_pos_mask, rope_deltas, visual_embeds_scattered};
}

ov::Tensor VisionEncoderModule::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) {
    const size_t spatial_merge_size = m_processor_config.merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw.at(0);
        size_t h = grid_thw.at(1);
        size_t w = grid_thw.at(2);

        max_grid_size = std::max({max_grid_size, h, w});

        // According to spatial merge size, create height & width position IDs
        std::vector<size_t> hpos_ids;
        std::vector<size_t> wpos_ids;
        size_t h_blocks = h / spatial_merge_size;
        size_t w_blocks = w / spatial_merge_size;
        hpos_ids.reserve(h * w);
        wpos_ids.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        hpos_ids.push_back(hb * spatial_merge_size + hs);
                        wpos_ids.push_back(wb * spatial_merge_size + ws);
                    }
                }
            }
        }

        // Stack and repeat for each t
        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < hpos_ids.size(); ++j) {
                all_pos_ids.push_back({hpos_ids[j], wpos_ids[j]});
            }
        }
    }

    // Calculate rotary embeddings for max_grid_size
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_request_queue.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    const float theta = 10000.0f;
    
    std::vector<float> inv_freq(dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(dim / 2));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(dim / 2);
        for (size_t j = 0; j < dim / 2; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    float* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids.at(i);
        size_t h_idx = pos.at(0);
        size_t w_idx = pos.at(1);
        std::copy_n(freqs[h_idx].begin(), dim / 2, output_data + i * dim);
        std::copy_n(freqs[w_idx].begin(), dim / 2, output_data + i * dim + dim / 2);
    }
    return rotary_pos_emb;
}

size_t VisionEncoderModule::calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const {
    size_t token_num = 0;
    for (auto grid_thw : vec_grid_thw) {
        token_num += calc_tokens_num(grid_thw[0], grid_thw[1], grid_thw[2]);
    }
    return token_num;
}

size_t VisionEncoderModule::calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const {
    return grid_t * grid_h * grid_w / m_merge_length;
}

ov::Tensor VisionEncoderModule::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t vision_start_token_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    const size_t spatial_merge_size = 2; // m_vision_encoder->get_processor_config().merge_size;
    const size_t tokens_per_second = m_vlm_config.vision_config_tokens_per_second;
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;

    if (history_vision_count.size() > 0) {
        size_t vid_idx = 0;
        size_t img_idx = 0;
        for (size_t i = 0; i < history_vision_count.size(); i++) {
            size_t ed = vid_idx + history_vision_count[i].first;
            for (; vid_idx < ed; vid_idx++) {
                reordered_images_grid_thw.push_back(videos_grid_thw.at(vid_idx - video_id));
            }
            ed = img_idx + history_vision_count[i].second;
            for (; img_idx < ed; img_idx++) {
                reordered_images_grid_thw.push_back(images_grid_thw.at(img_idx - image_id));
            }
        }
    } else {
        for (size_t new_frame_id : videos_sequence) {
            reordered_images_grid_thw.push_back(videos_grid_thw.at(new_frame_id - video_id));
        }
        for (size_t new_image_id : images_sequence) {
            reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
        }
    }

    const int64_t* input_ids = input_ids_tensor.data<int64_t>();
    size_t batch_size = input_ids_tensor.get_shape().at(0);
    size_t seq_len = input_ids_tensor.get_shape().at(1);

    std::vector<size_t> vision_start_indices;
    for (size_t i = 0; i < seq_len; ++i) {
        if (input_ids[i] == vision_start_token_id) {
            vision_start_indices.push_back(i);
        }
    }

    ov::Tensor position_ids{ov::element::i64, {3, batch_size, seq_len}};
    int64_t* pos_data = position_ids.data<int64_t>();
    
    size_t st = 0;
    int64_t next_pos = 0;
    size_t grid_idx = 0;

    for (size_t i = 0; i < vision_start_indices.size(); ++i) {
        size_t ed = vision_start_indices.at(i);

        // Process text tokens before image
        if (st < ed) {
            for (size_t pos = st; pos < ed; ++pos) {
                pos_data[pos] = next_pos;               // temporal
                pos_data[seq_len + pos] = next_pos;     // height
                pos_data[2 * seq_len + pos] = next_pos; // width
                next_pos++;
            }
        }

        // Process image start token
        pos_data[ed] = next_pos;               // temporal
        pos_data[seq_len + ed] = next_pos;     // height
        pos_data[2 * seq_len + ed] = next_pos; // width
        next_pos++;
        ed++;

        // Process image token with grid
        if (grid_idx < reordered_images_grid_thw.size()) {
            const auto& grid = reordered_images_grid_thw.at(grid_idx);
            size_t llm_grid_t = grid.at(0);
            size_t llm_grid_h = grid.at(1) / spatial_merge_size;
            size_t llm_grid_w = grid.at(2) / spatial_merge_size;
            size_t llm_grid_sz = llm_grid_h * llm_grid_w;
            size_t ed_image = ed + llm_grid_t * llm_grid_sz;

            // Fill temporal dimension
            for (size_t t = 0; t < llm_grid_t; t++) {
                std::fill_n(pos_data + ed + t * llm_grid_sz, llm_grid_sz, next_pos + t * tokens_per_second);
            }

            // Fill height and width dimensions
            int64_t* height_data = pos_data + seq_len + ed;
            int64_t* width_data = pos_data + 2 * seq_len + ed;
            for (size_t t = 0; t < llm_grid_t; t++) {
                size_t offset_sz = t * llm_grid_sz;
                for (size_t h = 0; h < llm_grid_h; ++h) {
                    size_t offset = h * llm_grid_w + offset_sz;
                    std::fill_n(height_data + offset, llm_grid_w, next_pos + h);
                    for (size_t w = 0; w < llm_grid_w; ++w) {
                        width_data[offset + w] = next_pos + w;
                    }
                }
            }

            next_pos += std::max(((llm_grid_t - 1) * tokens_per_second + 1), std::max(llm_grid_h, llm_grid_w));
            st = ed_image;
            grid_idx++;
        }
    }

    // Process remaining text tokens
    if (st < seq_len) {
        for (size_t pos = st; pos < seq_len; ++pos) {
            pos_data[pos] = next_pos;               // temporal
            pos_data[seq_len + pos] = next_pos;     // height
            pos_data[2 * seq_len + pos] = next_pos; // width
            next_pos++;
        }
    }

    return position_ids;
}

}
}
}