#include "splitted_model_infer.hpp"

#include <regex>

#include "logger.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/utils/thread_helper.hpp"

namespace ov::genai::module {

CSplittedModelInfer::CSplittedModelInfer(const std::string& model_path,
                                         const std::string& device,
                                         const bool& dynamic_load_model_weights,
                                         const ov::AnyMap& properties)
    : m_dynamic_load_model_weights(dynamic_load_model_weights),
      m_is_gpu(device.find("GPU") != std::string::npos || device.find("gpu") != std::string::npos),
      m_properties(properties) {
#ifndef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
    OPENVINO_ASSERT(!m_dynamic_load_model_weights,
                    "Dynamic loading of model weights is not enabled in this build. Please set "
                    "ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS to 1 and rebuild.");
#endif

    if (m_dynamic_load_model_weights) {
        OPENVINO_ASSERT(m_is_gpu, "Dynamic loading of model weights is currently only supported for GPU device.");
    }
    // parse all splitted model paths, model_path is the directory that contains all splitted models
    get_splitted_model_paths(model_path, device);
    load_model(model_path, m_properties, device);
}

void CSplittedModelInfer::get_splitted_model_paths(const std::string& model_path, const std::string& device) {
    // each splitted model should be named as **_l{index}_*.xml. For example: model_l0_.xml, model_l1_.xml
    m_splitted_model_paths.clear();

    std::regex pattern(R"(.*_l(\d+)_.*\.xml)");
    std::map<int, std::string> sorted_paths;

    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".xml") {
            std::string filename = entry.path().filename().string();
            std::smatch match;

            if (std::regex_search(filename, match, pattern)) {
                // check only contains one pattern is matched to avoid wrong file name like model_l0_l1_.xml
                OPENVINO_ASSERT(match.size() == 2,
                                "Invalid model file name: " + filename +
                                    ". Expected format: **_l{index}_*.xml, and only one index pattern is allowed.");

                int index = std::stoi(match[1].str());
                sorted_paths[index] = entry.path().string();
                continue;
            }

            // check if the file name end with "_preprocess.xml" or "_postprocess.xml" for preprocess and postprocess
            // model
            if (filename.size() > 15 && filename.substr(filename.size() - 15) == "_preprocess.xml") {
                m_preprocess_model_path = entry.path().string();
            } else if (filename.size() > 16 && filename.substr(filename.size() - 16) == "_postprocess.xml") {
                m_postprocess_model_path = entry.path().string();
            }
#if USE_FULL_MODEL
            if (filename.size() > 9 && filename.substr(filename.size() - 9) == "_full.xml") {
                m_full_compiled_model =
                    utils::singleton_core().compile_model(entry.path().string(), device, m_properties);
                m_full_infer_request = m_full_compiled_model.create_infer_request();
            }
#endif
        }
    }

    for (int i = 0; i < static_cast<int>(sorted_paths.size()); ++i) {
        auto it = sorted_paths.find(i);
        OPENVINO_ASSERT(
            it != sorted_paths.end(),
            "Model partitions should be continuous and start from index 0. Missing index: " + std::to_string(i));
        m_splitted_model_paths.push_back(it->second);
    }

    OPENVINO_ASSERT(!m_splitted_model_paths.empty(), "No splitted models found in " + model_path);
    OPENVINO_ASSERT(
        m_splitted_model_paths.size() >= 2,
        "At least two partition models are required (excluding preprocessing/postprocessing models). Found only " +
            std::to_string(m_splitted_model_paths.size()) + " partition models.");
    OPENVINO_ASSERT(!m_preprocess_model_path.empty() && !m_postprocess_model_path.empty(),
                    "Both preprocessing (_preprocess.xml) and postprocessing (_postprocess.xml) models are required.");
}

void CSplittedModelInfer::load_model(const std::string& model_path,
                                     const ov::AnyMap& properties,
                                     const std::string& device) {
#if USE_FULL_MODEL
#else
    {
        auto model = utils::singleton_core().read_model(m_preprocess_model_path);
        m_preprocess_compiled_model = utils::singleton_core().compile_model(model, device, properties);
        if (m_is_gpu) {
            // For GPU, all infer requests must share the same context to share weights.
            m_context = m_preprocess_compiled_model.get_context();
        }
        m_preprocess_infer_request = m_preprocess_compiled_model.create_infer_request();
    }
    {
        auto model = utils::singleton_core().read_model(m_postprocess_model_path);
        if (m_is_gpu) {
            // For GPU, all infer requests must share the same context to share weights.
            m_postprocess_compiled_model = utils::singleton_core().compile_model(model, m_context, properties);
        } else {
            m_postprocess_compiled_model = utils::singleton_core().compile_model(model, device, properties);
        }
        m_postprocess_infer_request = m_postprocess_compiled_model.create_infer_request();
    }

    auto properties_splitted_model = properties;
    for (const auto& path : m_splitted_model_paths) {
        auto model = utils::singleton_core().read_model(path);
        if (m_is_gpu) {
            if (m_dynamic_load_model_weights) {
                properties_splitted_model[ov::weights_path.name()] =
                    std::filesystem::path(path).replace_extension(".bin").string();
                auto cm = utils::singleton_core().compile_model(model, m_context, properties_splitted_model);
#    ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
                // Release model weights after compilation to save GPU memory. Load weights again in infer() when
                // weights are needed.
                cm.release_model_weights();
#    endif
                m_compiled_models.push_back(std::move(cm));
            } else {
                m_compiled_models.push_back(
                    utils::singleton_core().compile_model(model, m_context, properties_splitted_model));
                m_infer_requests.push_back(m_compiled_models.back().create_infer_request());
            }
        } else {
            m_compiled_models.push_back(
                utils::singleton_core().compile_model(model, device, properties_splitted_model));
            m_infer_requests.push_back(m_compiled_models.back().create_infer_request());
        }
    }
#endif
}

CSplittedModelInfer::~CSplittedModelInfer() {}

ov::Tensor CSplittedModelInfer::convert_to_remote_tensor(const ov::Tensor& tensor) {
    if (tensor.is<ov::RemoteTensor>()) {
        return tensor;
    } else {
        ov::Tensor remote_tensor = m_context.create_tensor(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(remote_tensor);
        return remote_tensor;
    }
}

void CSplittedModelInfer::infer(const ov::AnyMap& inputs) {
#if USE_FULL_MODEL
    for (const auto& input : inputs) {
        m_full_infer_request.set_tensor(input.first, input.second.as<ov::Tensor>());
    }

    m_full_infer_request.infer();
#else
    int num_splitted_models = static_cast<int>(m_compiled_models.size());
    OPENVINO_ASSERT(num_splitted_models > 1,
                    "Splitted models should be at least 2, but got " + std::to_string(num_splitted_models));

#    ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
#        if ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
    std::future<bool> future_flag;
    if (m_dynamic_load_model_weights) {
        future_flag = std::move(thread_utils::load_model_weights_async(m_compiled_models[0]));
    }
#        else   // ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
    if (m_dynamic_load_model_weights) {
        PROFILE(pm, "load_model_weights");
        m_compiled_models[0].load_model_weights();
    }
#        endif  // ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
#    endif      // ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS

    // Preprocess
    for (const auto& input : inputs) {
        m_preprocess_infer_request.set_tensor(input.first, input.second.as<ov::Tensor>());
    }
    m_preprocess_infer_request.infer();

    // The "tokens" tensor produced by the preprocess stage is used as the initial hidden_states.
    ov::Tensor hidden_states_tensor = m_preprocess_infer_request.get_tensor("tokens");
    ov::Tensor text_embeds_tensor = m_preprocess_infer_request.get_tensor("text_embeds");      // [-1,-1,1536]
    ov::Tensor timestep_proj_tensor = m_preprocess_infer_request.get_tensor("timestep_proj");  // [-1,6,1536]
    ov::Tensor rotary_cos_tensor = m_preprocess_infer_request.get_tensor("rotary_cos");        // [-1,-1,64]
    ov::Tensor rotary_sin_tensor = m_preprocess_infer_request.get_tensor("rotary_sin");        // [-1,-1,64]

    if (m_is_gpu && !hidden_states_tensor.is<ov::RemoteTensor>()) {
        hidden_states_tensor = convert_to_remote_tensor(hidden_states_tensor);
        text_embeds_tensor = convert_to_remote_tensor(text_embeds_tensor);
        timestep_proj_tensor = convert_to_remote_tensor(timestep_proj_tensor);
        rotary_cos_tensor = convert_to_remote_tensor(rotary_cos_tensor);
        rotary_sin_tensor = convert_to_remote_tensor(rotary_sin_tensor);
    }

    ov::Tensor temb_tensor = m_preprocess_infer_request.get_tensor("temb");
    ov::Tensor ppf_tensor = m_preprocess_infer_request.get_tensor("ppf");
    ov::Tensor pph_tensor = m_preprocess_infer_request.get_tensor("pph");
    ov::Tensor ppw_tensor = m_preprocess_infer_request.get_tensor("ppw");

    // Splitted models
    std::future<bool> next_future_flag;
    for (int i = 0; i < num_splitted_models; ++i) {
        PROFILE(pm, "splitted_model_infer_" + std::to_string(i));
        ov::InferRequest curInferRequest;
        if (m_dynamic_load_model_weights) {
#    ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
            if (i + 1 < num_splitted_models) {
#        if ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
                next_future_flag = thread_utils::load_model_weights_async(m_compiled_models[i + 1]);
#        else
                m_compiled_models[i + 1].load_model_weights();
#        endif  // ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
            }
#        if ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
            if (future_flag.valid())
                future_flag.wait();
#        endif  // ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
            curInferRequest = m_compiled_models[i].create_infer_request();
#    endif      // ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
        } else {
            curInferRequest = m_infer_requests[i];
        }

        curInferRequest.set_output_tensor(0, hidden_states_tensor);
        curInferRequest.set_tensor("hidden_states", hidden_states_tensor);
        curInferRequest.set_tensor("text_embeds", text_embeds_tensor);
        curInferRequest.set_tensor("timestep_proj", timestep_proj_tensor);
        curInferRequest.set_tensor("rotary_cos", rotary_cos_tensor);
        curInferRequest.set_tensor("rotary_sin", rotary_sin_tensor);
        {
            PROFILE(pmi, "infer");
            curInferRequest.infer();
        }

#    ifdef ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
        if (m_dynamic_load_model_weights) {
#        if ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
            auto release_future =
                thread_utils::release_model_weights_async(m_compiled_models[i], std::move(curInferRequest));
            if (release_future.valid()) {
                release_future.wait();
            }
#        else
            curInferRequest = ov::InferRequest();  // release infer request before releasing model weights to ensure the
                                                   // model weights can be released successfully.
            m_compiled_models[i].release_model_weights();
#        endif
        }

#        if ENABLE_MULTIPLE_THREAD_LOAD_MODEL_WEIGHT
        future_flag = std::move(next_future_flag);
#        endif
#    endif  // ENABLE_DYNAMIC_LOAD_MODEL_WEIGHTS
    }

    GENAI_DEBUG(
        "hidden_states_tensor is remote tensor: " + std::to_string(hidden_states_tensor.is<ov::RemoteTensor>()) +
        ", shape:" + tensor_utils::shape_to_string(hidden_states_tensor.get_shape()));

    // Postprocess
    m_postprocess_infer_request.set_tensor("hidden_states", hidden_states_tensor);
    m_postprocess_infer_request.set_tensor("temb", temb_tensor);
    m_postprocess_infer_request.set_tensor("ppf", ppf_tensor);
    m_postprocess_infer_request.set_tensor("pph", pph_tensor);
    m_postprocess_infer_request.set_tensor("ppw", ppw_tensor);
    m_postprocess_infer_request.infer();
#endif
}

ov::Tensor CSplittedModelInfer::get_output_tensor(const size_t& index) {
#if USE_FULL_MODEL
    return m_full_infer_request.get_output_tensor(index);
#else
    return m_postprocess_infer_request.get_output_tensor(index);
#endif
}

void CSplittedModelInfer::set_output_tensor(size_t idx, const ov::Tensor& tensor) {
#if USE_FULL_MODEL
    m_full_infer_request.set_output_tensor(idx, tensor);
#else
    m_postprocess_infer_request.set_output_tensor(idx, tensor);
#endif
}
}  // namespace ov::genai::module