// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"

namespace {

// todo: remove when openvino-tokenizers will support left padding
ov::genai::TokenizedInputs pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token_id) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token_id)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token_id)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

#ifdef _WIN32
#    include <windows.h>
#    define MAX_ABS_PATH _MAX_PATH
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
#else
#    include <dlfcn.h>
#    include <limits.h>
#    define MAX_ABS_PATH PATH_MAX
#    define get_absolute_path(result, path) realpath(path.c_str(), result)

std::string get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    std::ignore = get_absolute_path(&absolutePath[0], path);
    if (!absolutePath.empty()) {
        // on Linux if file does not exist or no access, function will return NULL, but
        // `absolutePath` will contain resolved path
        absolutePath.resize(absolutePath.find('\0'));
        return std::string(absolutePath);
    }
    std::stringstream ss;
    ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
    throw std::runtime_error(ss.str());
}
#endif

std::string get_ov_genai_library_path() {
    #ifdef _WIN32
        CHAR genai_library_path[MAX_PATH];
        HMODULE hm = NULL;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPSTR>(get_ov_genai_library_path),
                                &hm)) {
            std::stringstream ss;
            ss << "GetModuleHandle returned " << GetLastError();
            throw std::runtime_error(ss.str());
        }
        GetModuleFileNameA(hm, (LPSTR)genai_library_path, sizeof(genai_library_path));
        return std::string(genai_library_path);
    #elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
        Dl_info info;
        dladdr(reinterpret_cast<void*>(get_ov_genai_library_path), &info);
        return get_absolute_file_path(info.dli_fname).c_str();
    #else
    #    error "Unsupported OS"
    #endif  // _WIN32
}

std::filesystem::path with_openvino_tokenizers(const std::filesystem::path& path) {
    #ifdef _WIN32
        constexpr char tokenizers[] = "openvino_tokenizers.dll";
    #elif __linux__
        constexpr char tokenizers[] = "libopenvino_tokenizers.so";
    #elif __APPLE__
        constexpr char tokenizers[] = "libopenvino_tokenizers.dylib";
    #endif
        return path.parent_path() / tokenizers;
}

}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    int64_t m_pad_token_id = 0;
    int64_t m_bos_token_id = 1;
    int64_t m_eos_token_id = 2;

    TokenizerImpl() = default;
    TokenizerImpl(std::string tokenizers_path, const std::string device) {
        ov::Core core;
        
        if (ov::genai::utils::is_xml(tokenizers_path))
            OPENVINO_THROW("tokenizers_path should be a path to a dir not a xml file");

        const char* ov_tokenizers_path = getenv(ov::genai::utils::get_tokenizers_env_name());
        if (ov_tokenizers_path) {
            core.add_extension(ov_tokenizers_path);
        } else {
            OPENVINO_THROW("openvino_tokenizers path is not set");
        }

        std::shared_ptr<ov::Model> tokenizer_model, detokenizer_model;
        try {
            tokenizer_model = core.read_model(tokenizers_path + "/openvino_tokenizer.xml");
            detokenizer_model = core.read_model(tokenizers_path + "/openvino_detokenizer.xml");
        } catch (...) {
            OPENVINO_THROW("Cannot compile tokenizer and/or detokenizer. Please check that "
                        "openvino_tokenizer.xml and openvino_detokenizer.xml exist in \"" + tokenizers_path + "\"");
        }
        m_tokenize_request = core.compile_model(tokenizer_model, device).create_infer_request();
        m_detokenizer_request = core.compile_model(detokenizer_model, device).create_infer_request();

        auto rt_info = tokenizer_model->get_rt_info();
        if (rt_info.count("eos_token_id") > 0)
            m_eos_token_id = rt_info["eos_token_id"].as<int64_t>();
        if (rt_info.count("bos_token_id") > 0)
            m_bos_token_id = rt_info["bos_token_id"].as<int64_t>();
        if (rt_info.count("pad_token_id") > 0)
            m_pad_token_id = rt_info["pad_token_id"].as<int64_t>();
        }

    TokenizedInputs encode(std::string prompt) {
        size_t batch_size = 1;
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
        m_tokenize_request.infer();
        return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
    }

    TokenizedInputs encode(std::vector<std::string>& prompts) {
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
        auto size_ = m_tokenize_request.get_input_tensor().get_shape();
        m_tokenize_request.infer();
        pad_left(m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask"), m_pad_token_id);
        
        // todo: fix mask filled with '2' instead of '0' 
        // https://github.com/openvinotoolkit/openvino_tokenizers/pull/90 should've fixed this
        ov::Tensor attention_mask = m_tokenize_request.get_tensor("attention_mask");
        int64_t* attention_mask_data = attention_mask.data<int64_t>();
        std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);
        
        return {m_tokenize_request.get_tensor("input_ids"), m_tokenize_request.get_tensor("attention_mask")};
    }

    std::string decode(std::vector<int64_t> tokens) {
        size_t batch_size = 1;
        m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        m_detokenizer_request.infer();
        return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens) {
        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        auto res = m_detokenizer_request.get_output_tensor();
        
        std::vector<std::string> strings;
        for (int i = 0; i < res.get_shape()[0]; ++i) {
            strings.emplace_back(res.data<std::string>()[i]);
        }
        return strings;
    }

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines) {
        // todo: implement calling detokenizer in a single batch
        std::vector<std::string> results;
        for (auto& line: lines){
            ov::Tensor tokens = ov::Tensor{ov::element::i64, {1, line.size()}, line.data()};
            m_detokenizer_request.set_input_tensor(tokens);
            m_detokenizer_request.infer();
            auto res = m_detokenizer_request.get_output_tensor();
            auto res_str = res.data<std::string>()[0];
            results.emplace_back(res_str);
        }
        
        return results;
    }
};

Tokenizer::Tokenizer(const std::string& tokenizers_path, const std::string& device) {
    ov::genai::utils::GenAIEnvManager env_manager(with_openvino_tokenizers(get_ov_genai_library_path()).string());
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizers_path, device);
}

TokenizedInputs Tokenizer::encode(const std::string prompt) {
    return m_pimpl->encode(std::move(prompt));
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text) {
    return encode(std::vector<std::string>(text.begin(), text.end()));
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
}

int64_t Tokenizer::get_bos_token_id() const {
    return m_pimpl->m_bos_token_id;
}

int64_t Tokenizer::get_eos_token_id() const {
    return m_pimpl->m_eos_token_id;
}

int64_t Tokenizer::get_pad_token_id() const {
    return m_pimpl->m_pad_token_id;
}

void Tokenizer::set_pad_token_id(int64_t pad_token_id) {
    m_pimpl->m_pad_token_id = pad_token_id;
}

void Tokenizer::set_bos_token_id(int64_t bos_token_id) {
    m_pimpl->m_bos_token_id = bos_token_id;
}

void Tokenizer::set_eos_token_id(int64_t eos_token_id) {
    m_pimpl->m_eos_token_id = eos_token_id;
}

Tokenizer::~Tokenizer() = default;

}  // namespace genai
}  // namespace ov
