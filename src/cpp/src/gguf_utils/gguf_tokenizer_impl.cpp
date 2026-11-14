// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_tokenizer_impl.hpp"

#include "gguf_utils.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include "openvino/genai/version.hpp"

namespace ov::genai {

namespace {
std::filesystem::path resolve_ov_tokenizers_path() {
#ifdef _WIN32
    const wchar_t* p = _wgetenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME_W);
    OPENVINO_ASSERT(p != nullptr, "Environment variable for tokenizer path is not set");
    return std::filesystem::path(std::wstring(p));
#else
    const char* p = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
    OPENVINO_ASSERT(p != nullptr, "Environment variable for tokenizer path is not set");
    return std::filesystem::path(p);
#endif
}

}  // namespace

GGUFTokenizerParameters GGUFTokenizerParameters::from_model(const std::shared_ptr<ov::Model>& model) {
    return GGUFTokenizerParameters(gguf_tokenizer_metadata_from_model(model));
}

Tokenizer make_gguf_tokenizer_from_model(const std::shared_ptr<ov::Model>& model,
                                         const std::filesystem::path& gguf_path,
                                         const ov::AnyMap& properties) {
    auto tokenizer_properties = properties;
    tokenizer_properties[gguf_utils::GGUF_SOURCE_PATH_ARG_NAME] = gguf_path;
    return GGUFTokenizer(GGUFTokenizerParameters(take_gguf_tokenizer_metadata(model)), tokenizer_properties);
}

GGUFTokenizer::GGUFTokenizer(const std::filesystem::path& gguf_path, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(is_gguf_model(gguf_path), "GGUFTokenizer requires a .gguf file");
    m_pimpl = std::make_shared<Tokenizer::TokenizerImpl>(gguf_path, properties);
}

GGUFTokenizer::GGUFTokenizer(const GGUFTokenizerParameters& parameters, const ov::AnyMap& properties) {
    m_pimpl = std::make_shared<GGUFTokenizerImpl>(parameters, properties);
}

GGUFTokenizerImpl::GGUFTokenizerImpl(const GGUFTokenizerParameters& gguf_tokenizer_parameters,
                                          const ov::AnyMap& properties) {
    ScopedVar env_manager(tokenizers_relative_to_genai());
    m_shared_object_ov_tokenizers = load_shared_object(resolve_ov_tokenizers_path());

    const auto gguf_props = gguf_utils::extract_gguf_properties(properties);
    std::shared_ptr<ov::Model> ov_tokenizer, ov_detokenizer;
    std::map<std::string, GGUFMetaData> tokenizer_config{};
    std::tie(ov_tokenizer, ov_detokenizer, tokenizer_config) =
        create_tokenizer_from_parameters(m_shared_object_ov_tokenizers, gguf_tokenizer_parameters.config);
    if (gguf_props.enable_save_ov_model && !gguf_props.source_path) {
        GENAI_WARN("enable_save_ov_model was requested but this Tokenizer was constructed from GGUF tokenizer "
                   "parameters, which carry no source directory, so openvino_tokenizer.xml / "
                   "openvino_detokenizer.xml are NOT written. Construct Tokenizer from the .gguf path instead.");
    }
    finalize_gguf_tokenizer(ov_tokenizer,
                            ov_detokenizer,
                            tokenizer_config,
                            gguf_props.rest,
                            gguf_props.enable_save_ov_model ? gguf_props.source_path : std::nullopt);
}

void GGUFTokenizerImpl::finalize_gguf_tokenizer(const std::shared_ptr<ov::Model>& ov_tokenizer,
                                                const std::shared_ptr<ov::Model>& ov_detokenizer,
                                                const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                                const ov::AnyMap& filtered_properties,
                                                const std::optional<std::filesystem::path>& save_from_path) {
    if (auto val = get_if_exist<ov::Tensor>(tokenizer_config, "padding_token_id")) {
        m_pad_token_id = static_cast<int64_t>((*val).data<uint32_t>()[0]);
    }
    if (auto val = get_if_exist<ov::Tensor>(tokenizer_config, "bos_token_id")) {
        m_bos_token_id = static_cast<int64_t>((*val).data<uint32_t>()[0]);
    }
    if (auto val = get_if_exist<ov::Tensor>(tokenizer_config, "eos_token_id")) {
        m_eos_token_id = static_cast<int64_t>((*val).data<uint32_t>()[0]);
    }
    // SentencePiece decoding omits CONTROL pieces even when special-token filtering is
    // disabled. Read their spelling from the vocabulary so chat templates receive BOS/EOS.
    if (auto vocabulary = get_if_exist<std::vector<std::string>>(tokenizer_config, "tokens")) {
        const auto piece = [&](int64_t id) {
            return id >= 0 && static_cast<size_t>(id) < vocabulary->size() ? vocabulary->at(id) : std::string{};
        };
        m_pad_token = piece(m_pad_token_id);
        m_bos_token = piece(m_bos_token_id);
        m_eos_token = piece(m_eos_token_id);
    }
    if (auto val = get_if_exist<std::string>(tokenizer_config, "chat_template")) {
        m_chat_template = *val;
    }
    if (!m_chat_template.empty()) {
        m_original_chat_template = m_chat_template;
        m_chat_template = patch_gguf_chat_template(m_chat_template);
    }
    ov_tokenizer->set_rt_info(ov::genai::get_version().buildNumber, "openvino_genai_version");
    ov_detokenizer->set_rt_info(ov::genai::get_version().buildNumber, "openvino_genai_version");
    if (save_from_path) {
        save_gguf_tokenizer_models(ov_tokenizer, ov_detokenizer, *save_from_path);
    }

    setup_tokenizer(std::make_pair(ov_tokenizer, ov_detokenizer), filtered_properties);
    if (auto patcher = gguf_chat_template_patcher(tokenizer_config)) {
        set_chat_template_patcher(std::move(patcher));
    }
}

}  // namespace ov::genai
