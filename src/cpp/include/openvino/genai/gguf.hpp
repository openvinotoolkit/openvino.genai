// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"

namespace ov::genai {

/**
 * @brief GGUF tokenizer metadata a Tokenizer can be built from: the `tokenizer.*` keys of a
 * .gguf file.
 *
 * `config` is keyed by the metadata sub-key -- the part after the last dot of `tokenizer.ggml.*` /
 * `tokenizer.chat_template`, e.g. "model", "tokens", "merges", "scores", "token_type", "pre",
 * "bos_token_id", "eos_token_id", "chat_template". Each value holds one of std::string,
 * std::vector<std::string> or ov::Tensor, mirroring the GGUF metadata variant.
 */
struct OPENVINO_GENAI_EXPORTS GGUFTokenizerParameters {
    /// tokenizer sub-key -> value (string / vector<string> / Tensor)
    ov::AnyMap config;

    GGUFTokenizerParameters() = default;
    explicit GGUFTokenizerParameters(ov::AnyMap config) : config(std::move(config)) {}

    /**
     * @brief Read the parameters out of a model converted by the OpenVINO GGUF frontend, which
     * attaches them to the model's runtime info so the .gguf need not be re-opened.
     * @param gguf_model model produced by reading a .gguf through the GGUF frontend.
     * @throws ov::Exception if the model carries no GGUF tokenizer metadata (e.g. a plain IR or
     *         one built by the legacy GGUF reader).
     */
    static GGUFTokenizerParameters from_model(const std::shared_ptr<ov::Model>& gguf_model);
};

/**
 * @brief Builds a Tokenizer from GGUF tokenizer metadata or a local GGUF file.
 */
class OPENVINO_GENAI_EXPORTS GGUFTokenizer : public Tokenizer {
public:
    explicit GGUFTokenizer(const std::filesystem::path& gguf_path, const ov::AnyMap& properties = {});

    /**
     * @brief Builds the tokenizer without reopening a GGUF already loaded by the frontend.
     * @param parameters GGUF tokenizer metadata; see GGUFTokenizerParameters::from_model().
     * @param properties Properties passed to ov::Core::compile_model.
     */
    explicit GGUFTokenizer(const GGUFTokenizerParameters& parameters, const ov::AnyMap& properties = {});
};

/**
* @brief Selects which reader converts a `.gguf` file.
*
* Accepted values:
*  - `"LEGACY"` (default) - the previous hand-written GGUF reader, supporting three architectures:
*    `llama`, `qwen2`, and `qwen3`.
*  - `"FRONTEND"` - the OpenVINO GGUF frontend, supporting a broader range of architectures.
*    This is a preview feature that may have limitations.
*
* The legacy reader remains the default while the frontend is a preview feature.
* The frontend is intended to replace the legacy reader in the future.
*
* Example: `LLMPipeline(model_path, "CPU", ov::genai::gguf_reader("FRONTEND"))`.
*/
static constexpr ov::Property<std::string> gguf_reader{"GGUF_READER"};


}  // namespace ov::genai
