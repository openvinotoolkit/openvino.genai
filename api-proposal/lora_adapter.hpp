// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>
#include <optional>
#include <filesystem>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/genai/tokenizer.hpp"


namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS AdapterController;
struct AdapterControllerImpl;
class AdapterImpl;

// Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier
class OPENVINO_GENAI_EXPORTS Adapter {
    std::shared_ptr<AdapterImpl> m_pimpl;

    friend AdapterController;
    friend AdapterControllerImpl;
    friend bool operator== (const Adapter& a, const Adapter& b);

    friend Adapter flux_adapter_normalization(const Adapter& adapter);
    friend Adapter diffusers_adapter_normalization(const Adapter& adapter);

    Adapter(const std::shared_ptr<AdapterImpl>& pimpl);
public:
    explicit Adapter(const std::filesystem::path& path);
    explicit Adapter(const ov::Tensor& safetensor);
    Adapter() = default;

    operator bool() const {
        return bool(m_pimpl);
    }
};


struct OPENVINO_GENAI_EXPORTS AdapterConfig {
    enum Mode {
        MODE_AUTO,          // Automatically selected (depends on the place where this mode is applied and device selection)
        MODE_DYNAMIC,       // A, B, alpha are fully variable
        MODE_STATIC_RANK,   // A and B have static shape, alpha is variable // FIXME: WA to unlock experiments, gives a unique perf level
        MODE_STATIC,        // A, B and alpha are constants. Use instead of MODE_FUSE if preserving weights precision is required at the cost of inference time
        MODE_FUSE           // A, B and alpha are constants, fused to main matrix W
    };

    Mode get_mode() const { return mode; }
    void set_mode(Mode);

    // Methods to get and set optional name prefix to filter tensor names in LoRA adapter file applicable to a particular model.
    // The prefix can be set at the user level or at a particular GenAI pipeline level. Usually GenAI pipelines should set
    // the prefix appropriately, and no need to be worried from user side.
    // But if the user has non-standard adapter file where the default prefix doesn't work, in this case
    // user should set the prefix. If the prefix is set at the user side, it is not overridden by the pipeline logic.
    // Use nullopt to indicate that the prefix is not set from the user side and let a particular GenAI pipeline set the default value.
    // The default value is nullopt.
    const std::optional<std::string>& get_tensor_name_prefix() const { return tensor_name_prefix; }
    void set_tensor_name_prefix(const std::optional<std::string>& _tensor_name_prefix) { tensor_name_prefix = _tensor_name_prefix; }

    AdapterConfig (Mode mode = MODE_AUTO);

    AdapterConfig (const Adapter& adapter, float alpha, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<std::pair<Adapter, float>>{{adapter, alpha}}, mode) {}

    AdapterConfig (const Adapter& adapter, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<Adapter>{adapter}, mode) {}

    template <typename AT, typename std::enable_if<std::is_constructible<Adapter, AT>::value, bool>::type = true>
    AdapterConfig (const std::initializer_list<AT>& adapters, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<Adapter>(adapters), mode) {}

    AdapterConfig (const std::initializer_list<std::pair<Adapter, float>>& adapters, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<std::pair<Adapter, float>>(adapters), mode) {}

    AdapterConfig (const std::vector<Adapter>& adapters, Mode mode = MODE_AUTO);

    AdapterConfig (const std::vector<std::pair<Adapter, float>>& adapters, Mode mode = MODE_AUTO);

    AdapterConfig& add(const Adapter& adapter, float alpha);
    AdapterConfig& add(const Adapter& adapter);
    AdapterConfig& set_alpha(const Adapter& adapter, float alpha);
    float get_alpha(const Adapter& adapter) const;
    AdapterConfig& remove(const Adapter&);
    const std::vector<Adapter>& get_adapters() const { return adapters; }
    std::vector<std::pair<Adapter, float>> get_adapters_and_alphas() const;
    void set_adapters_and_alphas(const std::vector<std::pair<Adapter, float>>& adapters);

    // Update adapters and alphas from other config. Mode and tensor_name_prefix are updated if they are set not to default values in other config.
    // It means that if other.get_mode() == MODE_AUTO, it will not override value in this config. If tensor_name_prefix is not set (== nullopt) then it won't be updated either.
    void update (const AdapterConfig& other);

    // Returns true if it is not a trivial config
    operator bool() const {
        return !adapters.empty();
    }

private:

    Mode mode;
    std::vector<Adapter> adapters;
    std::vector<float> alphas;
    std::optional<std::string> tensor_name_prefix;

};


class AdaptersProperty : public ov::Property<AdapterConfig> {
public:
    inline constexpr static const char* name () { return "adapters"; }

    constexpr AdaptersProperty() : ov::Property<AdapterConfig>(name()) {}

    inline std::pair<std::string, ov::Any> operator()(const AdapterConfig& config) const {
        return ov::Property<AdapterConfig>::operator()(config);
    }

    inline std::pair<std::string, ov::Any> operator()() const {
        return operator()(AdapterConfig());
    }

    inline std::pair<std::string, ov::Any> operator()(AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(mode));
    }

    inline std::pair<std::string, ov::Any> operator()(const Adapter& adapter, float alpha) const {
        return operator()(AdapterConfig(adapter, alpha));
    }

    inline std::pair<std::string, ov::Any> operator()(const Adapter& adapter, float alpha, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapter, alpha, mode));
    }

    inline std::pair<std::string, ov::Any> operator()(const Adapter& adapter, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapter, mode));
    }

    template <typename AT, typename std::enable_if<std::is_constructible<Adapter, AT>::value, bool>::type = true>
    inline std::pair<std::string, ov::Any> operator()(const std::initializer_list<AT>& adapters) const {
        return operator()(AdapterConfig(adapters));
    }

    template <typename AT, typename std::enable_if<std::is_constructible<Adapter, AT>::value, bool>::type = true>
    inline std::pair<std::string, ov::Any> operator()(const std::initializer_list<AT>& adapters, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapters, mode));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::initializer_list<std::pair<Adapter, float>>& adapters) const {
        return operator()(AdapterConfig(adapters));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::initializer_list<std::pair<Adapter, float>>& adapters, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapters, mode));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::vector<Adapter>& adapters) const {
        return operator()(AdapterConfig(adapters));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::vector<Adapter>& adapters, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapters, mode));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::vector<std::pair<Adapter, float>>& adapters) const {
        return operator()(AdapterConfig(adapters));
    }

    inline std::pair<std::string, ov::Any> operator()(const std::vector<std::pair<Adapter, float>>& adapters, AdapterConfig::Mode mode) const {
        return operator()(AdapterConfig(adapters, mode));
    }
};


static constexpr AdaptersProperty adapters;


class OPENVINO_GENAI_EXPORTS AdapterController {

    std::shared_ptr<AdapterControllerImpl> m_pimpl;
    friend AdapterControllerImpl;

public:

    AdapterController() = default;

    AdapterController(std::shared_ptr<ov::Model> model, const AdapterConfig& config, std::string device);

    // Apply adapters configured in the current config set last time, or set and use new config given as optional `config` argument
    void apply(ov::InferRequest request, const std::optional<AdapterConfig>& config = std::nullopt);

    // Returns true if a given name is one of the state names created by this adapter controller for dynamic LoRA
    // Helps to distinguish LoRA states from other states (e.g. KV cache state) in the model for a partial state reset.
    bool has_state_name(const std::string& name);

    operator bool() const {
        return bool(m_pimpl);
    }
};


}  // namespace genai
}  // namespace ov
