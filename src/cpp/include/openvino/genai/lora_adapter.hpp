// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>
#include <optional>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/genai/tokenizer.hpp"


namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS AdapterController;
struct AdapterControllerImpl;

// Inmutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier
class OPENVINO_GENAI_EXPORTS Adapter {
    class Impl;
    std::shared_ptr<Impl> m_pimpl;
    friend AdapterController;
    friend AdapterControllerImpl;
    friend bool operator== (const Adapter& a, const Adapter& b);
    friend bool operator< (const Adapter& a, const Adapter& b);
public:
    explicit Adapter(const std::string& path);
    Adapter() = default;

    operator bool() const {
        return bool(m_pimpl);
    }
};

// bool OPENVINO_GENAI_EXPORTS operator== (const Adapter& a, const Adapter& b);
// bool OPENVINO_GENAI_EXPORTS operator< (const Adapter& a, const Adapter& b);


struct OPENVINO_GENAI_EXPORTS AdapterConfig {
    enum Mode {
        MODE_AUTO,          // Automatically selected (depends on the place where this mode is applied and device selection)
        MODE_DYNAMIC,       // A, B, alpha are fully variable
        MODE_STATIC_RANK,   // A and B have static shape, alpha is variable // FIXME: WA to unlock experiments, gives a unique perf level
        MODE_STATIC,        // A, B and alpha are constants
        MODE_FUSE           // A, B and alpha are constants, fused to main matrix W
    };

    Mode get_mode() const { return mode; }
    void set_mode(Mode);

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

    // Returns true if it is not a trivial config
    operator bool() const {
        return !adapters.empty();
    }

private:

    Mode mode;
    std::vector<Adapter> adapters;
    std::vector<float> alphas;

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

    AdapterController(std::shared_ptr<ov::Model> model, const AdapterConfig& config, const std::string& prefix, std::string device = "");

    // Apply adapters configured in the current config set last time, or set and use new config given as optional `config` argument
    void apply(ov::InferRequest& request, const std::optional<AdapterConfig>& config = std::nullopt);

    // the next call of apply will set all adapter tensors regardless of config change, use this method if full state.reset is called for the controlled model
    void force_full_apply(bool full_apply = true);

    operator bool() const {
        return bool(m_pimpl);
    }
};


}  // namespace genai
}  // namespace ov
