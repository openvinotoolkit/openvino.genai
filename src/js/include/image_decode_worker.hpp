// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

#include <napi.h>

#include "include/helper.hpp"
#include "openvino/openvino.hpp"

template <typename PipelineType>
class ImageDecodeWorker : public Napi::AsyncWorker {
    // ov::Tensor for the image pipelines, ov::genai::VideoGenerationResult for Text2VideoPipeline.
    using DecodeResult =
        std::decay_t<decltype(std::declval<PipelineType&>().decode(std::declval<const ov::Tensor&>()))>;

public:
    ImageDecodeWorker(Napi::Function& callback,
                      std::shared_ptr<PipelineType> pipe,
                      ov::Tensor latent,
                      std::shared_ptr<std::atomic<bool>> is_busy)
        : Napi::AsyncWorker(callback),
          m_pipe(std::move(pipe)),
          m_latent(std::move(latent)),
          m_is_busy(std::move(is_busy)) {}
    ~ImageDecodeWorker() override = default;

    void Execute() override {
        m_result = m_pipe->decode(m_latent);
    }

    void OnOK() override {
        m_is_busy->store(false);
        Callback().Call({Env().Null(), cpp_to_js<DecodeResult, Napi::Value>(Env(), m_result)});
    }

    void OnError(const Napi::Error& e) override {
        m_is_busy->store(false);
        Callback().Call({Napi::Error::New(Env(), e.Message()).Value(), Env().Undefined()});
    }

private:
    std::shared_ptr<PipelineType> m_pipe;
    ov::Tensor m_latent;
    DecodeResult m_result;
    std::shared_ptr<std::atomic<bool>> m_is_busy;
};
