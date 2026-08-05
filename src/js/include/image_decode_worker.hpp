// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <utility>

#include <napi.h>

#include "include/helper.hpp"
#include "openvino/openvino.hpp"

template <typename PipelineType>
class ImageDecodeWorker : public Napi::AsyncWorker {
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
        m_image = m_pipe->decode(m_latent);
    }

    void OnOK() override {
        m_is_busy->store(false);
        Callback().Call({Env().Null(), cpp_to_js<ov::Tensor, Napi::Value>(Env(), m_image)});
    }

    void OnError(const Napi::Error& e) override {
        m_is_busy->store(false);
        Callback().Call({Napi::Error::New(Env(), e.Message()).Value(), Env().Undefined()});
    }

private:
    std::shared_ptr<PipelineType> m_pipe;
    ov::Tensor m_latent;
    ov::Tensor m_image;
    std::shared_ptr<std::atomic<bool>> m_is_busy;
};
