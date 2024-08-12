// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>

#include <wx/wx.h>
#include <wx/thread.h>

#include "core/core.hpp"

class ImageToImagePipeline {
public:
    ImageToImagePipeline();
    void Init(std::string& model, std::string& device);
    void Run(std::string& prompt,
                 std::string& negative_prompt,
                 std::string& input_image_path,
                 int steps,
                 uint32_t seed);

private:
    StableDiffusionControlnetPipeline *pipe;
};

class AppFrame;

class WorkerThread : public wxThread {
public:
    WorkerThread(AppFrame* handler);
    void RequestRun();

protected:
    virtual ExitCode Entry();

private:
    AppFrame* frame;
    std::mutex mutex;
    std::condition_variable cond;
    bool shouldRun;
    bool shouldExit;
};

wxDECLARE_EVENT(wxEVT_COMMAND_IMAGE_GEN_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_IMAGE_GEN_COMPLETED, wxThreadEvent);
