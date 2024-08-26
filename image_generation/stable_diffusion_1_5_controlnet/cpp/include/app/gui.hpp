// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <string>
#include <cstdint>

#include "core.hpp"
#include "worker.hpp"

struct UIState {
    // left
    char prompt[1024] = "Dancing Darth Vader, best quality, extremely detailed";
    char negative_prompt[1024] = "monochrome, lowres, bad anatomy, worst quality, low quality";
    int steps = 20;
    int64_t seed = 963503610;
    float cfg = 7.5;
    int width = 512;
    int height = 512;
    float strength = 1.0;
    int resize_mode = 0;
    std::vector<std::string> samplers;
    int active_sampler_index = 0;

    // right
    std::vector<std::string> devices;
    int active_device_index = 0;
    std::string model_path;
};

struct UIPreviewState {
    std::string image_path;
    bool should_load = false;

    int image_width = 0;
    int image_height = 0;

    GLuint preview_texture = 0;
};

struct ResultState {
    GLuint texture = 0;
    int image_width = 0;
    int image_height = 0;
    bool should_render = false;
    ov::Tensor image;
    std::string output_name = "";
};

class App {
public:
    int Init();
    int Run();


private:
    void RenderLeftPanel();
    void RenderRightPanel();
    void Render();
    int Clean();
    void LoadInputImageData();
    void LoadResultImageData();

    const char* glsl_version = "#version 130";
    GLFWwindow* window;
    UIState state;
    UIPreviewState preview_state;
    ResultState result_state;
    std::shared_ptr<StableDiffusionControlnetPipeline> pipe = nullptr;
    Worker worker;

    std::atomic<bool> running{false};
    float xscale = 1.0;
    float yscale = 1.0;
};