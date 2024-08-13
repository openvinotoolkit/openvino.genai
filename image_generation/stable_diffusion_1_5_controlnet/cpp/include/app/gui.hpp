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
    char prompt[1024] = "Dancing Darth Vader, best quality, extremely detailed";
    char negative_prompt[1024] = "monochrome, lowres, bad anatomy, worst quality, low quality";
    int steps = 20;
    int64_t seed = 42;

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
    StableDiffusionControlnetPipeline *pipe = nullptr;
    Worker worker;
};