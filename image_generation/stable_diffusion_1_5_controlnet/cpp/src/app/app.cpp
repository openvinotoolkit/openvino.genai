#include <stdio.h>
#include <string>
#include <filesystem>
#include <random>

#include "gui.hpp"
#include "worker.hpp"
#include "tinyfiledialogs.h"

#include "utils.hpp"
#include "imwrite.hpp"

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

uint32_t gen_seed() {
    std::random_device rd;                                       
    std::mt19937_64 gen(rd());                                   
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    return dis(gen);
}


ov::Tensor postprocess_image(ov::Tensor decoded_image) {
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());
    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

std::string openFileDialog() {
    const char* filters[] = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
    const char* filePath = tinyfd_openFileDialog("Select an Image",  // Dialog title
                                                 "",                 // Default path
                                                 1,                  // Number of filters
                                                 filters,            // Filters
                                                 "Image files",  // Filter description
                                                 1               // Single selection
    );

    return filePath ? filePath : "";  // Return empty string if canceled
}



bool validate_directory(const std::string& path) {
    bool has_xml = false;
    bool has_bin = false;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".xml") {
            has_xml = true;
        } else if (entry.path().extension() == ".bin") {
            has_bin = true;
        }
        if (has_xml && has_bin) {
            return true;
        }
    }
    return false;
}

std::string openFolderDialog() {
    const char* path = tinyfd_selectFolderDialog("Select Model Directory", "");
    if (path && validate_directory(path)) {
        return path;
    }
    return "";
}

int App::Init() {
    if (!glfwInit())
        return 1;

    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(1200, 800, "Stable Diffusion Controlnet Demo", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    glfwGetWindowContentScale(window, &xscale, &yscale);
    
    // resize window
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int new_width = static_cast<int>(width * xscale);
    int new_height = static_cast<int>(height * yscale);
    glfwSetWindowSize(window, new_width, new_height);

    // resize fonts and all
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    io.Fonts->AddFontDefault();
    io.FontGlobalScale = xscale;
    ImGui::GetStyle().ScaleAllSizes(xscale);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // init ov
    ov::Core core;
    state.devices = core.get_available_devices();
    state.active_device_index = 0;

    // init worker
    worker.Start();

    return 0;
}

int App::Clean() {
    worker.Stop();

    if (preview_state.preview_texture)
        glDeleteTextures(1, &preview_state.preview_texture);

    if (result_state.texture)
        glDeleteTextures(1, &result_state.texture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

int App::Run() {
    while (!glfwWindowShouldClose(window)) {
        Render();
    }

    Clean();
    return 0;
}

void App::LoadInputImageData() {
    if (preview_state.preview_texture)
        glDeleteTextures(1, &preview_state.preview_texture);

    auto image_tensor = read_image_to_tensor(preview_state.image_path.c_str());
    preview_state.image_height = image_tensor.get_shape()[1];
    preview_state.image_width = image_tensor.get_shape()[2];
    int channels = image_tensor.get_shape()[3];

    auto* image_data = image_tensor.data<uint8_t>();

    glGenTextures(1, &preview_state.preview_texture);
    glBindTexture(GL_TEXTURE_2D, preview_state.preview_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 (channels == 4 ? GL_RGBA : GL_RGB),
                 preview_state.image_width,
                 preview_state.image_height,
                 0,
                 (channels == 4 ? GL_BGRA_EXT : GL_BGR_EXT),
                 GL_UNSIGNED_BYTE,
                 image_data);
}

void App::RenderLeftPanel() {
    ImGui::BeginChild("LeftPanel", ImVec2(400 * xscale, 0), true);
    ImGui::Text("Options");

    // prompt & negative prompt
    ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
    ImGui::Text("Prompt");
    ImGui::InputTextMultiline("##prompt",
                              state.prompt,
                              IM_ARRAYSIZE(state.prompt),
                              ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 8),
                              flags);
    ImGui::Text("Negative Prompt");
    ImGui::InputTextMultiline("##negative_prompt",
                              state.negative_prompt,
                              IM_ARRAYSIZE(state.negative_prompt),
                              ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 8),
                              flags);
    // steps
    ImGui::InputInt("Steps", &state.steps);

    // seeds
    int64_t seed_min = -1;
    int64_t seed_max = 4294967295;
    ImGui::DragScalar("Seed",
                      ImGuiDataType_S64,
                      &state.seed,
                      1,
                      &seed_min,
                      &seed_max,
                      "%ld",
                      ImGuiSliderFlags_AlwaysClamp);

    // conditioning image
    if (ImGui::Button("Select Image")) {
        std::string next_image_path = openFileDialog();
        if (!next_image_path.empty()) {
            if (preview_state.image_path != next_image_path) {
                preview_state.image_path = next_image_path;
                preview_state.should_load = true;
            }
        }
    }
     ImGui::SameLine();
     ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0 / 7.0f, 0.6f, 0.6f));
     ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0 / 7.0f, 0.7f, 0.7f));
     ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0/ 7.0f, 0.8f, 0.8f));
     if (ImGui::Button("Remove")) {
         preview_state.image_path = "";
         preview_state.should_load = false;
         if (preview_state.preview_texture) {
             glDeleteTextures(1, &preview_state.preview_texture);
             preview_state.preview_texture = 0;
         }
     }
     ImGui::PopStyleColor(3);
     ImGui::PopID();

    if (preview_state.should_load) {
        // load image data and construct texture
        LoadInputImageData();
        preview_state.should_load = false;
    }

    if (preview_state.preview_texture) {
        ImGui::Text("Controlnet Image: %s", preview_state.image_path.c_str());
        float aspect_ratio = (float)preview_state.image_width / preview_state.image_height;
        ImVec2 preview_size(250*xscale, 250*yscale);

        if (aspect_ratio > 1.0f) {
            // Image is wider than tall, limit by width
            preview_size.y = preview_size.x / aspect_ratio;
        } else {
            // Image is taller than wide, limit by height
            preview_size.x = preview_size.y * aspect_ratio;
        }
        ImGui::Image((void*)(intptr_t)preview_state.preview_texture, preview_size);
    }

    ImGui::EndChild();
}

void App::LoadResultImageData() {
    if (result_state.texture)
        glDeleteTextures(1, &result_state.texture);

    auto image_tensor = result_state.image;
    result_state.image_height = image_tensor.get_shape()[1];
    result_state.image_width = image_tensor.get_shape()[2];
    int channels = image_tensor.get_shape()[3];

    auto* image_data = image_tensor.data<uint8_t>();

    glGenTextures(1, &result_state.texture);
    glBindTexture(GL_TEXTURE_2D, result_state.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 (channels == 4 ? GL_RGBA : GL_RGB),
                 result_state.image_width,
                 result_state.image_height,
                 0,
                 (channels == 4 ? GL_RGBA : GL_RGB),
                 GL_UNSIGNED_BYTE,
                 image_data);
}


void App::RenderRightPanel() {
    ImGui::BeginChild("RightPanel", ImVec2(0, 0), true);
    std::vector<const char*> items;
    for (int i = 0; i < state.devices.size(); ++i)
        items.push_back(state.devices[i].c_str());

    ImGui::Combo("device", &state.active_device_index, items.data(), items.size());

    ImGui::Text("Model Path: ");
    ImGui::SameLine();
    if (state.model_path.empty()) {
        if (ImGui::Button("...")) {
            std::string model_path = openFolderDialog();
            if (model_path.empty()) {
                ImGui::Text("model path must contains *.xml and *.bin");
            } else {
                state.model_path = model_path;
                worker.Request([this] {
                    pipe = new StableDiffusionControlnetPipeline(state.model_path,
                                                                 state.devices[state.active_device_index]);
                });
            }
        }
    } else {
        ImGui::Text("%s", state.model_path.c_str());
        if (pipe == nullptr) {
            ImGui::Text("Loading..");
        }
    }
    if (pipe != nullptr) {
        
        if (running) {
            ImGui::Text("Running..");
            if (result_state.texture) {
                glDeleteTextures(1, &result_state.texture);
                result_state.texture = 0;
            }
        } else {
            ImGui::Text("Ready");
            if (ImGui::Button("Run")) {
                worker.Request([this] {
                    running = true;
                    uint32_t input_seed;
                    if (state.seed != -1) {
                        input_seed = state.seed;
                    } else {
                        input_seed = gen_seed();
                    }
                    StableDiffusionControlnetPipelineParam param = {
                        state.prompt,
                        state.negative_prompt,
                        preview_state.image_path,
                        state.steps,
                        input_seed,
                    };
                    auto decoded_image = pipe->Run(param);
                    result_state.image = postprocess_image(decoded_image);

                    // will save hisotry calls here
                    const std::string folder_name = "images";
                    try {
                        if (!std::filesystem::exists(folder_name)) {
                            std::cout << "Directory does not exist, creating: " << folder_name << std::endl;
                            std::filesystem::create_directory(folder_name);
                        } 
                        imwrite(std::string("./images/seed_") + std::to_string(input_seed) + ".bmp",
                                result_state.image,
                                true);
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to create dir" << e.what() << std::endl;
                    }

                    result_state.should_render = true;
                    running = false;
                });
            }
        }
    }
    if (result_state.should_render) {
        LoadResultImageData();
        result_state.should_render = false;
    }

    if (result_state.texture) {
        float aspect_ratio = (float)result_state.image_width / result_state.image_height;
        ImVec2 preview_size(512*xscale, 512*yscale);

        if (aspect_ratio > 1.0f) {
            // Image is wider than tall, limit by width
            preview_size.y = preview_size.x / aspect_ratio;
        } else {
            // Image is taller than wide, limit by height
            preview_size.x = preview_size.y * aspect_ratio;
        }
        ImGui::Image((void*)(intptr_t)result_state.texture, preview_size);
    }

    ImGui::EndChild();
}

void App::Render() {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(display_w, display_h));

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("Stable Diffusion Controlnet", nullptr, window_flags);

    RenderLeftPanel();
    ImGui::SameLine();
    RenderRightPanel();

    ImGui::End();

    ImGui::Render();

    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

int main(int, char**)
{
    App app;

    if (app.Init()) {
        fprintf(stderr, "Failed to init app\n");
        return -1;
    }

    return app.Run();
}