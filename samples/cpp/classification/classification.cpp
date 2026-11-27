// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../visual_language_chat/load_image.hpp"
#include "openvino/genai/classification/classification_pipeline.hpp"

// DIAG (temporary, do not merge): locate the Windows-only crash of this sample.
#define DIAG(message) (std::cerr << "DIAG " << message << std::endl)

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>

#include <cstdint>

namespace {
bool diag_env(const char* name) {
    return GetEnvironmentVariableA(name, nullptr, 0) > 0;
}

std::string module_of(const void* address, uintptr_t& offset) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(address),
                            &module)) {
        offset = reinterpret_cast<uintptr_t>(address);
        return "unknown";
    }
    char path[MAX_PATH] = {};
    GetModuleFileNameA(module, path, MAX_PATH);
    offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module);
    return path;
}

void print_frame(size_t index, const void* address) {
    uintptr_t offset = 0;
    const std::string name = module_of(address, offset);
    std::cerr << "DIAG frame " << index << ' ' << name << " +0x" << std::hex << offset << std::dec << std::endl;
}

LONG WINAPI crash_filter(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD& record = *info->ExceptionRecord;
    std::cerr << "DIAG crash code=0x" << std::hex << record.ExceptionCode << " address=" << record.ExceptionAddress
              << std::dec;
    if (record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2) {
        // 0 - read, 1 - write, 8 - execute (DEP)
        std::cerr << " access_type=" << record.ExceptionInformation[0] << " target=0x" << std::hex
                  << record.ExceptionInformation[1] << std::dec;
    }
    std::cerr << std::endl;
#if defined(_M_X64)
    CONTEXT context = *info->ContextRecord;
    for (size_t frame = 0; frame < 32 && context.Rip != 0; ++frame) {
        print_frame(frame, reinterpret_cast<const void*>(context.Rip));
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (entry == nullptr) {
            // A leaf function without unwind info: the return address is on top of the stack.
            context.Rip = *reinterpret_cast<DWORD64*>(context.Rsp);
            context.Rsp += sizeof(DWORD64);
            continue;
        }
        PVOID handler_data = nullptr;
        DWORD64 establisher_frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER,
                         image_base,
                         context.Rip,
                         entry,
                         &context,
                         &handler_data,
                         &establisher_frame,
                         nullptr);
    }
#endif
    return EXCEPTION_CONTINUE_SEARCH;
}

// Prints the loaded modules outside of the Windows directory: shows which OpenVINO binaries the process uses.
void print_modules() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL ok = Module32FirstW(snapshot, &entry); ok; ok = Module32NextW(snapshot, &entry)) {
        char path[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_UTF8, 0, entry.szExePath, -1, path, static_cast<int>(sizeof(path)), nullptr, nullptr);
        if (std::string(path).find("\\Windows\\") == std::string::npos) {
            std::cerr << "DIAG module " << path << std::endl;
        }
    }
    CloseHandle(snapshot);
}
}  // namespace

#define DIAG_SETUP() SetUnhandledExceptionFilter(crash_filter)
#define DIAG_MODULES() print_modules()
#else
namespace {
bool diag_env(const char* name) {
    return std::getenv(name) != nullptr;
}
}  // namespace

#define DIAG_SETUP()
#define DIAG_MODULES()
#endif

int main(int argc, char* argv[]) try {
    DIAG_SETUP();
    std::atexit([] {
        DIAG("atexit handlers running");
    });
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<TEXT>' [<IMAGE_FILE> ...]");
    }

    const std::string models_path = argv[1];
    const std::string text = argv[2];
    // CLS_DIAG_IMAGE_3D: pass [H, W, 3] views of the loaded images like the Python sample does.
    const bool image_3d = diag_env("CLS_DIAG_IMAGE_3D");
    DIAG("start, image_3d=" << image_3d);
    std::vector<ov::Tensor> source_images;  // own the stb buffers behind the 3D views
    std::vector<ov::Tensor> images;
    for (int arg_idx = 3; arg_idx < argc; ++arg_idx) {
        ov::Tensor image = utils::load_image(argv[arg_idx]);
        DIAG("loaded " << argv[arg_idx] << " shape " << image.get_shape());
        if (image_3d) {
            const ov::Shape shape = image.get_shape();
            ov::Tensor view(image.get_element_type(), ov::Shape{shape[1], shape[2], shape[3]}, image.data());
            source_images.push_back(std::move(image));
            images.push_back(std::move(view));
        } else {
            images.push_back(std::move(image));
        }
    }

    const std::string device = "CPU";  // GPU can be used as well

    DIAG("creating pipeline");
    ov::genai::ClassificationPipeline pipeline(models_path, device);
    DIAG("pipeline created");
    DIAG_MODULES();

    // Images are referenced in the text by <ov_genai_image_N> tags; untagged images are prepended to the prompt.
    const ov::genai::ClassificationResult result = pipeline.classify(text, images);
    DIAG("classified");

    std::cout << std::fixed << std::setprecision(4);
    const std::vector<float>& scores = result.scores.at(0);
    for (size_t label_id = 0; label_id < scores.size(); ++label_id) {
        std::cout << label_id << ": " << scores[label_id] << '\n';
    }
    std::cout << std::defaultfloat << std::flush;
    DIAG("printed, leaving main");

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
