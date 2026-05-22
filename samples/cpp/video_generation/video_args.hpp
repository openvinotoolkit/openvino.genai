// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <openvino/core/except.hpp>

namespace video_args {

struct Options {
    std::optional<int64_t> height;
    std::optional<int64_t> width;
    std::optional<int64_t> num_frames;
    std::optional<int64_t> num_inference_steps;
    std::vector<std::string> positional;
};

inline int64_t parse_int(const char* flag, const char* value) {
    OPENVINO_ASSERT(value != nullptr, "Missing value for ", flag);
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    OPENVINO_ASSERT(end != value && *end == '\0', "Invalid integer value for ", flag, ": ", value);
    return static_cast<int64_t>(parsed);
}

// Parse --height/--width/--num-frames/--num-inference-steps flags from argv.
// Anything else stays in `positional` (in original order). Flags can appear in
// any position relative to positional arguments.
inline Options parse(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_value = [&](const char* flag) {
            OPENVINO_ASSERT(i + 1 < argc, "Missing value for ", flag);
            return argv[++i];
        };
        if (arg == "--height") {
            opts.height = parse_int("--height", take_value("--height"));
        } else if (arg == "--width") {
            opts.width = parse_int("--width", take_value("--width"));
        } else if (arg == "--num-frames") {
            opts.num_frames = parse_int("--num-frames", take_value("--num-frames"));
        } else if (arg == "--num-inference-steps") {
            opts.num_inference_steps = parse_int("--num-inference-steps", take_value("--num-inference-steps"));
        } else {
            opts.positional.push_back(arg);
        }
    }
    return opts;
}

}  // namespace video_args
