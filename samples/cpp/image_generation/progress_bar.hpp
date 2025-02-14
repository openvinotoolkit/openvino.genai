// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "indicators/progress_bar.hpp"

bool progress_bar(size_t step, size_t num_steps, ov::Tensor& /* latent */) {
    using namespace indicators;

    static std::optional<ProgressBar> bar;

    if (!bar) {
        bar.emplace(
            option::BarWidth{50},
            option::ForegroundColor{Color::green},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true}
        );
    }

    std::stringstream stream;
    stream << "Image generation step " << (step + 1) << " / " << num_steps;

    bar->set_option(option::PostfixText{stream.str()});
    bar->set_progress((100 * (step + 1)) / num_steps);

    if (step + 1 == num_steps) {
        bar.reset();  // Required when multiple progress bars are used, without recreation of the object the second progress bar won't be displayed correctly
    }

    return false;
}
