// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace ov::genai::funasr_frontend {

/**
 * Applies normalized = (feature + shift) * scale with `<AddShift>` and `<Rescale>` vectors from a Kaldi Nnet file.
 */
class CMVN {
public:
    CMVN(const std::filesystem::path& cmvn_file, size_t feature_size);

    /**
     * @param rows row_count * feature_size float values, normalized in place.
     */
    void apply(float* rows, size_t row_count) const;

private:
    std::vector<float> m_shifts;
    std::vector<float> m_scales;
};

}  // namespace ov::genai::funasr_frontend
