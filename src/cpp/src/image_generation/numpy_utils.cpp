#include "image_generation/numpy_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace numpy_utils {

void rescale_zero_terminal_snr(std::vector<float>& betas) {
    // Convert betas to alphas_bar_sqrt
    std::vector<float> alphas, alphas_bar_sqrt;
    for (float b : betas) {
        alphas.push_back(1.0f - b);
    }

    for (size_t i = 1; i <= alphas.size(); ++i) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        alphas_bar_sqrt.push_back(std::sqrt(alpha_cumprod));
    }

    float alphas_bar_sqrt_0 = alphas_bar_sqrt[0];
    float alphas_bar_sqrt_T = alphas_bar_sqrt[alphas_bar_sqrt.size() - 1];

    for (float& x : alphas_bar_sqrt) {
        // Shift so the last timestep is zero.
        x = x - alphas_bar_sqrt_T;
        // Scale so the first timestep is back to the old value.
        x *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T);
        // Revert sqrt
        x = std::pow(x, 2);
    }

    // Revert cumprod
    std::vector<float> end = alphas_bar_sqrt, begin = alphas_bar_sqrt;
    end.erase(end.begin());
    begin.pop_back();

    alphas[0] = alphas_bar_sqrt[0];
    for (size_t i = 1; i < alphas.size(); ++i) {
        alphas[i] = end[i - 1] / begin[i - 1];
    }

    std::transform(alphas.begin(), alphas.end(), betas.begin(), [](float x) {
        return (1 - x);
    });
}

std::vector<float> interp(const std::vector<std::int64_t>& x, const std::vector<size_t>& xp, const std::vector<float>& fp) {
    OPENVINO_ASSERT(xp.size() == fp.size(), "`xp` and `fp`vectors must have the same sizes");

    std::vector<float> interp_res;

    for (const auto& i : x) {
        if (i <= xp[0]) {
            interp_res.push_back(fp[0]);
        } else if (i >= xp[xp.size() - 1]) {
            interp_res.push_back(fp[fp.size() - 1]);
        } else {
            // Find the first xp element that is not less than x[i]
            auto it = std::lower_bound(xp.begin(), xp.end(), i);

            // idx of the left boundary
            size_t idx = std::distance(xp.begin(), it) - 1;

            float x0 = xp[idx], x1 = xp[idx + 1];
            float y0 = fp[idx], y1 = fp[idx + 1];

            float interp_val = (y1 - y0) / (x1 - x0) * (i - x0) + y0;

            interp_res.push_back(interp_val);
        }
    }

    return interp_res;
}

void concat_3d_by_rows(const float* data_1, const float* data_2, float* res, const ov::Shape shape_1, const ov::Shape shape_2) {
    OPENVINO_ASSERT(shape_1.size() == 3 && shape_2.size() == 3, "Shape dimensions must be 3");
    OPENVINO_ASSERT(shape_1[0] == shape_2[0] && shape_1[1] == shape_2[1], "Tensors for concatenation must have the same dimensions");

    for (size_t i = 0; i < shape_1[0]; ++i) {
        for (size_t j = 0; j < shape_1[1]; ++j) {
            size_t offset_1 = (i * shape_1[1] + j) * shape_1[2];
            size_t offset_2 = (i * shape_2[1] + j) * shape_2[2];

            size_t step = (i * shape_1[1] + j) * (shape_1[2] + shape_2[2]);

            std::memcpy(res + step, data_1 + offset_1, shape_1[2] * sizeof(float));
            std::memcpy(res + step + shape_1[2], data_2 + offset_2, shape_2[2] * sizeof(float));
        }
    }
}

void concat_2d_by_rows(const float* data_1, const float* data_2, float* res, const ov::Shape shape_1, const ov::Shape shape_2) {
    OPENVINO_ASSERT(shape_1.size() == 2 && shape_2.size() == 2, "Shape dimensions must be 2");
    OPENVINO_ASSERT(shape_1[0] == shape_2[0], "Tensors for concatenation must have the same dimensions");

    for (size_t i = 0; i < shape_1[0]; ++i) {
        size_t offset_1 = i * shape_1[1];
        size_t offset_2 = i * shape_2[1];

        size_t step = i * (shape_1[1] + shape_2[1]);

        std::memcpy(res + step, data_1 + offset_1, shape_1[1] * sizeof(float));
        std::memcpy(res + step + shape_1[1],
                    data_2 + offset_2,
                    shape_2[1] * sizeof(float));
    }
}

void concat_3d_by_cols(const float* data_1, const float* data_2, float* res, const ov::Shape shape_1, const ov::Shape shape_2) {
    OPENVINO_ASSERT(shape_1.size() == 3 && shape_2.size() == 3, "Shape dimensions must be 3");
    OPENVINO_ASSERT(shape_1[0] == shape_2[0] && shape_1[2] == shape_2[2], "Tensors for concatenation must have the same dimensions");

    for (size_t i = 0; i < shape_1[0]; ++i) {
        size_t shift_1 = i * shape_1[1] * shape_1[2];
        size_t shift_2 = i * shape_2[1] * shape_2[2];

        size_t step = shift_1 + shift_2;

        std::memcpy(res + step, data_1 + shift_1, shape_1[1] * shape_1[2] * sizeof(float));
        std::memcpy(res + step + shape_1[1] * shape_1[2], data_2 + shift_2, shape_2[1] * shape_2[2] * sizeof(float));
    }
}

void concat_3d_by_channels(const float* data_1, const float* data_2, float* res, const ov::Shape shape_1, const ov::Shape shape_2) {
    OPENVINO_ASSERT(shape_1.size() == 3 && shape_2.size() == 3, "Shape dimensions must be 3");
    OPENVINO_ASSERT(shape_1[1] == shape_2[1] && shape_1[2] == shape_2[2], "Tensors for concatenation must have the same dimensions");

    size_t size_1 = shape_1[0] * shape_1[1] * shape_1[2];
    size_t size_2 = shape_2[0] * shape_2[1] * shape_2[2];

    std::memcpy(res, data_1, size_1 * sizeof(float));
    std::memcpy(res + size_1, data_2, size_2 * sizeof(float));
}

void concat_2d_by_channels(const float* data_1, const float* data_2, float* res, const ov::Shape shape_1, const ov::Shape shape_2) {
    OPENVINO_ASSERT(shape_1.size() == 2 && shape_2.size() == 2, "Shape dimensions must be 2");
    OPENVINO_ASSERT(shape_1[1] == shape_2[1], "Tensors for concatenation must have the same dimensions");

    size_t size_1 = shape_1[0] * shape_1[1];
    size_t size_2 = shape_2[0] * shape_2[1];

    std::memcpy(res, data_1, size_1 * sizeof(float));
    std::memcpy(res + size_1, data_2, size_2 * sizeof(float));
}


} // namespace ov
} // namespace genai
} // namespace numpy_utils
