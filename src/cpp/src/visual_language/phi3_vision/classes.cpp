
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/phi3_vision/classes.hpp"

#include "visual_language/clip.hpp"
#include "openvino/opsets/opset13.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

constexpr size_t INPUT_IMAGE_SIZE = 336;
const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>\n";
}

ov::Tensor padding_336(const ov::Tensor& unpadded) {
    ov::Shape _1ss3 = unpadded.get_shape();
    size_t s1 = _1ss3.at(1), s2 = _1ss3.at(2);
    if (s1 < s2) {
        size_t tar = size_t(std::ceil(float(s1) / INPUT_IMAGE_SIZE) * INPUT_IMAGE_SIZE);
        size_t top_padding = (tar - s1) / 2;
        ov::Tensor padded{ov::element::u8, {1, tar, s2, 3}};
        uint8_t* padded_data = padded.data<uint8_t>();
        std::fill_n(padded_data, padded.get_size(), 255);
        std::copy_n(unpadded.data<uint8_t>(), unpadded.get_size(), padded_data + top_padding * s2 * 3);
        return padded;
    }
    size_t tar = size_t(std::ceil(float(s2) / INPUT_IMAGE_SIZE) * INPUT_IMAGE_SIZE);
    size_t left_padding = (tar - s2) / 2;
    ov::Tensor padded{ov::element::u8, {1, s1, tar, 3}};
    uint8_t* padded_data = padded.data<uint8_t>();
    std::fill_n(padded_data, padded.get_size(), 255);
    auto unpadded_data = unpadded.data<uint8_t>();
    for (size_t row = 0; row < s1; ++row) {
        std::copy_n(unpadded_data + row * s2 * 3, s2 * 3, padded_data + row * tar * 3 + left_padding * 3);
    }
    return padded;
}

ov::Tensor HD_transform(const ov::Tensor& uint8, size_t num_crops) {
    ov::Shape _1hwc = uint8.get_shape();
    size_t height = _1hwc.at(1), width = _1hwc.at(2);
    bool trans = false;
    if (width < height) {
        std::swap(height, width);
        trans = true;
    }
    float ratio = float(width) / height;
    unsigned scale = 1;
    while (scale * std::ceil(scale / ratio) <= num_crops) {
        ++scale;
    }
    --scale;
    OPENVINO_ASSERT(scale > 0);
    size_t new_w = scale * INPUT_IMAGE_SIZE;
    size_t new_h = new_w / ratio;
    clip_image_u8 src{}, dst{};
    auto uint8_data = uint8.data<uint8_t>();
    if (trans) {
        src = clip_image_u8{int(height), int(width), {uint8_data, uint8_data + uint8.get_size()}};
        bilinear_resize(src, dst, new_h, new_w);
        return padding_336(ov::Tensor{ov::element::u8, {1, new_w, new_h, 3}, dst.buf.data()});
    }
    src = clip_image_u8{int(width), int(height), {uint8_data, uint8_data + uint8.get_size()}};
    bilinear_resize(src, dst, new_w, new_h);
    return padding_336(ov::Tensor{ov::element::u8, {1, new_h, new_w, 3}, dst.buf.data()});
}

void bicubic_resize_phi3(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const float tx = static_cast<float>(nx) / static_cast<float>(target_width);
    const float ty = static_cast<float>(ny) / static_cast<float>(target_height);

    constexpr float _1_3 = 1.0f / 3.0f;
    constexpr float _1_6 = 1.0f / 6.0f;

    float pixels[4];

    auto clip_coord = [](int x, int lower, int upper) -> int {
        return std::max(lower, std::min(x, upper));
    };

    for (int i = 0; i < target_height; i++) {
        const float fy = ty * i;
        const int y = static_cast<int>(fy);
        const float dy = fy - y;

        const int y_coords[4] = {
            clip_coord(y - 1, 0, ny - 1),
            clip_coord(y, 0, ny - 1),
            clip_coord(y + 1, 0, ny - 1),
            clip_coord(y + 2, 0, ny - 1)
        };

        for (int j = 0; j < target_width; j++) {
            const float fx = tx * j;
            const int x = static_cast<int>(fx);
            const float dx = fx - x;

            const int x_coords[4] = {
                clip_coord(x - 1, 0, nx - 1),
                clip_coord(x, 0, nx - 1),
                clip_coord(x + 1, 0, nx - 1),
                clip_coord(x + 2, 0, nx - 1)
            };

            const int dst_base_idx = (i * target_width + j) * 3;

            for (int k = 0; k < 3; k++) {
                for (int jj = 0; jj < 4; jj++) {
                    const int row_base = y_coords[jj] * nx;
                    const uint8_t* row_ptr = &img.buf[row_base * 3 + k];

                    const float p[4] = {
                        static_cast<float>(row_ptr[x_coords[0] * 3]),
                        static_cast<float>(row_ptr[x_coords[1] * 3]),
                        static_cast<float>(row_ptr[x_coords[2] * 3]),
                        static_cast<float>(row_ptr[x_coords[3] * 3])
                    };

                    const float a0 = p[1];
                    const float d0 = p[0] - a0;
                    const float d2 = p[2] - a0;
                    const float d3 = p[3] - a0;
                    const float a1 = -_1_3 * d0 + d2 - _1_6 * d3;
                    const float a2 = 0.5f * (d0 + d2);
                    const float a3 = -_1_6 * d0 - 0.5f * d2 + _1_6 * d3;

                    pixels[jj] = a0 + dx * (a1 + dx * (a2 + dx * a3));
                }

                const float a0 = pixels[1];
                const float d0 = pixels[0] - a0;
                const float d2 = pixels[2] - a0;
                const float d3 = pixels[3] - a0;
                const float a1 = -_1_3 * d0 + d2 - _1_6 * d3;
                const float a2 = 0.5f * (d0 + d2);
                const float a3 = -_1_6 * d0 - 0.5f * d2 + _1_6 * d3;

                const float result = a0 + dy * (a1 + dy * (a2 + dy * a3));

                dst.buf[dst_base_idx + k] = static_cast<uint8_t>(
                    std::min(std::max(std::round(result), 0.0f), 255.0f)
                );
            }
        }
    }
}

ov::Tensor mean_scale(const ov::Tensor& uint8, const ProcessorConfig& config) {
    auto uint_8_data = uint8.data<uint8_t>();
    ov::Tensor float_normalized{ov::element::f32, uint8.get_shape()};
    float* float_data = float_normalized.data<float>();
    OPENVINO_ASSERT(0 == uint8.get_size() % 3, "RGB");
    for (size_t idx = 0; idx < uint8.get_size(); idx += 3) {
        float_data[idx] = (float(uint_8_data[idx]) / 255.0f - config.image_mean[0]) / config.image_std[0];
        float_data[idx + 1] = (float(uint_8_data[idx + 1]) / 255.0f - config.image_mean[1]) / config.image_std[1];
        float_data[idx + 2] = (float(uint_8_data[idx + 2]) / 255.0f - config.image_mean[2]) / config.image_std[2];
    }
    return float_normalized;
}

ov::Tensor channels_first(const ov::Tensor& _1hw3) {
    const ov::Shape shape = _1hw3.get_shape();
    const size_t height = shape.at(1);
    const size_t width = shape.at(2);
    const size_t hw = height * width;

    ov::Tensor _13hw = ov::Tensor{ov::element::f32, {1, 3, height, width}};
    const float* _1hw3_data = _1hw3.data<float>();
    float* _13hw_data = _13hw.data<float>();

    float* dst_channels[3] = {
        _13hw_data,              // R channel
        _13hw_data + hw,         // G channel
        _13hw_data + 2 * hw      // B channel
    };

    for (size_t row = 0; row < height; ++row) {
        const size_t row_offset = row * width;
        const float* src_row = _1hw3_data + row_offset * 3;
        for (size_t col = 0; col < width; ++col) {
            const size_t dst_offset = row_offset + col;
            const size_t src_offset = col * 3;
            dst_channels[0][dst_offset] = src_row[src_offset];     // R
            dst_channels[1][dst_offset] = src_row[src_offset + 1]; // G
            dst_channels[2][dst_offset] = src_row[src_offset + 2]; // B
        }
    }
    return _13hw;
}

// Reimplementation of Python im.reshape(1, 3, h//336, 336, w//336, 336).permute(0,2,4,1,3,5).reshape(-1, 3, 336, 336)
ov::Tensor slice_image(const ov::Tensor& image) {
    ov::Shape shape = image.get_shape();
    size_t N = shape[0];
    size_t C = shape[1];
    size_t H = shape[2];
    size_t W = shape[3];

    size_t num_h_slices = H / INPUT_IMAGE_SIZE;
    size_t num_w_slices = W / INPUT_IMAGE_SIZE;

    // Step 1: Define and populate the reshaped tensor in the correct shape order
    ov::Tensor reshaped{ov::element::f32, {N, num_h_slices, num_w_slices, C, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE}};
    float* reshaped_data = reshaped.data<float>();
    auto image_data = image.data<float>();

    // Populate the reshaped tensor
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < num_h_slices; ++h) {
            for (size_t w = 0; w < num_w_slices; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t i = 0; i < INPUT_IMAGE_SIZE; ++i) {
                        for (size_t j = 0; j < INPUT_IMAGE_SIZE; ++j) {
                            size_t src_idx = n * C * H * W + c * H * W + (h * INPUT_IMAGE_SIZE + i) * W + (w * INPUT_IMAGE_SIZE + j);
                            size_t dst_idx = n * num_h_slices * num_w_slices * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                h * num_w_slices * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                w * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                c * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                i * INPUT_IMAGE_SIZE + j;
                            reshaped_data[dst_idx] = image_data[src_idx];
                        }
                    }
                }
            }
        }
    }

    // Step 2: Define the permuted tensor in the final shape
    ov::Tensor permuted{ov::element::f32, {N * num_h_slices * num_w_slices, C, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE}};
    float* permuted_data = permuted.data<float>();

    // Perform permutation by flattening N, num_h_slices, and num_w_slices
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < num_h_slices; ++h) {
            for (size_t w = 0; w < num_w_slices; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t i = 0; i < INPUT_IMAGE_SIZE; ++i) {
                        for (size_t j = 0; j < INPUT_IMAGE_SIZE; ++j) {
                            size_t src_idx = n * num_h_slices * num_w_slices * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                h * num_w_slices * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                w * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                c * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                i * INPUT_IMAGE_SIZE + j;
                            size_t dst_idx = (n * num_h_slices * num_w_slices + h * num_w_slices + w) * C * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                c * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE +
                                                i * INPUT_IMAGE_SIZE + j;
                            permuted_data[dst_idx] = reshaped_data[src_idx];
                        }
                    }
                }
            }
        }
    }

    return permuted;
}

ov::Tensor concatenate_batch(const ov::Tensor& float_first, const ov::Tensor& float_second) {
    ov::Shape shape_first = float_first.get_shape();
    ov::Shape shape_second = float_second.get_shape();
    OPENVINO_ASSERT(shape_first.at(1) == shape_second.at(1), "Channels must be the same");
    OPENVINO_ASSERT(shape_first.at(2) == shape_second.at(2), "Height must be the same");
    OPENVINO_ASSERT(shape_first.at(3) == shape_second.at(3), "Width must be the same");
    ov::Tensor concatenated{ov::element::f32, {shape_first.at(0) + shape_second.at(0), shape_first.at(1), shape_first.at(2), shape_first.at(3)}};
    float* concatenated_data = concatenated.data<float>();
    auto first_data = float_first.data<float>();
    auto second_data = float_second.data<float>();
    std::copy(first_data, first_data + float_first.get_size(), concatenated_data);
    std::copy(second_data, second_data + float_second.get_size(), concatenated_data + float_first.get_size());
    return concatenated;
}

ov::Tensor pad_to_max_num_crops_tensor(const ov::Tensor& nchw, size_t max_crops) {
    ov::Shape shape = nchw.get_shape();
    size_t num_crops = shape[0];
    if (num_crops >= max_crops) {
        return nchw;
    }
    ov::Tensor padded{ov::element::f32, {max_crops, shape[1], shape[2], shape[3]}};
    float* padded_data = padded.data<float>();
    auto nchw_data = nchw.data<float>();
    std::copy_n(nchw_data, nchw.get_size(), padded_data);
    return padded;
}

std::tuple<ov::Tensor, ImageSize> get_pixel_values_phi3_v(const ov::Tensor& image, const ProcessorConfig& config) {
    ov::Tensor hd_image = HD_transform(image, config.phi3_v.num_crops);
    ImageSize image_size{hd_image.get_shape().at(2), hd_image.get_shape().at(1)};
    clip_image_u8 img{int(hd_image.get_shape().at(2)), int(hd_image.get_shape().at(1)), {hd_image.data<uint8_t>(), hd_image.data<uint8_t>() + hd_image.get_size()}};
    clip_image_u8 dst;
    bicubic_resize_phi3(img, dst, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
    ov::Tensor global_image{ov::element::u8, {1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3}, dst.buf.data()};
    global_image = mean_scale(global_image, config);
    hd_image = mean_scale(hd_image, config);
    global_image = channels_first(global_image);
    hd_image = channels_first(hd_image);
    ov::Tensor slices = slice_image(hd_image);
    ov::Tensor concatenated = concatenate_batch(global_image, slices);
    ov::Tensor pixel_values = pad_to_max_num_crops_tensor(concatenated, config.phi3_v.num_crops);
    return {std::move(pixel_values), image_size};
}

// Reimplementation of python
// N, L, C = image_features.shape
// assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
// num_images = N // (h_crop * w_crop)
// H = int(L**0.5)
// print(L, H)
// image_features_hd = (
//     image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
//     .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
//     .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
//     .reshape(N, -1, 4 * C)  # N, 144, 4096
//     .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
//     .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
//     .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
// )
// Obtained in the following way
// import torch
// import openvino as ov
// import numpy as np
// class Model(torch.nn.Module):
//     def forward(self, image_features, h_crop, w_crop):
//         """
//         image_features: (num_images*num_crops, 24*24, 1024)
//         output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
//         """
//         N, L, C = image_features.shape
//         num_images = N // (h_crop * w_crop)
//         H = (torch.tensor(L, dtype=torch.float32)**0.5).int()
//         image_features_hd = (
//             image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
//             .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
//             .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
//             .reshape(N, -1, 4 * C)  # N, 144, 4096
//             .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
//             .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
//             .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
//         return {"o": image_features_hd}
// model = Model()
// example_input = {"image_features": torch.rand((4, 576, 1024), dtype=torch.float32), "h_crop": torch.tensor(2, dtype=torch.int32), "w_crop": torch.tensor(2, dtype=torch.int32)}
// ov_model = ov.convert_model(model, example_input=example_input, input=ov.PartialShape([-1, 576, 1024]))
// # ov_model.outputs[0].get_tensor().set_names({"out"})
// ov.save_model(ov_model, "reshape_hd_patches_2x2merge.xml")
// inp = np.arange(4 * 576 * 1024).reshape([4, 576, 1024])
// test = ov.Core().compile_model(ov_model, "CPU")
// print(ov_model)
// print(test([inp, 2, 2])["o"].flatten())
// 2. Run https://github.com/slyalin/openvino_devtools/blob/bcd4a51b1354b24b2316ac3e1c77b2f87ae7a497/openvino_devtools/ov2py.py with the IR.
// 3. Translate the printed Python implementation to C++.
ov::CompiledModel create_hd_feature_transformer() {
    using namespace ov;
    using namespace element;
    using namespace opset13;
    using namespace std;
    auto t0 = make_shared<Parameter>(f32, PartialShape{-1, 576, 1024});
    auto t1 = make_shared<Parameter>(i32, PartialShape{});
    auto t2 = make_shared<Parameter>(i32, PartialShape{});
    auto t3 = make_shared<ShapeOf>(t0);
    auto t4 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t5 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t6 = make_shared<Gather>(t3, t4, t5);
    auto t7 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{1});
    auto t8 = make_shared<Reshape>(t6, t7, false);
    auto t9 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{1});
    auto t10 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t11 = make_shared<Gather>(t3, t9, t10);
    auto t12 = make_shared<Convert>(t11, element::f32);
    auto t13 = make_shared<Constant>(f32, Shape{}, vector<float>{0.5});
    auto t14 = make_shared<Power>(t12, t13, "numpy");
    auto t15 = make_shared<Convert>(t14, element::i32);
    auto t16 = make_shared<Convert>(t15, element::i64);
    auto t17 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t18 = make_shared<Unsqueeze>(t16, t17);
    auto t19 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t20 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t21 = make_shared<Gather>(t3, t19, t20);
    auto t22 = make_shared<Concat>(NodeVector{t8, t18, t18, t21}, 0);
    auto t23 = make_shared<Reshape>(t0, t22, false);
    auto t24 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t25 = make_shared<Divide>(t16, t24, "numpy");
    auto t26 = make_shared<Floor>(t25);
    auto t27 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t28 = make_shared<Unsqueeze>(t26, t27);
    auto t29 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t30 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{2});
    auto t31 = make_shared<Concat>(NodeVector{t8, t28, t29, t28, t30, t21}, 0);
    auto t32 = make_shared<Reshape>(t23, t31, false);
    auto t33 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 1, 3, 2, 4, 5});
    auto t34 = make_shared<Transpose>(t32, t33);
    auto t35 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{-1});
    auto t36 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{4});
    auto t37 = make_shared<Multiply>(t21, t36, "numpy");
    auto t38 = make_shared<Concat>(NodeVector{t8, t35, t37}, 0);
    auto t39 = make_shared<Reshape>(t34, t38, false);
    auto t40 = make_shared<Multiply>(t1, t2, "numpy");
    auto t41 = make_shared<Convert>(t40, element::i64);
    auto t42 = make_shared<Divide>(t6, t41, "numpy");
    auto t43 = make_shared<Floor>(t42);
    auto t44 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{0});
    auto t45 = make_shared<Unsqueeze>(t43, t44);
    auto t46 = make_shared<Convert>(t1, element::i64);
    auto t47 = make_shared<Unsqueeze>(t46, t44);
    auto t48 = make_shared<Convert>(t2, element::i64);
    auto t49 = make_shared<Unsqueeze>(t48, t44);
    auto t50 = make_shared<Constant>(i64, Shape{1}, vector<int64_t>{-1});
    auto t51 = make_shared<Concat>(NodeVector{t45, t47, t49, t28, t28, t50}, 0);
    auto t52 = make_shared<Reshape>(t39, t51, false);
    auto t53 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 1, 3, 2, 4, 5});
    auto t54 = make_shared<Transpose>(t52, t53);
    auto t55 = make_shared<Multiply>(t1, t15, "numpy");
    auto t56 = make_shared<Convert>(t55, element::i64);
    auto t57 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t58 = make_shared<Divide>(t56, t57, "numpy");
    auto t59 = make_shared<Floor>(t58);
    auto t60 = make_shared<Constant>(i32, Shape{}, vector<int32_t>{0});
    auto t61 = make_shared<Unsqueeze>(t59, t60);
    auto t62 = make_shared<Multiply>(t2, t15, "numpy");
    auto t63 = make_shared<Convert>(t62, element::i64);
    auto t64 = make_shared<Constant>(i64, Shape{}, vector<int64_t>{2});
    auto t65 = make_shared<Divide>(t63, t64, "numpy");
    auto t66 = make_shared<Floor>(t65);
    auto t67 = make_shared<Unsqueeze>(t66, t60);
    auto t68 = make_shared<Concat>(NodeVector{t45, t61, t67, t37}, 0);
    auto t69 = make_shared<Reshape>(t54, t68, false);
    shared_ptr<Model> model = make_shared<Model>(make_shared<Result>(t69), ParameterVector{t0, t1, t2});
    return utils::singleton_core().compile_model(
        model, "CPU"
    );
}

ov::Tensor reshape_hd_patches_2x2merge(const ov::Tensor& image_features, size_t h_crop, size_t w_crop, InferRequest& hd_feature_transformer) {
    ov::Shape shape = image_features.get_shape();
    OPENVINO_ASSERT(3 == shape.size());
    OPENVINO_ASSERT(24 * 24 == shape.at(1));
    OPENVINO_ASSERT(1024 == shape.at(2));
    hd_feature_transformer.set_input_tensor(0, image_features);
    ov::Tensor height{ov::element::i32, {}, &h_crop};
    hd_feature_transformer.set_input_tensor(1, height);
    ov::Tensor width{ov::element::i32, {}, &w_crop};
    hd_feature_transformer.set_input_tensor(2, width);
    hd_feature_transformer.infer();
    return hd_feature_transformer.get_output_tensor();
}

// image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
// output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
ov::Tensor add_image_newline(const ov::Tensor& image_features_hd, const std::vector<float>& sub_GN) {
    const ov::Shape& nhwc = image_features_hd.get_shape();  // [N, 12*h_crop, 12*w_crop, 4096]
    const float* in = image_features_hd.data<float>();
    ov::Tensor image_features_hd_new_line{ov::element::f32, {nhwc.at(0), nhwc.at(1) * (nhwc.at(2) + 1), nhwc.at(3)}};
    float* out = image_features_hd_new_line.data<float>();
    for (size_t batch_id = 0; batch_id < nhwc.at(0); ++batch_id) {
        for (size_t row_id = 0; row_id < nhwc.at(1); ++row_id) {
            for (size_t col_id = 0; col_id < nhwc.at(2); ++col_id) {
                std::copy_n(
                    in + batch_id * nhwc.at(1) * nhwc.at(2) * nhwc.at(3) + row_id * nhwc.at(2) * nhwc.at(3) + col_id * nhwc.at(3),
                    nhwc.at(3),
                    out + batch_id * nhwc.at(1) * (nhwc.at(2) + 1) * nhwc.at(3) + row_id * (nhwc.at(2) + 1) * nhwc.at(3) + col_id * nhwc.at(3)
                );
            }
            std::copy(
                sub_GN.begin(),
                sub_GN.end(),
                out + batch_id * nhwc.at(1) * (nhwc.at(2) + 1) * nhwc.at(3) + row_id * (nhwc.at(2) + 1) * nhwc.at(3) + nhwc.at(2) * nhwc.at(3)
            );
        }
    }
    return image_features_hd_new_line;
}

ov::Tensor concatenate_2d(const ov::Tensor& first_1lf, const std::vector<float>& second_f, const ov::Tensor& third_1lf) {
    size_t first_l = first_1lf.get_shape().at(1);
    constexpr size_t second_l = 1;
    size_t third_l = third_1lf.get_shape().at(1);
    size_t features = first_1lf.get_shape().at(2);
    OPENVINO_ASSERT(second_f.size() == features);
    ov::Tensor out_1lf{ov::element::f32, {1, first_l + second_l + third_l, features}};
    float* out = out_1lf.data<float>();
    std::copy_n(first_1lf.data<float>(), first_l * features, out);
    std::copy(second_f.begin(), second_f.end(), out + first_l * features);
    std::copy_n(third_1lf.data<float>(), third_l * features, out + (first_l + second_l) * features);
    return out_1lf;
}

// image_features.resized_source: (num_crops+1, 24*24, 1024)
ov::Tensor hd_feature_transform(const EncodedImage& image_features, InferRequest& hd_feature_transformer, const std::vector<float>& sub_GN, const std::vector<float>& glb_GN, ov::InferRequest& vision_projection) {
    const ov::Shape& image_features_shape = image_features.resized_source.get_shape();
    ov::Tensor global_image_features{ov::element::f32, {1, image_features_shape.at(1), image_features_shape.at(2)}, image_features.resized_source.data<float>()};
    // global feature can be viewed as a special HD case with num_crops 1x1
    ov::Tensor global_image_features_hd = reshape_hd_patches_2x2merge(global_image_features, 1, 1, hd_feature_transformer);
    ov::Tensor global_image_features_hd_newline = add_image_newline(global_image_features_hd, sub_GN);  // [1,12*(12+1),4096]
    constexpr size_t INPUT_IMAGE_SIZE = 336;
    size_t h_crop = image_features.resized_source_size.height / INPUT_IMAGE_SIZE;
    size_t w_crop = image_features.resized_source_size.width / INPUT_IMAGE_SIZE;
    size_t num_crops = h_crop * w_crop;

    // NOTE: real num_crops is padded
    // (num_crops, 24*24, 1024)
    ov::Tensor sub_image_features{ov::element::f32, {
        num_crops,
        image_features_shape.at(1),
        image_features_shape.at(2)
    }, image_features.resized_source.data<float>() + image_features_shape.at(1) * image_features_shape.at(2)};
    ov::Tensor sub_image_features_hd = reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop, hd_feature_transformer);  // [1, 24, 24, 4096]
    ov::Tensor sub_image_features_hd_newline = add_image_newline(sub_image_features_hd, sub_GN);  // [1,h_crop*12*(w_crop*12+1), 4096]
    ov::Tensor image_embeddings = concatenate_2d(sub_image_features_hd_newline, glb_GN, global_image_features_hd_newline);  // [1,l,4096]
    vision_projection.set_input_tensor(image_embeddings);
    vision_projection.infer();
    ov::Tensor out = vision_projection.get_output_tensor();
    ov::Tensor res{out.get_element_type(), out.get_shape()};
    out.copy_to(res);
    return res;
}

std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input, std::shared_ptr<ov::Node> target_size) {
    using namespace ov::op;

    // Convert to float32 before interpolation (required for bicubic)
    auto input_f32 = std::make_shared<v0::Convert>(input, ov::element::f32);

    // For NHWC format, resize axes are [1, 2] (height, width dimensions)
    auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 2});

    v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.cube_coeff = -0.5f;  // Standard coefficient for bicubic interpolation (Catmull-Rom)
    attrs.nearest_mode = v11::Interpolate::NearestMode::FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;

    return std::make_shared<v11::Interpolate>(input_f32, target_size, axes, attrs);
}

std::shared_ptr<ov::Node> create_mean_scale(std::shared_ptr<ov::Node> input_u8_or_f32, const ProcessorConfig& config) {
    using namespace ov::op;

    std::shared_ptr<ov::Node> input_f32;

    // Convert to float32 if input is uint8, otherwise use as-is
    if (input_u8_or_f32->get_element_type() == ov::element::u8) {
        input_f32 = std::make_shared<v0::Convert>(input_u8_or_f32, ov::element::f32);
    } else {
        input_f32 = std::move(input_u8_or_f32);
    }

    // Follow the original mean_scale() function logic exactly:
    // (float(uint8_data[idx]) / 255.0f - config.image_mean[c]) / config.image_std[c]
    // Step 1: x / 255.0
    auto scale_255 = v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{255.0f});
    auto divided_by_255 = std::make_shared<v1::Divide>(input_f32, scale_255);

    // Step 2: Create mean and std constants [R, G, B] - broadcasted along channel dimension
    // For NHWC format, we need shape [1, 1, 1, 3] to broadcast correctly
    auto mean_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 3},
        std::vector<float>{config.image_mean[0], config.image_mean[1], config.image_mean[2]});
    auto std_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 3},
        std::vector<float>{config.image_std[0], config.image_std[1], config.image_std[2]});

    // Step 3: (x/255.0 - mean)
    auto mean_subtracted = std::make_shared<v1::Subtract>(divided_by_255, mean_const);

    // Step 4: (x/255.0 - mean) / std
    auto result = std::make_shared<v1::Divide>(mean_subtracted, std_const);

    return result;
}

std::shared_ptr<ov::Node> create_channels_first(std::shared_ptr<ov::Node> input_nhwc) {
    using namespace ov::op;

    // Transpose from NHWC (0,1,2,3) to NCHW (0,3,1,2)
    auto transpose_order = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
    return std::make_shared<v1::Transpose>(input_nhwc, transpose_order);
}

std::shared_ptr<ov::Node> create_slice_image(std::shared_ptr<ov::Node> input_nchw) {
    using namespace ov::op;

    // Input: (N, C, H, W) -> Output: (N*num_h_slices*num_w_slices, C, 336, 336)
    auto shape_node = std::make_shared<v3::ShapeOf>(input_nchw);
    // Index constants for gathering shape dimensions
    auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}); // N
    auto axis_1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}); // C
    auto axis_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}); // H
    auto axis_3 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}); // W
    auto axis_0_node = v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}); // Gather axis

    auto N = std::make_shared<v8::Gather>(shape_node, axis_0, axis_0_node);
    auto C = std::make_shared<v8::Gather>(shape_node, axis_1, axis_0_node);
    auto H = std::make_shared<v8::Gather>(shape_node, axis_2, axis_0_node);
    auto W = std::make_shared<v8::Gather>(shape_node, axis_3, axis_0_node);

    // Patch size constant (336)
    auto patch_size = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{INPUT_IMAGE_SIZE});

    // Calculate number of slices (num_h = H / patch_size, num_w = W / patch_size)
    auto num_h = std::make_shared<v1::Divide>(H, patch_size);
    auto num_w = std::make_shared<v1::Divide>(W, patch_size);

    // Reshape to 6D [N, C, num_h, patch_size, num_w, patch_size]
    auto target_shape_6d = std::make_shared<v0::Concat>(ov::NodeVector{N, C, num_h, patch_size, num_w, patch_size}, 0);
    auto reshape_6d = std::make_shared<v1::Reshape>(input_nchw, target_shape_6d, false);

    // Transpose (Permute)
    // Current: 0:N, 1:C, 2:num_h, 3:S, 4:num_w, 5:S
    // Target:  0:N, 2:num_h, 4:num_w, 1:C, 3:S, 5:S
    auto permute_order = v0::Constant::create(ov::element::i64, ov::Shape{6}, std::vector<int64_t>{0, 2, 4, 1, 3, 5});
    auto permuted = std::make_shared<v1::Transpose>(reshape_6d, permute_order);

    // Flatten to 4D [N * num_h * num_w, C, S, S]
    auto minus_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto target_shape_4d = std::make_shared<v0::Concat>(ov::NodeVector{minus_one, C, patch_size, patch_size}, 0);
    auto final_reshape = std::make_shared<v1::Reshape>(permuted, target_shape_4d, false);

    return final_reshape;
}

std::shared_ptr<ov::Node> create_concatenate_batch(std::shared_ptr<ov::Node> global_processed, std::shared_ptr<ov::Node> hd_sliced) {
    using namespace ov::op;

    // Concatenate along batch dimension (axis 0)
    // global_processed: (1, C, H, W)
    // hd_sliced: (num_slices, C, H, W)
    // Output: (1 + num_slices, C, H, W)
    return std::make_shared<v0::Concat>(ov::NodeVector{std::move(global_processed), std::move(hd_sliced)}, 0);
}

std::shared_ptr<ov::Node> create_pad_to_max_crops(std::shared_ptr<ov::Node> input_nchw, std::shared_ptr<ov::Node> max_crops_param) {
    using namespace ov::op;

    // Get current input batch size (num_crops)
    auto shape_of = std::make_shared<v3::ShapeOf>(input_nchw);
    auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto axis_0_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}); // Axis for Gather
    auto num_crops = std::make_shared<v8::Gather>(shape_of, axis_0, axis_0_scalar);

    // Calculate required padding amount: padding_needed = max(0, max_crops - num_crops)
    auto diff = std::make_shared<v1::Subtract>(max_crops_param, num_crops);
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto padding_needed = std::make_shared<v1::Maximum>(diff, zero);

    // Configure Pad operation arguments (pads_end)
    auto zero_3 = v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 0}); // Zeros for C, H, W dimensions
    auto zero_4 = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0}); // pads_begin
    auto pads_end = std::make_shared<v0::Concat>(ov::OutputVector{padding_needed, zero_3}, 0);

    // Execute Pad operation (Constant mode, fill with 0)
    auto pad_value = v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto padded = std::make_shared<v1::Pad>(
        input_nchw,
        zero_4,      // pads_begin
        pads_end,    // pads_end
        pad_value,   // pad_value
        ov::op::PadMode::CONSTANT
    );

    return padded;
}

std::shared_ptr<ov::Model> patch_image_preprocess_into_vision_encoder_model(
    const std::shared_ptr<ov::Model>& vision_encoder_model,
    const ProcessorConfig& config) {

    using namespace ov;
    using namespace ov::op;

    // Input: HD transformed image in NHWC format (uint8)
    // Shape: {1, -1, -1, 3} => {batch=1, height=dynamic, width=dynamic, channels=3 (RGB)}
    auto hd_image = std::make_shared<v0::Parameter>(element::u8, PartialShape{1, -1, -1, 3});
    // Target size for global image resize [height, width]
    auto global_target_size = std::make_shared<v0::Parameter>(element::i64, PartialShape{2});
    // Max crops parameter for dynamic padding
    auto max_crops = std::make_shared<v0::Parameter>(element::i64, PartialShape{});

    // Process global image (resize + normalize + channels_first)
    auto global_resized = create_bicubic_resize(hd_image, global_target_size);
    auto global_normalized = create_mean_scale(std::move(global_resized), config);
    auto global_processed = create_channels_first(std::move(global_normalized));

    // Process HD image (normalize + channels_first + slice)
    auto hd_normalized = create_mean_scale(hd_image, config);
    auto hd_processed = create_channels_first(std::move(hd_normalized));
    auto hd_sliced = create_slice_image(std::move(hd_processed));

    // Concatenate global and HD results
    auto concatenated = create_concatenate_batch(std::move(global_processed), std::move(hd_sliced));

    // Pad to max crops
    auto padded_result = create_pad_to_max_crops(std::move(concatenated), max_crops);

    auto vision_params = vision_encoder_model->get_parameters();
    auto vision_results = vision_encoder_model->get_results();

    vision_params[0]->output(0).replace(padded_result);

    return std::make_shared<Model>(
        std::move(vision_results),
        ParameterVector{std::move(hd_image), std::move(global_target_size), std::move(max_crops)}
    );
}

} // namespace

namespace phi_utils {
std::string normalize_prompt(
    const std::string& prompt, size_t base_id, size_t n_images, const std::regex& native_pattern, void(*write_native)(std::ostream& os, size_t idx)
) {
    std::smatch match;
    std::regex_search(prompt, match, native_pattern);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, write_native);
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(match.empty(), "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return image_prompt;
    }
    // Restore ids from native tags
    if (!match.empty()) {
        size_t image_id = std::stoul(match.str(1));
        OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
        image_sequence.push_back(image_id - 1);
        constexpr int submatch_id_to_return = 1;
        for (std::sregex_token_iterator iter{
            match.suffix().first,
            prompt.end(),
            native_pattern,
            submatch_id_to_return
        }; iter != std::sregex_token_iterator{}; ++iter) {
            size_t image_id = std::stoul(*iter);
            OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
            image_sequence.push_back(image_id - 1);
        }
        if (!image_sequence.empty()) {
            verify_ids(image_sequence, base_id, n_images);
            return image_prompt;
        }
    }
    // Prepend native tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        write_native(stream, image_sequence.back());
    }
    stream << prompt;
    return stream.str();
}

/// @brief ov::Tensor is tokenized text, size_t is image tag
std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer, const std::regex& native_pattern) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{
        prefix_begin,
        text.end(),
        native_pattern,
        {0, 1}  // Every match emits two values: whole match and submatch
    }; iter != std::sregex_token_iterator{}; ++iter) {
        if (is_submatch) {
            size_t idx = std::stoul(iter->str());
            OPENVINO_ASSERT(idx != 0);
            tokenized.push_back(idx - 1);
        } else {
            std::string regular_text{prefix_begin, iter->first};
            if (!regular_text.empty()) {
                tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
            }
            prefix_begin = iter->second;
        }
        is_submatch = !is_submatch;
    }
    std::string regular_text{prefix_begin, text.end()};
    if (!regular_text.empty()) {
        tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
    }
    return tokenized;
}

ov::Tensor insert_image_placeholders(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    const std::vector<size_t>& tokens_per_images
) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(utils::overloaded{
            [](const ov::Tensor& chunk) {
                return chunk.get_shape().at(1);
            },
            [&](size_t image_id) {
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                size_t length = chunk.get_shape().at(1);
                std::copy_n(
                    chunk.data<int64_t>(),
                    length,
                    merged.data<int64_t>() + offset
                );
                return length;
            },
            [&](size_t image_id) {
                int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                std::fill_n(
                    merged.data<int64_t>() + offset,
                    tokens_per_images.at(image_id),
                    fill_value  // -1 to distinguish 0 token and 0 image id.
                );
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    int64_t last_token = tokens.data<int64_t>()[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < tokens.get_shape().at(1); ++offset) {
        // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
        // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
        // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
        // If last_token and next_token are negative, it's continuation of an image placeholder - skip
        // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
        int64_t next_token = tokens.data<int64_t>()[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
            chunks.emplace_back(
                std::in_place_type<ov::Tensor>,
                ov::element::i64,
                ov::Shape{1, offset - text_start},
                const_cast<int64_t*>(tokens.data<int64_t>()) + text_start
            );
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }
    // Add the last chunk
    size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
        chunks.emplace_back(
            std::in_place_type<ov::Tensor>,
            ov::element::i64,
            ov::Shape{1, full_length - text_start},
            const_cast<int64_t*>(tokens.data<int64_t>()) + text_start
        );
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

}  // namespace phi_utils

EncodedImage VisionEncoderPhi3V::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ImageSize image_size;

    if (use_ov_vision_preprocess) {
        ov::Tensor hd_image = HD_transform(image, config.phi3_v.num_crops);
        image_size = ImageSize{hd_image.get_shape().at(2), hd_image.get_shape().at(1)};

        int64_t global_size[2] = {INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE};
        ov::Tensor global_target_size(ov::element::i64, ov::Shape{2}, global_size);

        int64_t max_crops_value = static_cast<int64_t>(config.phi3_v.num_crops);
        ov::Tensor max_crops_tensor(ov::element::i64, ov::Shape{}, &max_crops_value);

        encoder.set_input_tensor(0, hd_image);
        encoder.set_input_tensor(1, global_target_size);
        encoder.set_input_tensor(2, max_crops_tensor);
    } else {
        const auto& [pixel_values, is] = get_pixel_values_phi3_v(image, config);
        image_size = is;
        encoder.set_input_tensor(pixel_values);
    }

    ov::Tensor res{ov::element::f32, encoder.get_output_tensor().get_shape()};
    encoder.set_output_tensor(res);
    encoder.infer();

    EncodedImage encoded_image = {std::move(res), image_size};

    CircularBufferQueueElementGuard<ov::InferRequest> hd_feature_transformer_ireq_guard(this->m_ireq_queue_hd_feature_transformer.get());
    CircularBufferQueueElementGuard<ov::InferRequest> vision_projection_ireq_guard(this->m_ireq_queue_vision_projection.get());
    ov::InferRequest& hd_feature_transformer = hd_feature_transformer_ireq_guard.get();
    ov::InferRequest& vision_projection = vision_projection_ireq_guard.get();
    encoded_image.images_features_projection = hd_feature_transform(encoded_image, hd_feature_transformer, m_vlm_config.sub_GN, m_vlm_config.glb_GN, vision_projection);
    return encoded_image;
}

bool can_use_ov_vision_preprocess() {
    const char* env = std::getenv("VISION_PREPROCESS");
    return !(env && std::string(env) == "CPP");
}

VisionEncoderPhi3V::VisionEncoderPhi3V(const std::filesystem::path& model_dir,
                                       const std::string& device,
                                       const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties),
      use_ov_vision_preprocess(can_use_ov_vision_preprocess()) {
    if (use_ov_vision_preprocess) {
        auto vision_encoder_model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
        auto model = patch_image_preprocess_into_vision_encoder_model(vision_encoder_model, m_processor_config);
        auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
        m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled_model.get_property(ov::optimal_number_of_infer_requests),
            [&compiled_model]() -> ov::InferRequest {
                return compiled_model.create_infer_request();
            });
    }

    auto compiled_model = create_hd_feature_transformer();
    m_ireq_queue_hd_feature_transformer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {});
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderPhi3V::VisionEncoderPhi3V(const ModelsMap& models_map,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties),
      use_ov_vision_preprocess(can_use_ov_vision_preprocess()) {
    if (use_ov_vision_preprocess) {
        const auto& [vision_encoder_model, vision_encoder_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto model_org = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
        auto model = patch_image_preprocess_into_vision_encoder_model(model_org, m_processor_config);
        auto compiled_model = utils::singleton_core().compile_model(model, device, properties);

        m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled_model.get_property(ov::optimal_number_of_infer_requests),
            [&compiled_model]() -> ov::InferRequest {
                return compiled_model.create_infer_request();
            });
    }

    auto compiled_model = create_hd_feature_transformer();
    m_ireq_queue_hd_feature_transformer = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_projection").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_projection").second;
    compiled_model = utils::singleton_core().compile_model(vision_encoder_model, vision_encoder_weights, device, properties);
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

InputsEmbedderPhi3V::InputsEmbedderPhi3V(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}


InputsEmbedderPhi3V::InputsEmbedderPhi3V(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

NormalizedPrompt InputsEmbedderPhi3V::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {phi_utils::normalize_prompt(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
}

ov::Tensor InputsEmbedderPhi3V::get_inputs_embeds(const std::string& image_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    size_t base_id = m_tokens_per_images.size();
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }
    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = phi_utils::split_tokenize(image_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", image_prompt}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = image_prompt;
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = phi_utils::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    ov::Tensor new_merged_tokens = phi_utils::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = phi_utils::drop_image_placeholders(new_tokens);
    ov::Tensor inputs_embeds{ov::element::f32, {1, new_tokens.get_shape().at(1), m_vlm_config.hidden_size}};
    size_t offset = 0;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    for (const std::variant<ov::Tensor, size_t>& chunk : tokens) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                const ov::Tensor& text_embeds = m_embedding->infer(req, chunk);
                size_t text_length = text_embeds.get_shape().at(1);
                std::copy_n(
                    text_embeds.data<float>(),
                    text_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return text_length;
            },
            [&](size_t image_id) {
                const ov::Tensor& image_embeds = images_features_proj.at(image_id - base_id);
                size_t im_length = image_embeds.get_shape().at(1);
                std::copy_n(
                    image_embeds.data<float>(),
                    image_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return im_length;
            }
        }, chunk);
    }

    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }
    return inputs_embeds;
}

void InputsEmbedderPhi3V::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL)
        m_tokens_per_images = m_prev_tokens_per_images;
    else
        m_prev_tokens_per_images = m_tokens_per_images;
}

void InputsEmbedderPhi3V::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
}

void InputsEmbedderPhi3V::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
}

} // namespace ov::genai
