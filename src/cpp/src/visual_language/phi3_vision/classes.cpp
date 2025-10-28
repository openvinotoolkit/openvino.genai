
// Copyright (C) 2023-2025 Intel Corporation
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
    ov::Shape shape = _1hw3.get_shape();
    ov::Tensor _13hw = ov::Tensor{ov::element::f32, {1, 3, shape.at(1), shape.at(2)}};
    auto _1hw3_data = _1hw3.data<float>();
    float* _13hw_data = _13hw.data<float>();
    for (size_t plane = 0; plane < 3; ++plane) {
        for (size_t row = 0; row < shape.at(1); ++row) {
            for (size_t col = 0; col < shape.at(2); ++col) {
                _13hw_data[plane * shape.at(1) * shape.at(2) + row * shape.at(2) + col] = _1hw3_data[row * shape.at(2) * 3 + col * 3 + plane];
            }
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
    bicubic_resize(img, dst, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
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
            chunks.emplace_back(
                std::in_place_type<ov::Tensor>,
                ov::element::i64,
                ov::Shape{1, offset - text_start},
                tokens.data<int64_t>() + text_start
            );
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }
    // Add the last chunk
    size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        chunks.emplace_back(
            std::in_place_type<ov::Tensor>,
            ov::element::i64,
            ov::Shape{1, full_length - text_start},
            tokens.data<int64_t>() + text_start
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

    const auto& [pixel_values, image_size] = get_pixel_values_phi3_v(image, config);
    encoder.set_input_tensor(pixel_values);
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

VisionEncoderPhi3V::VisionEncoderPhi3V(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(model_dir, device, properties) {
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

VisionEncoderPhi3V::VisionEncoderPhi3V(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(models_map, config_dir_path, device, properties) {
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

NormlizedPrompt InputsEmbedderPhi3V::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
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
