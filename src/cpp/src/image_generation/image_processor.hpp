// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/runtime/infer_request.hpp"

#include "openvino/op/interpolate.hpp"

namespace ov {
namespace genai {

class IImageProcessor {
public:
    explicit IImageProcessor(const std::string& device);

    virtual ~IImageProcessor() = default;

    virtual ov::Tensor execute(ov::Tensor image);

protected:
    void compile(std::shared_ptr<ov::Model> model);

    ov::InferRequest m_request;
    std::string m_device;
};

class ImageProcessor : public IImageProcessor {
public:
    explicit ImageProcessor(const std::string& device, bool do_normalize = true, bool do_binarize = false, bool gray_scale_source = false);

    static void merge_image_preprocessing(std::shared_ptr<ov::Model> model, bool do_normalize = true, bool do_binarize = false, bool gray_scale_source = false);
};

class ImageResizer {
public:
    ImageResizer(const std::string& device, ov::element::Type type, ov::Layout layout, ov::op::v11::Interpolate::InterpolateMode interpolation_mode);

    ov::Tensor execute(ov::Tensor image, int64_t dst_height, int64_t dst_width);

private:
    size_t get_and_check_width_idx(const Layout& layout, const PartialShape& shape);
    size_t get_and_check_height_idx(const Layout& layout, const PartialShape& shape);

    ov::InferRequest m_request;
};

} // namespace genai
} // namespace ov
