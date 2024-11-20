// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset15.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

class IImageProcessor {
public:
    virtual ~IImageProcessor() = default;

    ov::Tensor preprocess(ov::Tensor image) override {
        m_request.set_input_tensor(mask_image);
        m_request.infer();
        return m_request.get_output_tensor();
    }

protected:
    std::shared_ptr<ov::Model create_empty_model() {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto result = std::make_shared<ov::op::v0::Result>(parameter);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    }

    void compile(ov::shared_ptr<ov::Model> model, const std::string& device) {
        // TODO: support GPU as well
        m_request = utils::singleton_core().compile_model(model, "CPU").create_infer_request();
    }

    ov::InferRequest m_request;
};

class ImageProcessor : public IImageProcessor {
public:
    explicit ImageProcessor(bool do_normalize = true, bool do_binarize = false) {
        auto image_processor_model = create_empty_model();
        merge_image_preprocessing(image_processor_model, do_normalize, do_binarize);

        // TODO: support GPU as well
        compile(image_processor_model, "CPU");
    }

    static void merge_image_preprocessing(ov::shared_ptr<ov::Model> model, bool do_normalize = true, bool do_binarize = false) {
        OPENVINO_ASSERT(do_normalize ^ do_binarize, "Both binarize and normalize are not supported");

        // https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L90-L110
        ov::preprocess::PrePostProcessor ppp(model);

        ppp.input().tensor()
            .set_layout("NHWC")
            .set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().model()
            .set_layout("NCHW");

        if (do_normalize) {
            ppp.input().tensor().set_layout("NHWC");
            ppp.input().model().set_layout("NCHW");

            ppp.input().tensor()
                .set_element_type(ov::element::u8);

            ppp.input().preprocess()
                .convert_layout()
                .convert_element_type(ov::element::f32)
                // this is less accurate that in VaeImageProcessor::normalize
                .scale(255.0 / 2.0)
                .mean(1.0f);
        } else if (do_binarize) {
            ppp.input().preprocess()
                .convert_element_type(ov::element::f32)
                .convert_color(ov::preprocess::ColorFormat::GRAY)
                .scale(255.0f)
                .custom([](const ov::Output<ov::Node>& port) {
                    auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
                    auto constant_1_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 1.0f);
                    auto constant_0_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.0f);
                    auto mask_bool = std::make_shared<ov::opset15::GreaterEqual>(port, constant_0_5);
                    auto mask_float = std::make_shared<ov::opset15::Select>(mask_bool, constant_1_0, constant_0_0);
                    return mask_float;
                });
        }

        ppp.build();
    }

private:
    std::shared_ptr<ov::Model create_empty_model() {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto result = std::make_shared<ov::op::v0::Result>(parameter);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    }

    ov::InferRequest m_request;
};

class ImageResizer : public IImageProcessor {
public:
    ImageResizer(const size_t dst_height, const size_t dst_width) {
        auto resize_model = create_empty_model();
        ov::preprocess::PrePostProcessor ppp(resize_model);

        size_t dst_height = mask_image.get_shape()[1] / m_vae->get_vae_scale_factor();
        size_t dst_width = mask_image.get_shape()[2] / m_vae->get_vae_scale_factor();

        ppp.input().tensor().set_layout("NCHW");
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST, dst_height, dst_width);

        resize_model = ppp.build();

        // TODO: support GPU as well
        compile(resize_model, "CPU");
    }
}

} // namespace genai
} // namespace ov
