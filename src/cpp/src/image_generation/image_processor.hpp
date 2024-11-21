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
    explicit IImageProcessor(const std::string& device) :
        m_device(device) {
    }

    virtual ~IImageProcessor() = default;

    virtual ov::Tensor execute(ov::Tensor image) {
        m_request.set_input_tensor(image);
        m_request.infer();
        return m_request.get_output_tensor();
    }

protected:
    std::shared_ptr<ov::Model> create_empty_model() {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto result = std::make_shared<ov::op::v0::Result>(parameter);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    }

    void compile(std::shared_ptr<ov::Model> model) {
        // TODO: support GPU as well
        m_request = utils::singleton_core().compile_model(model, m_device).create_infer_request();
    }

    ov::InferRequest m_request;
    std::string m_device;
};

class ImageProcessor : public IImageProcessor {
public:
    explicit ImageProcessor(bool do_normalize = true, bool do_binarize = false) :
        IImageProcessor("CPU") {
        auto image_processor_model = create_empty_model();
        merge_image_preprocessing(image_processor_model, do_normalize, do_binarize);

        // TODO: support GPU as well
        compile(image_processor_model);
    }

    static void merge_image_preprocessing(std::shared_ptr<ov::Model> model, bool do_normalize = true, bool do_binarize = false) {
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
};

class ImageResizer : public IImageProcessor {
public:
    ImageResizer(const size_t dst_height, const size_t dst_width) :
        IImageProcessor("CPU") {
        auto resize_model = create_empty_model();
        ov::preprocess::PrePostProcessor ppp(resize_model);

        ppp.input().tensor().set_layout("NCHW");
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST, dst_height, dst_width);

        resize_model = ppp.build();

        compile(resize_model);
    }
};

} // namespace genai
} // namespace ov
