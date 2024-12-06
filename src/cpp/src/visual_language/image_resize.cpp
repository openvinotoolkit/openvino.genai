#include "image_resize.hpp"

static ov::CompiledModel resize_model = create_resize_model();

ov::CompiledModel create_resize_model() {
    using namespace ov::op;

    ov::PartialShape image_shape = ov::PartialShape::dynamic(4);
    
    auto input = std::make_shared<v0::Parameter>(ov::element::u8, image_shape);
    auto converted = std::make_shared<v0::Convert>(input, ov::element::f32);

    auto sizes_param = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{2});
    
    std::vector<int32_t> axes = {1, 2};  // H, W dimensions in NHWC format
    
    auto axes_constant = std::make_shared<v0::Constant>(
        ov::element::i32,
        ov::Shape{2},
        axes
    );
    
    v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::BICUBIC_PILLOW;
    attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::util::InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.nearest_mode = ov::op::util::InterpolateBase::NearestMode::FLOOR;
    attrs.antialias = true;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    
    auto interpolated = std::make_shared<v11::Interpolate>(
        converted,
        sizes_param,
        axes_constant,
        attrs
    );
    
    auto clamped = std::make_shared<v0::Clamp>(interpolated, 0.0f, 255.0f);
    auto rounded = std::make_shared<v5::Round>(clamped, v5::Round::RoundMode::HALF_TO_EVEN);
    auto result = std::make_shared<v0::Convert>(rounded, ov::element::u8);

    auto model = std::make_shared<ov::Model>(
        result,
        ov::ParameterVector{input, sizes_param},
        "image_resizer"
    );

    return ov::genai::utils::singleton_core().compile_model(model, "CPU");
}

ov::Tensor resize_image(const ov::Tensor& image, int64_t height, int64_t width) {
    auto infer_request = resize_model.create_infer_request();
    
    std::vector<int32_t> sizes = {static_cast<int32_t>(height), static_cast<int32_t>(width)};
    ov::Tensor sizes_tensor(ov::element::i32, ov::Shape{2}, sizes.data());
    
    infer_request.set_input_tensor(0, image);
    infer_request.set_input_tensor(1, sizes_tensor);
    infer_request.infer();
    
    return infer_request.get_output_tensor();
}
