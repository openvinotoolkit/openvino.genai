#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include <openvino/op/interpolate.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/round.hpp>

#include "utils.hpp"

ov::CompiledModel create_resize_model();

ov::Tensor resize_image(const ov::Tensor& image, int64_t height, int64_t width);
