#pragma once
#include <napi.h>

#include "openvino/core/type/element_type.hpp"
#include "openvino/openvino.hpp"

ov::AnyMap to_anyMap(const Napi::Env&, const Napi::Value&);

/**
 * @brief  Template function to convert Javascript data types into C++ data types
 * @tparam TargetType destinated C++ data type
 * @param info Napi::CallbackInfo contains all arguments passed to a function or method
 * @param idx specifies index of a argument inside info.
 * @return specified argument converted to a TargetType.
 */
template <typename TargetType>
TargetType js_to_cpp(const Napi::Env& env, const Napi::Value& value);

/** @brief  A template specialization for TargetType ov::Any */
template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value);

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num);
