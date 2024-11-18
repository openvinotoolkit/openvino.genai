// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace genai {

/** 
 * @brief This pass adds UTF8Validate with replace mode operation to the detokenization output.
 * 
 * Before:
 *               \          |        /
 *                \         |       /
 *                 v        v      v
 *              +------------------------+
 *              |    StringTensorPack    |
 *              +------------------------+
 *                       |  |  |
 *                       v  v  v
 *
 * After:
 * 
 *               \          |        /
 *                \         |       /
 *                 v        v      v
 *                +------------------+
 *                |   UTF8Validate   |
 *                +------------------+
 *                       |  |  |
 *                       v  v  v
 *              +------------------------+
 *              |    StringTensorPack    |
 *              +------------------------+
 *                       |  |  |
 *                       v  v  v
 * 
**/
class AddUTF8Validate : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("AddUTF8Validate", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};


} // namespace genai
} // namespace ov
