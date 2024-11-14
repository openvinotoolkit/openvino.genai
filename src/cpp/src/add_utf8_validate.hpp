// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace genai {

/** 
 * @brief This pass modifies tokenizer ov::Model so that special tokens adding will be
 *  enabled or disabled depending on stateful value.
 * 
 *  +--------------+
 *  |  DefaultMode |
 *  +--------------+
 *         |
 *         |
 *         v
 *  +--------------+  +--------+  +------------------+
 *  |  ReadValue   |  |  ends  |  | const value = 0  |
 *  +--------------+  +--------+  +------------------+
 *             \          |        /
 *              \         |       /
 *               v        v      v
 *                +--------------+
 *                |    Select    |
 *                +--------------+
 *                       |
 *                       v
 *          +-------------------------+
 *          |     CombineSegments     |
 *          +-------------------------+
**/
class AddUTF8Validate : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("AddUTF8Validate", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};


} // namespace genai
} // namespace ov
