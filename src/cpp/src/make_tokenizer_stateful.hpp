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
class MakeCombineSegmentsSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MakeCombineSegmentsSatateful", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/** 
 * @brief This pass modifies tokenizer ov::Model so that special tokens adding will be
 *  enabled or disabled depending on stateful value.
 *                                          
 *                                  +--------------+
 *                                  |  DefaultMode |
 *                                  +--------------+
 *                                         |
 *                                         v
 *                                  +------------+   +-----------+
 *                                  |  ReadValue |   |  INT_MAX  |
 *                                  +------------+   +-----------+
 *                                          \           /
 *                                           \         /
 *                                            v       v
 *   +--------------------+     +---------+  +---------+
 *   |  Const with tokens |     |  start  |  |   Mul   |
 *   +--------------------+     +---------+  +---------+
 *                         \          |          /
 *                           \        |         /
 *                             v      v        v
 *                            +-----------------+
 *                            |      Slice      |
 *                            +-----------------+
 *                                     |
 *                                     v
 *                          +----------------------+
 *                          |     VocabDecoder     |
 *                          +----------------------+
**/
class MakeVocabDecoderSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MakeVocabDecoderSatateful", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

const std::string ADD_SPECIAL_TOKENS_VAR_ID = "add_special_tokens";
const std::string SKIP_SPECIAL_TOKENS_VAR_ID = "skip_special_tokens";

} // namespace genai
} // namespace ov
