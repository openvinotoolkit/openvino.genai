// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/matcher_pass.hpp"

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
class MakeAddSpecialTokensSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MakeAddSpecialTokensSatateful");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/** 
 * @brief This pass modifies tokenizer ov::Model so that max_pad_length input to
 * RaggedToDense is made modifiable during runtime so that padding can be controlled.
 */
class MakePaddingSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MakePaddingSatateful");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/** 
 * @brief This pass modifies tokenizer ov::Model so that max_pad_length input to
 * RaggedToDense is made modifiable during runtime so that padding can be controlled.
 */
class MakeTruncationSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MakeTruncationSatateful");
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
    OPENVINO_MODEL_PASS_RTTI("MakeVocabDecoderSatateful");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

const std::string ADD_SPECIAL_TOKENS_VAR_ID = "add_special_tokens";
const std::string SKIP_SPECIAL_TOKENS_VAR_ID = "skip_special_tokens";
const std::string MAX_PAD_LENGTH_VAR_ID = "max_pad_length";
const std::string MAX_TRUNCATION_LENGTH_VAR_ID = "max_truncation_length";

} // namespace genai
} // namespace ov
