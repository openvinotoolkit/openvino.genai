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
 * @brief This pass modifies tokenizer ov::Model so that inputs to RaggedToDense, CombineSegments 
 * become modifiable during runtime so that padding can be controlled.
 */
class MakePaddingSatateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MakePaddingSatateful");
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

inline const std::string ADD_SPECIAL_TOKENS_VAR_ID = "add_special_tokens";
inline const std::string SKIP_SPECIAL_TOKENS_VAR_ID = "skip_special_tokens";
inline const std::string MAX_LENGTH_VAR_ID = "max_length";
inline const std::string IS_MAX_LENGTH_SET = "is_max_length_set";
inline const std::string PAD_TO_MAX_LENGTH_VAR_ID = "pad_to_max_length";
inline const std::string PAD_RIGHT_VAR_ID = "pad_right";

} // namespace genai
} // namespace ov
