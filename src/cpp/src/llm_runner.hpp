// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

namespace ov {
namespace genai {

class LLMRunnerBase {
public:
    using Ptr = std::shared_ptr<LLMRunnerBase>;
    virtual void infer() = 0;
    virtual void reset_state() = 0;
    virtual cv::GCompiledModel& get_compile_model() = 0;
};

}  // namespace genai
}  // namespace ov
