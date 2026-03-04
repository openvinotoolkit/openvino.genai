#pragma once

#include <vector>
#include <openvino/runtime/tensor.hpp>

namespace ov::genai::module {
namespace utils {

class IVideoProcessor {
public:
    virtual ~IVideoProcessor() = default;

    // virtual void sample_frames(VideoMetadata metadata, int num_frames = 0, float fps = 0.0f) = 0;
    virtual void preprocess(const std::vector<ov::Tensor>& frames) = 0;
};

}  // namespace utils
}  // namespace ov::genai::module