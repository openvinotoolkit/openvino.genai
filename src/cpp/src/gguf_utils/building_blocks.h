#include <vector>
#include <openvino/openvino.hpp>

using namespace ov;
using namespace opset;


Output<Node> causal_mask(const Output<Node>& attention_mask, const Output<Node>& keys,
                                int64_t hidden_dim, const Shape& input_shape);