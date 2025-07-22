#include "openvino/pass/manager.hpp"
#include <nlohmann/json.hpp>
#include "openvino/core/model.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"

namespace ov::genai {

class AddSecondInputPass : public ov::pass::ModelPass {
public:
    AddSecondInputPass(const std::shared_ptr<void>& openvino_tokenizers_shared_object) {
        node_factory = reinterpret_cast<FactoryCreateType>(get_symbol(openvino_tokenizers_shared_object, "create_tokenizer_node"));
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
private:
    FactoryCreateType node_factory;
    std::vector<int> input_signature;
    std::vector<ov::Output<ov::Node>> inputs;
    nlohmann::json post_processor;
    ov::ParameterVector new_parameters;
    std::shared_ptr<ov::Node> equal_node;
    std::vector<ov::Output<ov::Node>> trunc_values;
    std::array<ov::Output<ov::Node>, 3> first_input;
    std::array<ov::Output<ov::Node>, 3> second_input;

    bool parse_inputs(std::shared_ptr<ov::Node>& combine_seg);
    bool assert_and_get_postprocessor(const std::shared_ptr<ov::Model>& model);
    void insert_splits();
    std::vector<ov::Output<ov::Node>> get_new_inputs(); 
};

}
