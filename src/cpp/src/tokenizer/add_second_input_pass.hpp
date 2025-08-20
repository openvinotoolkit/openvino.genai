#include "openvino/pass/manager.hpp"
#include <nlohmann/json.hpp>
#include "openvino/core/model.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"

namespace ov::genai {

/**
 * @class AddSecondInputPass
 * @brief Changes the ov::Model of tokenizer so that it accepts paired input.
 *
 * This pass modifies the model to accept two input sequences (paired input).
 * It concatenates both inputs, processes and tokenizes them together, and then splits them near the end,
 * before the CombineSegments node. If any constant SpecialToken depends on sequence inputs, they are zeroed.
 * The truncation operation is also modified to support max_length.
 */
class AddSecondInputPass : public ov::pass::ModelPass {
public:
    /**
    * @brief Constructs the AddSecondInputPass.
    *
    * This constructor initializes the pass with a shared object containing OpenVINO tokenizer functionality
    * and a reference to an error stream for reporting pass-specific errors.
    * It retrieves the tokenizer node factory function from the shared object for later use in the pass.
    *
    * @param openvino_tokenizers_shared_object Shared pointer to the OpenVINO tokenizer shared object.
    * @param pass_errors Reference to an output string stream for collecting error messages during the pass execution.
    */
    AddSecondInputPass(const std::shared_ptr<void>& openvino_tokenizers_shared_object, std::ostringstream& pass_errors):
    m_pass_errors(pass_errors) {
        m_node_factory = reinterpret_cast<FactoryCreateType>(get_symbol(openvino_tokenizers_shared_object, "create_tokenizer_node"));
    }

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
private:
    FactoryCreateType m_node_factory;
    std::vector<int> m_input_signature;
    std::vector<ov::Output<ov::Node>> m_inputs;
    nlohmann::json m_post_processor;
    ov::ParameterVector m_new_parameters;
    std::shared_ptr<ov::Node> m_equal_node;
    std::vector<ov::Output<ov::Node>> m_trunc_values;
    std::array<ov::Output<ov::Node>, 3> m_first_input;
    std::array<ov::Output<ov::Node>, 3> m_second_input;
    std::ostringstream& m_pass_errors;

    /// @brief Handles combining segment inputs and managing input_signature.
    /// The main sequence inputs are processed through nodes such as Truncate and CombineSegments.
    /// return true if inputs are parsed successfully, false otherwise.
    bool parse_inputs(std::shared_ptr<ov::Node>& combine_seg);
    
    /// @brief Asserts that the post-processor is present and retrieves it.
    /// If post-processor exists and allows paired input, it returns true.
    bool parse_and_assert_postprocessor(const std::shared_ptr<ov::Model>& model);
    
    /// @brief Inserts Splits for begins, ends before the CombineSegments node and returns new inputs.
    /// Also adds a modified Truncate operation for the second input.
    void insert_splits();

    /// @brief Creates new inputs for the CombineSegments node.
    /// It combines inputs for the first and second input, and adds special tokens for the second input.
    /// The new inputs are then returned as a list.
    std::vector<ov::Output<ov::Node>> get_new_inputs();
};

}
