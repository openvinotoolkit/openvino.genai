// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder_model_split.hpp"

#include <limits>
#include <optional>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr const char* DECODER_STATE_MARKER = ".decoder.";
constexpr const char* ENCODER_STATE_ID = "encoder_hidden_states";

struct RequiredInput {
    ov::Output<ov::Node> output;
    std::shared_ptr<ov::opset13::Parameter> parameter;
};

struct TextEmbeddingMatch {
    std::shared_ptr<ov::Node> gather;
    size_t hiddenSize;
};

struct EncoderStateMatch {
    std::shared_ptr<ov::opset13::ReadValue> readValue;
    std::shared_ptr<ov::opset13::Assign> assign;
    std::shared_ptr<ov::Node> shapeOf;
};

struct AudioMergeMatch {
    std::shared_ptr<ov::Node> select;
    int64_t audioTokenId;
};

int64_t getStaticRank(const ov::Output<ov::Node>& output) {
    const auto rank = output.get_partial_shape().rank();
    return rank.is_static() ? rank.get_length() : -1;
}

bool isCompatibleI64ShapeOf(const std::shared_ptr<ov::Node>& node) {
    // ShapeOf v0 and v3 have the same data input semantics needed by this
    // rewrite. Require i64 so every replacement preserves shape-tensor type.
    const bool compatibleVersion = ov::is_type<ov::op::v0::ShapeOf>(node) || ov::is_type<ov::op::v3::ShapeOf>(node);
    return compatibleVersion && node->get_output_size() == 1 && node->get_output_element_type(0) == ov::element::i64;
}

bool isEmbeddingGatherWithZeroBatchDims(const std::shared_ptr<ov::Node>& node) {
    // Accept compatible Gather revisions, but validate the version-specific
    // batch_dims attribute.
    if (const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node)) {
        return gather->get_batch_dims() == 0;
    }
    if (const auto gather = ov::as_type_ptr<ov::op::v7::Gather>(node)) {
        return gather->get_batch_dims() == 0;
    }
    return ov::is_type<ov::op::v1::Gather>(node);
}

// Detects a decompressed embedding weight feeding the Gather through a
// Convert-to-f32. Covers three exported forms:
//   * fp16-stored weight: Constant(f16) -> Convert(f32), no scale/zero-point;
//   * symmetric weight compression (int4_sym): Convert(Constant) * Constant(scale);
//   * asymmetric weight compression (int8): Convert(Constant) - Convert(zp), then * scale.
bool isDecompressedEmbeddingWeight(const ov::Output<ov::Node>& output) {
    const auto convertToF32 = ov::as_type_ptr<ov::opset13::Convert>(output.get_node_shared_ptr());
    if (convertToF32 == nullptr || convertToF32->get_output_element_type(0) != ov::element::f32) {
        return false;
    }

    // fp16-stored weights: the Convert consumes a Constant directly, with no
    // scale/zero-point dequantization in between.
    if (ov::is_type<ov::opset13::Constant>(convertToF32->input_value(0).get_node_shared_ptr())) {
        return true;
    }

    const auto multiply = ov::as_type_ptr<ov::opset13::Multiply>(convertToF32->input_value(0).get_node_shared_ptr());
    if (multiply == nullptr || multiply->get_input_size() != 2) {
        return false;
    }
    const bool scaleIsInput1 = ov::is_type<ov::opset13::Constant>(multiply->input_value(1).get_node_shared_ptr());
    const bool scaleIsInput0 = ov::is_type<ov::opset13::Constant>(multiply->input_value(0).get_node_shared_ptr());
    if (!scaleIsInput0 && !scaleIsInput1) {
        return false;
    }

    auto dequantized = scaleIsInput1 ? multiply->input_value(0) : multiply->input_value(1);
    // Asymmetric quantization subtracts a per-row zero point before scaling;
    // symmetric quantization (e.g. int4_sym) has no such term.
    if (const auto subtract = ov::as_type_ptr<ov::opset13::Subtract>(dequantized.get_node_shared_ptr())) {
        dequantized = subtract->input_value(0);
    }

    const auto convertFromLowPrecision = ov::as_type_ptr<ov::opset13::Convert>(dequantized.get_node_shared_ptr());
    return convertFromLowPrecision != nullptr &&
           ov::is_type<ov::opset13::Constant>(convertFromLowPrecision->input_value(0).get_node_shared_ptr());
}

bool hasTensorName(const ov::Output<ov::Node>& output, const std::string& name) {
    return output.get_names().count(name) != 0;
}

std::string formatNames(const std::set<std::string>& names) {
    std::ostringstream stream;
    stream << "{";
    for (auto iterator = names.begin(); iterator != names.end(); ++iterator) {
        if (iterator != names.begin()) {
            stream << ", ";
        }
        stream << *iterator;
    }
    stream << "}";
    return stream.str();
}

RequiredInput findRequiredInput(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    std::vector<ov::Output<ov::Node>> candidates;
    for (const auto& input : model->inputs()) {
        if (hasTensorName(input, name)) {
            candidates.push_back(input);
        }
    }

    OPENVINO_ASSERT(candidates.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one '",
                    name,
                    "' input, found ",
                    candidates.size());

    auto parameter = ov::as_type_ptr<ov::opset13::Parameter>(candidates.front().get_node_shared_ptr());
    OPENVINO_ASSERT(parameter != nullptr, "Qwen3-ASR input '", name, "' is not a Parameter");
    return {candidates.front(), std::move(parameter)};
}

void replaceExistingConsumers(const ov::Output<ov::Node>& oldOutput, const ov::Output<ov::Node>& newOutput) {
    const auto consumers = oldOutput.get_target_inputs();
    for (auto consumer : consumers) {
        consumer.replace_source_output(newOutput);
    }
}

bool outputDependsOn(const ov::Output<ov::Node>& output, const ov::Node* target) {
    std::vector<std::shared_ptr<ov::Node>> pending = {output.get_node_shared_ptr()};
    std::unordered_set<const ov::Node*> visited;

    while (!pending.empty()) {
        auto node = std::move(pending.back());
        pending.pop_back();
        if (node.get() == target) {
            return true;
        }
        if (!visited.insert(node.get()).second) {
            continue;
        }

        for (size_t index = 0; index < node->get_input_size(); ++index) {
            pending.push_back(node->input_value(index).get_node_shared_ptr());
        }
        for (const auto& dependency : node->get_control_dependencies()) {
            pending.push_back(dependency);
        }
    }
    return false;
}

std::unordered_set<const ov::Node*> collectReachableNodes(const ov::ResultVector& results,
                                                          const ov::SinkVector& sinks) {
    std::vector<std::shared_ptr<ov::Node>> pending;
    pending.reserve(results.size() + sinks.size());
    pending.insert(pending.end(), results.begin(), results.end());
    pending.insert(pending.end(), sinks.begin(), sinks.end());

    std::unordered_set<const ov::Node*> reachableNodes;
    while (!pending.empty()) {
        auto node = std::move(pending.back());
        pending.pop_back();
        if (!reachableNodes.insert(node.get()).second) {
            continue;
        }

        for (size_t index = 0; index < node->get_input_size(); ++index) {
            pending.push_back(node->input_value(index).get_node_shared_ptr());
        }
        for (const auto& dependency : node->get_control_dependencies()) {
            pending.push_back(dependency);
        }
    }
    return reachableNodes;
}

bool isReachableFromRoots(const ov::ResultVector& results, const ov::SinkVector& sinks, const ov::Node* target) {
    return collectReachableNodes(results, sinks).count(target) != 0;
}

std::vector<std::shared_ptr<ov::Node>> findDirectShapeOfConsumers(const ov::Output<ov::Node>& source) {
    std::vector<std::shared_ptr<ov::Node>> candidates;
    std::unordered_set<const ov::Node*> seen;
    for (const auto& target : source.get_target_inputs()) {
        auto* node = target.get_node();
        if (isCompatibleI64ShapeOf(node->shared_from_this()) && seen.insert(node).second) {
            candidates.push_back(node->shared_from_this());
        }
    }
    return candidates;
}

TextEmbeddingMatch findTextEmbedding(const std::shared_ptr<ov::Model>& model, const ov::Output<ov::Node>& inputIds) {
    std::vector<TextEmbeddingMatch> candidates;

    for (const auto& node : model->get_ops()) {
        if (!isEmbeddingGatherWithZeroBatchDims(node) || node->get_input_size() != 3) {
            continue;
        }

        const auto weightOutput = node->input_value(0);
        // The embedding table is either a plain f32 Constant, or (for int8/int4
        // weight-compressed exports) a dequantization chain rooted in one.
        const bool weightIsPlainConstant = ov::is_type<ov::opset13::Constant>(weightOutput.get_node_shared_ptr());
        if (!weightIsPlainConstant && !isDecompressedEmbeddingWeight(weightOutput)) {
            continue;
        }
        const auto convert = ov::as_type_ptr<ov::opset13::Convert>(node->input_value(1).get_node_shared_ptr());
        const auto axis = ov::as_type_ptr<ov::opset13::Constant>(node->input_value(2).get_node_shared_ptr());
        if (convert == nullptr || axis == nullptr || convert->get_input_size() != 1 ||
            convert->input_value(0) != inputIds) {
            continue;
        }

        const auto axisValues = axis->cast_vector<int64_t>();
        if (axisValues.size() != 1 || axisValues.front() != 0) {
            continue;
        }

        const auto weightShape = weightOutput.get_partial_shape();
        const auto outputShape = node->get_output_partial_shape(0);
        if (!weightShape.rank().is_static() || weightShape.rank().get_length() != 2 ||
            !outputShape.rank().is_static() || outputShape.rank().get_length() != 3 || !weightShape[1].is_static() ||
            !outputShape[2].is_static()) {
            continue;
        }

        const auto hiddenSize = weightShape[1].get_length();
        if (hiddenSize <= 0 || outputShape[2].get_length() != hiddenSize ||
            node->get_output_element_type(0) != ov::element::f32) {
            continue;
        }
        candidates.push_back({node, static_cast<size_t>(hiddenSize)});
    }

    OPENVINO_ASSERT(candidates.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one input_ids -> Convert -> "
                    "Gather(Constant or decompressed weight, axis=0) text embedding, found ",
                    candidates.size());
    return candidates.front();
}

EncoderStateMatch findEncoderState(const std::shared_ptr<ov::Model>& model,
                                   const ov::Output<ov::Node>& encoderHiddenStates) {
    std::vector<std::shared_ptr<ov::opset13::ReadValue>> readValues;
    for (const auto& target : encoderHiddenStates.get_target_inputs()) {
        auto readValue = ov::as_type_ptr<ov::opset13::ReadValue>(target.get_node()->shared_from_this());
        if (readValue != nullptr && target.get_index() == 0) {
            readValues.push_back(std::move(readValue));
        }
    }

    OPENVINO_ASSERT(readValues.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one ReadValue initialized by "
                    "encoder_hidden_states, found ",
                    readValues.size());
    const auto& readValue = readValues.front();
    OPENVINO_ASSERT(readValue->get_variable() != nullptr, "Qwen3-ASR encoder ReadValue has no Variable");
    OPENVINO_ASSERT(readValue->get_variable_id() == ENCODER_STATE_ID,
                    "Qwen3-ASR encoder ReadValue has unexpected variable ID '",
                    readValue->get_variable_id(),
                    "'");

    const auto shapeNodes = findDirectShapeOfConsumers(readValue->output(0));
    OPENVINO_ASSERT(shapeNodes.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one ShapeOf "
                    "consumer of the encoder state, found ",
                    shapeNodes.size());

    std::vector<std::shared_ptr<ov::opset13::Assign>> assigns;
    for (const auto& sink : model->get_sinks()) {
        auto assign = ov::as_type_ptr<ov::opset13::Assign>(sink);
        if (assign != nullptr && assign->get_variable() == readValue->get_variable()) {
            assigns.push_back(std::move(assign));
        }
    }
    OPENVINO_ASSERT(assigns.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one Assign for the "
                    "encoder state, found ",
                    assigns.size());

    return {readValue, assigns.front(), shapeNodes.front()};
}

AudioMergeMatch findAudioMerge(const std::shared_ptr<ov::Model>& model,
                               const ov::Output<ov::Node>& inputIds,
                               const TextEmbeddingMatch& textEmbedding,
                               const EncoderStateMatch& encoderState) {
    std::vector<AudioMergeMatch> candidates;

    for (const auto& node : model->get_ops()) {
        const auto select = ov::as_type_ptr<ov::opset13::Select>(node);
        if (select == nullptr || select->get_input_size() != 3 || select->get_output_size() != 1 ||
            getStaticRank(select->output(0)) != 3 || select->get_output_element_type(0) != ov::element::f32 ||
            !hasTensorName(select->output(0), "inputs_embeds")) {
            continue;
        }

        const auto outputShape = select->get_output_partial_shape(0);
        if (!outputShape[2].is_static() ||
            outputShape[2].get_length() != static_cast<int64_t>(textEmbedding.hiddenSize) ||
            select->input_value(2) != textEmbedding.gather->output(0) ||
            !outputDependsOn(select->input_value(1), encoderState.readValue.get())) {
            continue;
        }

        const auto condition = select->input_value(0);
        const auto unsqueeze = ov::as_type_ptr<ov::opset13::Unsqueeze>(condition.get_node_shared_ptr());
        if (unsqueeze == nullptr || unsqueeze->get_input_size() != 2 || getStaticRank(condition) != 3) {
            continue;
        }
        const auto conditionShape = condition.get_partial_shape();
        if (!conditionShape[2].is_static() || conditionShape[2].get_length() != 1) {
            continue;
        }

        const auto equal = ov::as_type_ptr<ov::opset13::Equal>(unsqueeze->input_value(0).get_node_shared_ptr());
        if (equal == nullptr || equal->get_input_size() != 2 || getStaticRank(equal->output(0)) != 2) {
            continue;
        }

        std::shared_ptr<ov::opset13::Constant> tokenConstant;
        if (equal->input_value(0) == inputIds) {
            tokenConstant = ov::as_type_ptr<ov::opset13::Constant>(equal->input_value(1).get_node_shared_ptr());
        } else if (equal->input_value(1) == inputIds) {
            tokenConstant = ov::as_type_ptr<ov::opset13::Constant>(equal->input_value(0).get_node_shared_ptr());
        }
        if (tokenConstant == nullptr || tokenConstant->get_element_type() != ov::element::i64 ||
            !tokenConstant->get_output_partial_shape(0).is_static()) {
            continue;
        }

        const auto tokenValues = tokenConstant->cast_vector<int64_t>();
        if (tokenValues.size() != 1) {
            continue;
        }
        candidates.push_back({select, tokenValues.front()});
    }

    OPENVINO_ASSERT(candidates.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one audio-merge Select with "
                    "Unsqueeze(Equal(input_ids, singleton i64 Constant)), found ",
                    candidates.size());
    return candidates.front();
}

ov::Output<ov::Node> findPositionIdsOutput(const std::shared_ptr<ov::Model>& model) {
    std::vector<ov::Output<ov::Node>> candidates;

    for (const auto& node : model->get_ops()) {
        for (size_t index = 0; index < node->get_output_size(); ++index) {
            const auto output = node->output(index);
            const auto broadcast = ov::as_type_ptr<ov::opset13::Broadcast>(node);
            if (!hasTensorName(output, "position_ids") || output.get_element_type() != ov::element::i64 ||
                getStaticRank(output) != 3 || broadcast == nullptr) {
                continue;
            }

            const auto outputShape = output.get_partial_shape();
            if (!outputShape[0].is_static() || outputShape[0].get_length() != 3 || broadcast->get_input_size() != 2 ||
                getStaticRank(broadcast->input_value(0)) != 3) {
                continue;
            }

            const auto broadcastInputShape = broadcast->input_value(0).get_partial_shape();
            const auto targetShapeShape = broadcast->input_value(1).get_partial_shape();
            if (!broadcastInputShape[0].is_static() || broadcastInputShape[0].get_length() != 1 ||
                !targetShapeShape.rank().is_static() || targetShapeShape.rank().get_length() != 1 ||
                !targetShapeShape[0].is_static() || targetShapeShape[0].get_length() != 3 ||
                output.get_target_inputs().empty()) {
                continue;
            }
            candidates.push_back(output);
        }
    }

    OPENVINO_ASSERT(candidates.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one "
                    "position_ids Broadcast [1,B,S] -> [3,B,S], found ",
                    candidates.size());
    return candidates.front();
}

std::shared_ptr<ov::opset13::Range> traceRangeThroughUnsqueezes(ov::Output<ov::Node> output) {
    std::unordered_set<const ov::Node*> visited;
    size_t unsqueezeCount = 0;
    while (true) {
        auto node = output.get_node_shared_ptr();
        if (!visited.insert(node.get()).second) {
            return nullptr;
        }
        if (const auto range = ov::as_type_ptr<ov::opset13::Range>(node)) {
            return unsqueezeCount > 0 && range->get_input_size() == 3 &&
                           range->get_output_element_type(0) == ov::element::i64 && getStaticRank(range->output(0)) == 1
                       ? range
                       : nullptr;
        }
        const auto unsqueeze = ov::as_type_ptr<ov::opset13::Unsqueeze>(node);
        if (unsqueeze == nullptr || unsqueeze->get_input_size() != 2) {
            return nullptr;
        }
        ++unsqueezeCount;
        output = unsqueeze->input_value(0);
    }
}

// Resolves a scalar f32 value from either a plain f32 rank-0 Constant, or a
// rank-0 Constant of another type (e.g. f16) feeding an f32 Convert -- the
// causal-mask thresholds are stored this second way in some exports.
std::optional<float> resolveScalarFloatConstant(const ov::Output<ov::Node>& output) {
    if (const auto constant = ov::as_type_ptr<ov::opset13::Constant>(output.get_node_shared_ptr())) {
        if (constant->get_element_type() != ov::element::f32 || getStaticRank(output) != 0) {
            return std::nullopt;
        }
        const auto values = constant->cast_vector<float>();
        return values.size() == 1 ? std::optional<float>(values.front()) : std::nullopt;
    }

    const auto convert = ov::as_type_ptr<ov::opset13::Convert>(output.get_node_shared_ptr());
    if (convert == nullptr || convert->get_output_element_type(0) != ov::element::f32 || getStaticRank(output) != 0) {
        return std::nullopt;
    }
    const auto constant = ov::as_type_ptr<ov::opset13::Constant>(convert->input_value(0).get_node_shared_ptr());
    if (constant == nullptr || getStaticRank(constant->output(0)) != 0) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<float>();
    return values.size() == 1 ? std::optional<float>(values.front()) : std::nullopt;
}

bool hasCausalLessEqualRangeTopology(const std::shared_ptr<ov::opset13::Select>& select) {
    if (select->get_input_size() != 3 || getStaticRank(select->input_value(0)) != 4 ||
        select->input_value(0).get_element_type() != ov::element::boolean) {
        return false;
    }

    const auto conditionBroadcast =
        ov::as_type_ptr<ov::opset13::Broadcast>(select->input_value(0).get_node_shared_ptr());
    if (conditionBroadcast == nullptr || conditionBroadcast->get_input_size() != 2) {
        return false;
    }

    const auto lessEqual =
        ov::as_type_ptr<ov::opset13::LessEqual>(conditionBroadcast->input_value(0).get_node_shared_ptr());
    if (lessEqual == nullptr || lessEqual->get_input_size() != 2 ||
        lessEqual->get_output_element_type(0) != ov::element::boolean || getStaticRank(lessEqual->output(0)) != 4) {
        return false;
    }

    const auto lhsRange = traceRangeThroughUnsqueezes(lessEqual->input_value(0));
    const auto rhsRange = traceRangeThroughUnsqueezes(lessEqual->input_value(1));
    if (lhsRange == nullptr || rhsRange == nullptr || lhsRange == rhsRange) {
        return false;
    }

    const auto zeroValue = resolveScalarFloatConstant(select->input_value(1));
    const auto negativeValue = resolveScalarFloatConstant(select->input_value(2));
    if (!zeroValue.has_value() || !negativeValue.has_value()) {
        return false;
    }
    return *zeroValue == 0.0f && *negativeValue < 0.0f;
}

bool hasExclusiveSdpaMaskPaths(const std::shared_ptr<ov::opset13::Select>& select,
                               const std::vector<std::shared_ptr<ov::opset13::ScaledDotProductAttention>>& sdpaNodes) {
    const auto selectOutput = select->output(0);
    const auto directConsumers = selectOutput.get_target_inputs();
    if (directConsumers.size() != sdpaNodes.size()) {
        return false;
    }

    std::unordered_set<const ov::Node*> allSlices;
    std::unordered_set<const ov::Node*> rootSlices;
    for (const auto& sdpa : sdpaNodes) {
        if (sdpa->get_input_size() <= 3 || sdpa->input_value(3).get_element_type() != ov::element::f32 ||
            getStaticRank(sdpa->input_value(3)) != 4) {
            return false;
        }

        auto current = sdpa->input_value(3);
        const ov::Node* rootSlice = nullptr;
        size_t sliceCount = 0;
        while (const auto slice = ov::as_type_ptr<ov::opset13::Slice>(current.get_node_shared_ptr())) {
            if (slice->get_input_size() < 1 || slice->get_output_size() != 1 ||
                slice->output(0).get_target_inputs().size() != 1 || !allSlices.insert(slice.get()).second) {
                return false;
            }
            rootSlice = slice.get();
            ++sliceCount;
            current = slice->input_value(0);
        }
        if (sliceCount == 0 || current != selectOutput || !rootSlices.insert(rootSlice).second) {
            return false;
        }
    }

    if (rootSlices.size() != sdpaNodes.size()) {
        return false;
    }
    for (const auto& consumer : directConsumers) {
        if (consumer.get_index() != 0 || !ov::is_type<ov::opset13::Slice>(consumer.get_node()->shared_from_this()) ||
            rootSlices.count(consumer.get_node()) != 1) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<ov::opset13::Select> findCausalMaskSelect(const std::shared_ptr<ov::Model>& model,
                                                          const std::shared_ptr<ov::Node>& audioSelect,
                                                          size_t expectedAttentionLayerCount) {
    std::vector<std::shared_ptr<ov::opset13::ScaledDotProductAttention>> sdpaNodes;
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::opset13::ScaledDotProductAttention>(node)) {
            OPENVINO_ASSERT(!sdpa->get_causal(), "Qwen3-ASR SDPA must use the explicit shared mask");
            sdpaNodes.push_back(sdpa);
        }
    }
    OPENVINO_ASSERT(sdpaNodes.size() == expectedAttentionLayerCount,
                    "Qwen3-ASR decoder split expected ",
                    expectedAttentionLayerCount,
                    " SDPA layers, found ",
                    sdpaNodes.size());

    std::vector<std::shared_ptr<ov::opset13::Select>> candidates;
    for (const auto& node : model->get_ops()) {
        const auto select = ov::as_type_ptr<ov::opset13::Select>(node);
        if (node == audioSelect || select == nullptr || select->get_output_size() != 1 ||
            select->get_output_element_type(0) != ov::element::f32 || getStaticRank(select->output(0)) != 4) {
            continue;
        }
        if (hasCausalLessEqualRangeTopology(select) && hasExclusiveSdpaMaskPaths(select, sdpaNodes)) {
            candidates.push_back(select);
        }
    }

    OPENVINO_ASSERT(candidates.size() == 1,
                    "Qwen3-ASR decoder split expected exactly one causal "
                    "LessEqual/Range Select feeding ",
                    expectedAttentionLayerCount,
                    " unique Slice-to-SDPA mask paths, found ",
                    candidates.size());
    return candidates.front();
}

std::vector<std::shared_ptr<ov::Node>> findInputShapeNodes(const ov::Output<ov::Node>& inputIds) {
    std::vector<std::shared_ptr<ov::Node>> shapeNodes;
    std::unordered_set<const ov::Node*> seen;
    const auto addShapeNode = [&](const std::shared_ptr<ov::Node>& node) {
        if (seen.insert(node.get()).second) {
            shapeNodes.push_back(node);
        }
    };

    for (const auto& target : inputIds.get_target_inputs()) {
        auto node = target.get_node()->shared_from_this();
        if (isCompatibleI64ShapeOf(node)) {
            addShapeNode(node);
            continue;
        }
        const auto convert = ov::as_type_ptr<ov::opset13::Convert>(node);
        if (convert == nullptr || convert->get_output_size() != 1) {
            continue;
        }

        for (const auto& convertedTarget : convert->output(0).get_target_inputs()) {
            auto convertedConsumer = convertedTarget.get_node()->shared_from_this();
            if (isCompatibleI64ShapeOf(convertedConsumer)) {
                addShapeNode(convertedConsumer);
            }
        }
    }

    OPENVINO_ASSERT(!shapeNodes.empty(), "Qwen3-ASR decoder split found no ShapeOf dependency on input_ids");
    return shapeNodes;
}

std::string removeDecoderStateMarkers(std::string variableId) {
    size_t position = 0;
    while ((position = variableId.find(DECODER_STATE_MARKER, position)) != std::string::npos) {
        variableId.replace(position, std::char_traits<char>::length(DECODER_STATE_MARKER), ".");
        ++position;
    }
    return variableId;
}

ov::SinkVector normalizeKvStateNames(const std::shared_ptr<ov::Model>& model, const ov::SinkVector& sinks) {
    std::unordered_map<const ov::op::util::Variable*, std::vector<std::shared_ptr<ov::opset13::ReadValue>>>
        readValuesByVariable;
    for (const auto& node : model->get_ops()) {
        auto readValue = ov::as_type_ptr<ov::opset13::ReadValue>(node);
        if (readValue != nullptr && readValue->get_variable() != nullptr) {
            readValuesByVariable[readValue->get_variable().get()].push_back(std::move(readValue));
        }
    }

    std::unordered_set<std::string> normalizedIds;

    for (const auto& sink : sinks) {
        auto assign = ov::as_type_ptr<ov::opset13::Assign>(sink);
        OPENVINO_ASSERT(assign != nullptr, "Qwen3-ASR pure LM contains a non-Assign sink");
        OPENVINO_ASSERT(assign->get_variable() != nullptr, "Qwen3-ASR KV Assign has no Variable");

        const auto variable = assign->get_variable();
        const auto variableId = variable->get_info().variable_id;
        OPENVINO_ASSERT(!variableId.empty(), "Qwen3-ASR KV Assign has an empty variable ID");
        OPENVINO_ASSERT(variableId != ENCODER_STATE_ID, "Qwen3-ASR encoder_hidden_states Assign was not removed");

        const auto readValuesIterator = readValuesByVariable.find(variable.get());
        OPENVINO_ASSERT(readValuesIterator != readValuesByVariable.end() && readValuesIterator->second.size() == 1,
                        "Qwen3-ASR decoder split expected exactly one ReadValue for KV state '",
                        variableId,
                        "'");
        const auto& readValue = readValuesIterator->second.front();
        OPENVINO_ASSERT(readValue->get_variable() == variable,
                        "Qwen3-ASR KV ReadValue and Assign do not share Variable '",
                        variableId,
                        "'");
        OPENVINO_ASSERT(
            readValue->get_variable_id() == assign->get_variable_id() && assign->get_variable_id() == variableId,
            "Qwen3-ASR KV state IDs disagree before normalization");

        auto variableInfo = variable->get_info();
        const auto originalShape = variableInfo.data_shape;
        const auto originalType = variableInfo.data_type;
        variableInfo.variable_id = removeDecoderStateMarkers(variableId);
        OPENVINO_ASSERT(variableInfo.variable_id.find(DECODER_STATE_MARKER) == std::string::npos,
                        "Qwen3-ASR failed to normalize KV variable ID '",
                        variableId,
                        "'");
        OPENVINO_ASSERT(normalizedIds.insert(variableInfo.variable_id).second,
                        "Qwen3-ASR duplicate normalized KV variable ID '",
                        variableInfo.variable_id,
                        "'");

        // Updating the cloned Variable preserves the existing state nodes,
        // metadata, control dependencies, and every consumer edge.
        variable->update(variableInfo);
        OPENVINO_ASSERT(
            variable->get_info().data_shape == originalShape && variable->get_info().data_type == originalType,
            "Qwen3-ASR KV normalization changed VariableInfo for state '",
            variableId,
            "'");
        OPENVINO_ASSERT(readValue->get_variable() == assign->get_variable() && readValue->get_variable() == variable,
                        "Qwen3-ASR KV ReadValue and Assign lost their shared Variable");
        OPENVINO_ASSERT(readValue->get_variable_id() == variableInfo.variable_id &&
                            assign->get_variable_id() == variableInfo.variable_id,
                        "Qwen3-ASR KV state IDs disagree after normalization");
    }

    return sinks;
}

std::shared_ptr<ov::opset13::Parameter> makeParameter(const ov::element::Type& type,
                                                      const ov::PartialShape& shape,
                                                      const std::string& name) {
    auto parameter = std::make_shared<ov::opset13::Parameter>(type, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).get_tensor().set_names({name});
    return parameter;
}

std::shared_ptr<ov::Model> buildTextEmbeddingModel(const TextEmbeddingMatch& match, const RequiredInput& inputIds) {
    match.gather->output(0).get_tensor().set_names({"inputs_embeds"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{match.gather->output(0)},
                                             ov::ParameterVector{inputIds.parameter},
                                             "qwen3_asr_text_embedding");
    model->validate_nodes_and_infer_types();

    OPENVINO_ASSERT(model->inputs().size() == 1 && hasTensorName(model->input(0), "input_ids"),
                    "Qwen3-ASR text embedding must have only the input_ids input");
    OPENVINO_ASSERT(model->outputs().size() == 1 && hasTensorName(model->output(0), "inputs_embeds"),
                    "Qwen3-ASR text embedding must have only the inputs_embeds output");
    return model;
}

void validateLanguageModel(const std::shared_ptr<ov::Model>& model,
                           const RequiredInput& inputIds,
                           const RequiredInput& encoderHiddenStates,
                           const EncoderStateMatch& encoderState,
                           size_t expectedKvSinkCount) {
    const std::set<std::string> expectedInputs = {
        "attention_mask",
        "beam_idx",
        "inputs_embeds",
        "position_ids",
    };
    const auto reachableNodes = collectReachableNodes(model->get_results(), model->get_sinks());
    std::set<std::string> actualInputs;
    for (const auto& input : model->inputs()) {
        std::vector<std::string> matchingNames;
        for (const auto& expectedName : expectedInputs) {
            if (hasTensorName(input, expectedName)) {
                matchingNames.push_back(expectedName);
            }
        }
        OPENVINO_ASSERT(matchingNames.size() == 1, "Qwen3-ASR pure LM input has missing or ambiguous canonical name");
        actualInputs.insert(matchingNames.front());
        OPENVINO_ASSERT(reachableNodes.count(input.get_node()) == 1,
                        "Qwen3-ASR pure LM input '",
                        matchingNames.front(),
                        "' is not reachable from this model's results or sinks");
    }
    OPENVINO_ASSERT(model->inputs().size() == expectedInputs.size() && actualInputs == expectedInputs,
                    "Unexpected Qwen3-ASR pure LM inputs: ",
                    formatNames(actualInputs),
                    "; expected ",
                    formatNames(expectedInputs));

    OPENVINO_ASSERT(expectedKvSinkCount != 0 && expectedKvSinkCount % 2 == 0,
                    "Qwen3-ASR expected KV sink count must be even and nonzero");
    OPENVINO_ASSERT(model->get_sinks().size() == expectedKvSinkCount,
                    "Qwen3-ASR pure LM expected ",
                    expectedKvSinkCount,
                    " KV Assign sinks, got ",
                    model->get_sinks().size());

    std::unordered_map<std::string, std::vector<std::shared_ptr<ov::opset13::ReadValue>>> readValuesById;
    for (const auto& node : model->get_ops()) {
        auto readValue = ov::as_type_ptr<ov::opset13::ReadValue>(node);
        if (readValue == nullptr) {
            continue;
        }
        const auto variableId = readValue->get_variable_id();
        OPENVINO_ASSERT(variableId != ENCODER_STATE_ID, "encoder_hidden_states ReadValue remains in Qwen3-ASR pure LM");
        OPENVINO_ASSERT(variableId.find(DECODER_STATE_MARKER) == std::string::npos,
                        "Unnormalized Qwen3-ASR KV ReadValue remains: '",
                        variableId,
                        "'");
        readValuesById[variableId].push_back(std::move(readValue));
    }

    OPENVINO_ASSERT(readValuesById.size() == expectedKvSinkCount,
                    "Qwen3-ASR pure LM expected ",
                    expectedKvSinkCount,
                    " KV ReadValue states, got ",
                    readValuesById.size());
    for (const auto& sink : model->get_sinks()) {
        const auto assign = ov::as_type_ptr<ov::opset13::Assign>(sink);
        OPENVINO_ASSERT(assign != nullptr, "Qwen3-ASR pure LM contains a non-Assign sink");
        const auto variableId = assign->get_variable_id();
        const auto readValueIterator = readValuesById.find(variableId);
        OPENVINO_ASSERT(readValueIterator != readValuesById.end() && readValueIterator->second.size() == 1,
                        "Qwen3-ASR KV Assign '",
                        variableId,
                        "' does not have exactly one matching ReadValue");
        OPENVINO_ASSERT(assign->get_variable() == readValueIterator->second.front()->get_variable(),
                        "Qwen3-ASR KV ReadValue and Assign do not share Variable '",
                        variableId,
                        "'");
        OPENVINO_ASSERT(variableId.find(DECODER_STATE_MARKER) == std::string::npos,
                        "Unnormalized Qwen3-ASR KV Assign remains: '",
                        variableId,
                        "'");
    }

    OPENVINO_ASSERT(reachableNodes.count(inputIds.parameter.get()) == 0,
                    "input_ids remains reachable in Qwen3-ASR pure LM");
    OPENVINO_ASSERT(reachableNodes.count(encoderHiddenStates.parameter.get()) == 0,
                    "encoder_hidden_states Parameter remains reachable in Qwen3-ASR pure LM");
    OPENVINO_ASSERT(reachableNodes.count(encoderState.readValue.get()) == 0,
                    "encoder_hidden_states ReadValue remains reachable in Qwen3-ASR pure LM");
    OPENVINO_ASSERT(reachableNodes.count(encoderState.assign.get()) == 0,
                    "encoder_hidden_states Assign remains reachable in Qwen3-ASR pure LM");
}

}  // namespace

namespace ov::genai {

Qwen3ASRDecoderModels splitQwen3ASRDecoderModel(const std::shared_ptr<ov::Model>& decoderModel) {
    OPENVINO_ASSERT(decoderModel != nullptr, "Cannot split a null Qwen3-ASR decoder model");

    // All matching and rewrites operate on a clone so callers retain the original
    // decoder graph.
    auto model = decoderModel->clone();
    const auto inputIds = findRequiredInput(model, "input_ids");
    const auto encoderHiddenStates = findRequiredInput(model, "encoder_hidden_states");
    const auto beamIdx = findRequiredInput(model, "beam_idx");

    OPENVINO_ASSERT(inputIds.output.get_element_type() == ov::element::i64 && getStaticRank(inputIds.output) == 2,
                    "Qwen3-ASR input_ids must be a rank-2 i64 tensor");
    OPENVINO_ASSERT(encoderHiddenStates.output.get_element_type() == ov::element::f32 &&
                        getStaticRank(encoderHiddenStates.output) == 3,
                    "Qwen3-ASR encoder_hidden_states must be a rank-3 f32 tensor");
    OPENVINO_ASSERT(beamIdx.output.get_element_type() == ov::element::i32 && getStaticRank(beamIdx.output) == 1,
                    "Qwen3-ASR beam_idx must be a rank-1 i32 tensor");

    const auto textEmbeddingMatch = findTextEmbedding(model, inputIds.output);
    const auto encoderState = findEncoderState(model, encoderHiddenStates.output);
    const auto audioMerge = findAudioMerge(model, inputIds.output, textEmbeddingMatch, encoderState);
    const auto positionIdsOutput = findPositionIdsOutput(model);
    const auto inputShapeNodes = findInputShapeNodes(inputIds.output);

    const auto encoderShape = encoderHiddenStates.output.get_partial_shape();
    OPENVINO_ASSERT(encoderShape[2].is_static() &&
                        encoderShape[2].get_length() == static_cast<int64_t>(textEmbeddingMatch.hiddenSize),
                    "Qwen3-ASR encoder and text embedding hidden sizes do not match");

    ov::SinkVector kvSinks;
    kvSinks.reserve(model->get_sinks().size());
    for (const auto& sink : model->get_sinks()) {
        if (sink != encoderState.assign) {
            kvSinks.push_back(sink);
        }
    }
    const auto kvSinkCount = kvSinks.size();
    OPENVINO_ASSERT(kvSinkCount != 0 && kvSinkCount % 2 == 0,
                    "Qwen3-ASR decoder split expected an even, nonzero KV sink count after "
                    "removing the encoder state, found ",
                    kvSinkCount);
    const auto attentionLayerCount = kvSinkCount / 2;
    const auto causalMaskSelect = findCausalMaskSelect(model, audioMerge.select, attentionLayerCount);

    auto textEmbedding = buildTextEmbeddingModel(textEmbeddingMatch, inputIds);
    auto inputsEmbeds = makeParameter(ov::element::f32,
                                      ov::PartialShape{ov::Dimension::dynamic(),
                                                       ov::Dimension::dynamic(),
                                                       static_cast<int64_t>(textEmbeddingMatch.hiddenSize)},
                                      "inputs_embeds");
    auto attentionMask = makeParameter(ov::element::i64,
                                       ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       "attention_mask");
    auto positionIds = makeParameter(ov::element::i64,
                                     ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                     "position_ids");

    replaceExistingConsumers(audioMerge.select->output(0), inputsEmbeds->output(0));

    // Preserve the model's native [1,B,S] -> [3,B,S] position broadcast behind a
    // rank-2 API.
    auto positionAxes = ov::opset13::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto positionIdsUnsqueezed = std::make_shared<ov::opset13::Unsqueeze>(positionIds, positionAxes);
    positionIdsOutput.get_node_shared_ptr()->input(0).replace_source_output(positionIdsUnsqueezed->output(0));

    auto inputsEmbedsShape = std::make_shared<ov::opset13::ShapeOf>(inputsEmbeds->output(0), ov::element::i64);
    auto sliceStart = ov::opset13::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto sliceStop = ov::opset13::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto sliceStep = ov::opset13::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto sliceAxes = ov::opset13::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto inputShapePrefix =
        std::make_shared<ov::opset13::Slice>(inputsEmbedsShape, sliceStart, sliceStop, sliceStep, sliceAxes);
    for (const auto& shapeNode : inputShapeNodes) {
        replaceExistingConsumers(shapeNode->output(0), inputShapePrefix->output(0));
    }
    replaceExistingConsumers(encoderState.shapeOf->output(0), inputsEmbedsShape->output(0));

    const auto causalMaskConsumers = causalMaskSelect->output(0).get_target_inputs();
    auto validMask = std::make_shared<ov::opset13::Convert>(attentionMask, ov::element::boolean);
    auto attentionAxes = ov::opset13::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
    auto validMask4D = std::make_shared<ov::opset13::Unsqueeze>(validMask, attentionAxes);
    auto negativeInfinity =
        ov::opset13::Constant::create(ov::element::f32, ov::Shape{}, {std::numeric_limits<float>::lowest()});
    auto maskedCausal =
        std::make_shared<ov::opset13::Select>(validMask4D, causalMaskSelect->output(0), negativeInfinity);
    for (auto consumer : causalMaskConsumers) {
        consumer.replace_source_output(maskedCausal->output(0));
    }

    auto normalizedKvSinks = normalizeKvStateNames(model, kvSinks);
    OPENVINO_ASSERT(!isReachableFromRoots(model->get_results(), normalizedKvSinks, inputIds.parameter.get()),
                    "Qwen3-ASR split left a reachable input_ids dependency");
    OPENVINO_ASSERT(!isReachableFromRoots(model->get_results(), normalizedKvSinks, encoderHiddenStates.parameter.get()),
                    "Qwen3-ASR split left a reachable encoder_hidden_states Parameter");
    OPENVINO_ASSERT(!isReachableFromRoots(model->get_results(), normalizedKvSinks, encoderState.readValue.get()),
                    "Qwen3-ASR split left a reachable encoder_hidden_states ReadValue");

    auto languageModel =
        std::make_shared<ov::Model>(model->get_results(),
                                    normalizedKvSinks,
                                    ov::ParameterVector{inputsEmbeds, attentionMask, positionIds, beamIdx.parameter},
                                    "qwen3_asr_pure_lm");
    languageModel->validate_nodes_and_infer_types();
    validateLanguageModel(languageModel, inputIds, encoderHiddenStates, encoderState, kvSinkCount);

    return {std::move(textEmbedding), std::move(languageModel), audioMerge.audioTokenId, textEmbeddingMatch.hiddenSize};
}

}  // namespace ov::genai
