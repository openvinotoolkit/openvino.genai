// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/dummy_vl/classes.hpp"

#include <numeric>

namespace ov::genai {

namespace {
ov::Shape get_position_ids_shape(const ov::Tensor& template_tensor, size_t inputs_embeds_size) {
	const ov::Shape template_shape = template_tensor.get_shape();
	if (template_shape.size() == 3) {
		ov::Shape shape = template_shape;
		shape[2] = inputs_embeds_size;
		return shape;
	}
	if (template_shape.size() == 2) {
		ov::Shape shape = template_shape;
		shape[1] = inputs_embeds_size;
		return shape;
	}
	return {1, inputs_embeds_size};
}

void fill_position_ids(ov::Tensor& position_ids, size_t history_size) {
	const ov::Shape shape = position_ids.get_shape();
	int64_t* data = position_ids.data<int64_t>();
	if (shape.size() == 3) {
		const size_t rows = shape[0];
		const size_t tokens = shape[2];
		for (size_t row = 0; row < rows; ++row) {
			for (size_t token_idx = 0; token_idx < tokens; ++token_idx) {
				data[row * tokens + token_idx] = static_cast<int64_t>(history_size + token_idx);
			}
		}
		return;
	}
	if (shape.size() == 2) {
		const size_t rows = shape[0];
		const size_t tokens = shape[1];
		for (size_t row = 0; row < rows; ++row) {
			for (size_t token_idx = 0; token_idx < tokens; ++token_idx) {
				data[row * tokens + token_idx] = static_cast<int64_t>(history_size + token_idx);
			}
		}
		return;
	}
	for (size_t token_idx = 0; token_idx < position_ids.get_size(); ++token_idx) {
		data[token_idx] = static_cast<int64_t>(history_size + token_idx);
	}
}
} // namespace

VisionEncoderDummyVL::VisionEncoderDummyVL(
	const std::filesystem::path& model_dir,
	const std::string& device,
	const ov::AnyMap properties) :
	VisionEncoder() {
	(void)model_dir;
	(void)device;
	(void)properties;
}

VisionEncoderDummyVL::VisionEncoderDummyVL(
	const ModelsMap& models_map,
	const std::filesystem::path& config_dir_path,
	const std::string& device,
	const ov::AnyMap properties) :
	VisionEncoder() {
	(void)models_map;
	(void)config_dir_path;
	(void)device;
	(void)properties;
}

EncodedImage VisionEncoderDummyVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
	(void)image;
	(void)config_map;
	return {};
}

EncodedVideo VisionEncoderDummyVL::encode_frames(const std::vector<ov::Tensor>& frames) {
	(void)frames;
	return {};
}

InputsEmbedderDummyVL::InputsEmbedderDummyVL(
	const VLMConfig& vlm_config,
	const std::filesystem::path& model_dir,
	const std::string& device,
	const ov::AnyMap device_config) :
	IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderDummyVL::InputsEmbedderDummyVL(
	const VLMConfig& vlm_config,
	const ModelsMap& models_map,
	const Tokenizer& tokenizer,
	const std::filesystem::path& config_dir_path,
	const std::string& device,
	const ov::AnyMap device_config) :
	IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

ov::Tensor InputsEmbedderDummyVL::get_inputs_embeds(
	const std::string& prompt,
	const std::vector<ov::genai::EncodedImage>& images,
	ov::genai::VLMPerfMetrics& metrics,
	bool recalculate_merged_embeddings,
	const std::vector<size_t>& image_sequence) {
	(void)prompt;
	(void)images;
	(void)metrics;
	(void)recalculate_merged_embeddings;
	(void)image_sequence;
	return {};
}

ov::Tensor InputsEmbedderDummyVL::get_inputs_embeds(
	const std::string& prompt,
	const std::vector<ov::genai::EncodedImage>& images,
	const std::vector<ov::genai::EncodedVideo>& videos,
	ov::genai::VLMPerfMetrics& metrics,
	bool recalculate_merged_embeddings,
	const std::vector<size_t>& image_sequence,
	const std::vector<size_t>& videos_sequence,
	const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
	(void)prompt;
	(void)images;
	(void)videos;
	(void)metrics;
	(void)recalculate_merged_embeddings;
	(void)image_sequence;
	(void)videos_sequence;
	(void)history_vision_count;
	return {};
}

std::vector<ov::genai::EncodedImage> InputsEmbedderDummyVL::encode_images(const std::vector<ov::Tensor>& images) {
	(void)images;
	return {};
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderDummyVL::encode_videos(const std::vector<ov::Tensor>& videos, const std::vector<VideoMetadata>& videos_metadata) {
	(void)videos;
	(void)videos_metadata;
	return {};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderDummyVL::get_position_ids(
	const size_t inputs_embeds_size,
	const size_t history_size) {
	OPENVINO_ASSERT(m_position_ids, "Position ids tensor is not initialized in InputsEmbedderDummyVL.");

	const bool has_template = !m_position_ids.get_shape().empty();
	const ov::Shape shape = has_template ? get_position_ids_shape(m_position_ids, inputs_embeds_size)
	                                    : ov::Shape{1, inputs_embeds_size};
	ov::Tensor position_ids{ov::element::i64, shape};
	fill_position_ids(position_ids, history_size);
	return {position_ids, std::nullopt};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderDummyVL::get_generation_phase_position_ids(
	const size_t inputs_embeds_size,
	const size_t history_size,
	int64_t rope_delta) {
	(void)rope_delta;
	return get_position_ids(inputs_embeds_size, history_size);
}

void InputsEmbedderDummyVL::start_chat(const std::string& system_message) {
	(void)system_message;
}

std::string InputsEmbedderDummyVL::get_last_pruned_prompt(const std::string& original_prompt) const {
	return original_prompt;
}

void InputsEmbedderDummyVL::finish_chat() { }

NormalizedPrompt InputsEmbedderDummyVL::normalize_prompt(
	const std::string& prompt,
	size_t image_base_id,
	size_t video_base_id,
	const std::vector<EncodedImage>& images,
	const std::vector<EncodedVideo>& videos) const {
	(void)image_base_id;
	(void)video_base_id;
	(void)images;
	(void)videos;
	return {prompt, {}, {}};
}

void InputsEmbedderDummyVL::expand_video_tags_in_prompt(
	std::string& unified_prompt,
	const std::vector<EncodedVideo>& encoded_videos,
	const std::vector<size_t>& videos_sequence,
	size_t video_base_id) const {
	(void)unified_prompt;
	(void)encoded_videos;
	(void)videos_sequence;
	(void)video_base_id;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderDummyVL::run_video_image_embeddings_merger(
	const std::vector<EncodedImage>& images,
	const std::vector<size_t>& images_sequence,
	const std::vector<EncodedVideo>& videos,
	const std::vector<size_t>& videos_sequence) {
	(void)images;
	(void)images_sequence;
	(void)videos;
	(void)videos_sequence;
	return {ov::Tensor{}, ov::Tensor{}};
}

ov::Tensor InputsEmbedderDummyVL::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const {
	(void)grids_thw;
	return {};
}

std::vector<std::array<size_t, 3>> InputsEmbedderDummyVL::get_vision_grid_thw_for_position_ids(
	const std::vector<std::array<size_t, 3>>& images_grid_thw,
	const std::vector<size_t>& images_sequence,
	const size_t image_id,
	const std::vector<std::array<size_t, 3>>& videos_grid_thw,
	const std::vector<size_t>& videos_sequence,
	const size_t video_id,
	const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) const {
	(void)images_grid_thw;
	(void)images_sequence;
	(void)image_id;
	(void)videos_grid_thw;
	(void)videos_sequence;
	(void)video_id;
	(void)history_vision_count;
	return {};
}

} // namespace ov::genai
