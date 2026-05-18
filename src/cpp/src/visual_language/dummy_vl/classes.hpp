// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// For dummy VL, VisionEncoder and InputsEmbedder are implemented by customer self.
// The dummy VL classes only handle generic config and generation-time metadata updates in ContinuousBatching,
// and intentionally do not implement model-specific embedding behavior.
#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "circular_buffer_queue.hpp"
#include "visual_language/cdpruner/cdpruner.hpp"

namespace ov::genai {

class VisionEncoderDummyVL : public VisionEncoder {
public:
	explicit VisionEncoderDummyVL(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);
	explicit VisionEncoderDummyVL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path, const std::string& device, const ov::AnyMap properties);

	EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
	EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

protected:
	/**
	 * @brief Encodes video frames by grouping them into chunks of config.temporal_patch_size adjacent frames
	 * and saves results into the encoded_video struct.
	 * The config can be ProcessorConfig or a derived class (e.g. VideoProcessorConfig for Qwen3-VL).
	 */
	void encode_frames_with_config(EncodedVideo& encoded_video, const std::vector<ov::Tensor>& frames, const ProcessorConfig& config);

private:
	using EncodeFunc = std::function<void(const std::vector<ov::Tensor>&, const ProcessorConfig&, ov::Tensor&, ImageSize&, size_t, size_t)>;

	EncodeFunc get_encode_func();

	void encode_with_imagepreprocess_cpp(
		const std::vector<ov::Tensor>& image,
		const ProcessorConfig& config,
		ov::Tensor& out_tensor,
		ImageSize& out_rsz_size,
		size_t frame_num = 1,
		size_t frame_id = 0);

	void encode_with_imagepreprocess_ov(
		const std::vector<ov::Tensor>& image,
		const ProcessorConfig& config,
		ov::Tensor& out_tensor,
		ImageSize& out_rsz_size,
		size_t frame_num = 1,
		size_t frame_id = 0);

	bool use_ov_vision_preprocess = true; // default use ov vision preprocess, control by env VISION_PREPROCESS=CPP to use cpp vision preprocess
};

class InputsEmbedderDummyVL : public InputsEmbedder::IInputsEmbedder {
public:
	InputsEmbedderDummyVL(
		const VLMConfig& vlm_config,
		const std::filesystem::path& model_dir,
		const std::string& device,
		const ov::AnyMap device_config);

	InputsEmbedderDummyVL(
		const VLMConfig& vlm_config,
		const ModelsMap& models_map,
		const Tokenizer& tokenizer, 
		const std::filesystem::path& config_dir_path,
		const std::string& device,
		const ov::AnyMap device_config);

	ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;
	ov::Tensor get_inputs_embeds(const std::string& prompt,
								 const std::vector<ov::genai::EncodedImage>& images,
								 const std::vector<ov::genai::EncodedVideo>& videos,
								 ov::genai::VLMPerfMetrics& metrics,
								 bool recalculate_merged_embeddings = true,
								 const std::vector<size_t>& image_sequence = {},
								 const std::vector<size_t>& videos_sequence = {},
								 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

	std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

	std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos, const std::vector<VideoMetadata>& videos_metadata = {}) override;

	std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

	std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) override;

	void start_chat(const std::string& system_message) override;

	std::string get_last_pruned_prompt(const std::string& original_prompt) const override;

	void finish_chat() override;

	NormalizedPrompt normalize_prompt(
		const std::string& prompt,
		size_t base_id,
		const std::vector<EncodedImage>& images) const override {
		auto norm_prompt = normalize_prompt(prompt, base_id, 0, images, {});
		return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
	}

	NormalizedPrompt normalize_prompt(
		const std::string& prompt,
		size_t image_base_id,
		size_t video_base_id,
		const std::vector<EncodedImage>& images,
		const std::vector<EncodedVideo>& videos) const override;

protected:
	virtual void expand_video_tags_in_prompt(
		std::string& unified_prompt,
		const std::vector<EncodedVideo>& encoded_videos,
		const std::vector<size_t>& videos_sequence,
		size_t video_base_id
	) const;

	virtual std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
		const std::vector<EncodedImage>& images, 
		const std::vector<size_t>& images_sequence,
		const std::vector<EncodedVideo>& videos,
		const std::vector<size_t>& videos_sequence);

	virtual ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const;

	virtual std::vector<std::array<size_t, 3>> get_vision_grid_thw_for_position_ids(
		const std::vector<std::array<size_t, 3>>& images_grid_thw,
		const std::vector<size_t>& images_sequence,
		const size_t image_id,
		const std::vector<std::array<size_t, 3>>& videos_grid_thw,
		const std::vector<size_t>& videos_sequence,
		const size_t video_id,
		const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
	) const;
};

} // namespace ov::genai
