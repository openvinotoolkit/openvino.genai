// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Tensor } from "openvino-node";
import { Text2VideoPipeline as Text2VideoPipelineWrapper } from "../addon.js";
import {
  ImageGenerationCallback,
  Text2VideoPipelineProperties,
  VideoGenerationConfig,
  VideoGenerationResult,
} from "../utils.js";

export type Text2VideoGenerateOptions = VideoGenerationConfig & {
  /**
   * Callback invoked after each denoising step with the current latent.
   * Return `true` to stop early, `false` to continue.
   */
  callback?: ImageGenerationCallback;
};

/**
 * Pipeline for text-to-video generation.
 *
 * Generates video tensor(s) for a single prompt.
 * Use the factory method Text2VideoPipeline() to create and
 * initialize an instance.
 */
export class Text2VideoPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: Text2VideoPipelineWrapper | null = null;
  protected readonly properties: Text2VideoPipelineProperties;

  constructor(modelPath: string, device: string, properties: Text2VideoPipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    const pipeline = new Text2VideoPipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Generate video(s) for a single prompt.
   * The options override the pipeline generation config for this call only.
   * Batch generation is supported through `num_videos_per_prompt`.
   * The returned `video` tensor has shape `[num_videos, num_frames, H, W, 3]`.
   */
  async generate(
    prompt: string,
    options: Text2VideoGenerateOptions = {},
  ): Promise<VideoGenerationResult> {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");

    const { callback, ...generationConfig } = options;

    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    return await generatePromise(prompt, generationConfig, callback);
  }

  /**
   * Decode a latent video into RGB frames using the VAE decoder.
   *
   * Useful inside a generation `callback` to obtain the intermediate video at a denoising step.
   * The returned `video` tensor has shape `[num_videos, num_frames, H, W, 3]` with `u8` pixels.
   */
  async decode(latent: Tensor): Promise<VideoGenerationResult> {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");
    const decodePromise = util.promisify(this.pipeline.decode.bind(this.pipeline));
    return await decodePromise(latent);
  }

  /**
   * Get current video generation config.
   */
  getGenerationConfig(): VideoGenerationConfig {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update video generation config.
   * Fields that are not passed keep their current values.
   * Throws while a generate() call is in progress.
   */
  setGenerationConfig(config: VideoGenerationConfig): void {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
