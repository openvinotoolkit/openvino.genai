// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import {
  Text2VideoPipeline as Text2VideoPipelineWrapper,
  VideoGenerationPerfMetrics,
  Text2VideoResult,
} from "../addon.js";
import type { Tensor } from "openvino-node";

/**
 * Video generation configuration parameters for Text2VideoPipeline.
 */
export type VideoGenerationConfig = {
  /** Negative prompt to guide generation away from undesired content. */
  negative_prompt?: string;
  /** Height of generated video frames in pixels. */
  height?: number;
  /** Width of generated video frames in pixels. */
  width?: number;
  /** Number of video frames to generate. */
  num_frames?: number;
  /** Number of denoising inference steps. */
  num_inference_steps?: number;
  /** Classifier-free guidance scale. Higher values produce outputs more aligned with the prompt. */
  guidance_scale?: number;
  /** Video frame rate. Affects rope interpolation scale. */
  frame_rate?: number;
  /** Number of videos to generate per call. */
  num_videos_per_prompt?: number;
  /** Maximum sequence length for T5 encoder / tokenizer. */
  max_sequence_length?: number;
  /** Guidance rescale factor. */
  guidance_rescale?: number;
};

/**
 * Options for generate() method.
 */
export type Text2VideoGenerateOptions = VideoGenerationConfig & {
  /** Callback invoked at each denoising step with (step, numSteps, latent). Return true to cancel. */
  callback?: (step: number, numSteps: number, latent: Tensor) => boolean;
};

/**
 * Pipeline for generating video from text prompts using OpenVINO GenAI.
 */
export class Text2VideoPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: Text2VideoPipelineWrapper | null = null;
  protected readonly properties: Record<string, unknown>;

  /**
   * Construct a Text2Video pipeline.
   * @param modelPath - Path to a folder containing model files.
   * @param device - Inference device (e.g. "CPU", "GPU").
   * @param properties - Additional device/pipeline properties.
   */
  constructor(
    modelPath: string,
    device: string,
    properties: Record<string, unknown> = {},
  ) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Initialize the underlying native pipeline.
   * @returns Resolves when initialization is complete.
   */
  async init(): Promise<void> {
    const pipeline = new Text2VideoPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);

    this.pipeline = pipeline;
  }

  /**
   * Generate video from a text prompt.
   * @param prompt - The text prompt to generate video from.
   * @param options - Optional generation parameters.
   * @returns Resolves with the generated video tensor and performance metrics.
   */
  async generate(
    prompt: string,
    options: Text2VideoGenerateOptions = {},
  ): Promise<Text2VideoResult> {
    if (!this.pipeline)
      throw new Error("Text2VideoPipeline is not initialized");

    const properties: Record<string, unknown> = {};

    if (options.negative_prompt !== undefined)
      properties["negative_prompt"] = options.negative_prompt;
    if (options.height !== undefined) properties["height"] = options.height;
    if (options.width !== undefined) properties["width"] = options.width;
    if (options.num_frames !== undefined)
      properties["num_frames"] = options.num_frames;
    if (options.num_inference_steps !== undefined)
      properties["num_inference_steps"] = options.num_inference_steps;
    if (options.guidance_scale !== undefined)
      properties["guidance_scale"] = options.guidance_scale;
    if (options.frame_rate !== undefined)
      properties["frame_rate"] = options.frame_rate;
    if (options.num_videos_per_prompt !== undefined)
      properties["num_videos_per_prompt"] = options.num_videos_per_prompt;
    if (options.max_sequence_length !== undefined)
      properties["max_sequence_length"] = options.max_sequence_length;
    if (options.guidance_rescale !== undefined)
      properties["guidance_rescale"] = options.guidance_rescale;
    if (options.callback !== undefined)
      properties["callback"] = options.callback;

    const innerGenerate = util.promisify(
      this.pipeline.generate.bind(this.pipeline),
    );
    return await innerGenerate(prompt, properties);
  }

  /**
   * Get the current video generation configuration.
   * @returns The current generation config.
   */
  getGenerationConfig(): VideoGenerationConfig {
    if (!this.pipeline)
      throw new Error("Text2VideoPipeline is not initialized");
    return this.pipeline.getGenerationConfig() as VideoGenerationConfig;
  }

  /**
   * Set video generation configuration.
   * @param config - Generation configuration parameters to set.
   */
  setGenerationConfig(config: VideoGenerationConfig): void {
    if (!this.pipeline)
      throw new Error("Text2VideoPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}

export type { VideoGenerationPerfMetrics, Text2VideoResult };
