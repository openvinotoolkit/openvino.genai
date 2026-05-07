// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Tensor } from "openvino-node";
import { Text2ImagePipeline as Text2ImagePipelineWrapper } from "../addon.js";
import { Text2ImagePerfMetrics } from "../perfMetrics.js";
import {
  ImageGenerationConfig,
  Text2ImageCallback,
  Text2ImagePipelineProperties,
} from "../utils.js";

export type Text2ImageGenerateOptions = ImageGenerationConfig & {
  /**
   * Callback invoked after each denoising step.
   * Return `true` to stop early, `false` to continue.
   */
  callback?: Text2ImageCallback;
};

/**
 * Pipeline for text-to-image generation.
 *
 * Generates image tensor(s) for a single prompt.
 * Use the factory method Text2ImagePipeline() to create and
 * initialize an instance.
 */
export class Text2ImagePipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: Text2ImagePipelineWrapper | null = null;
  protected readonly properties: Text2ImagePipelineProperties;

  constructor(modelPath: string, device: string, properties: Text2ImagePipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    const pipeline = new Text2ImagePipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Generate image tensor(s) for a single prompt.
   * Batch image generation is supported through `generationConfig.num_images_per_prompt`.
   * Returned tensor shape is `[N, H, W, 3]` for batched generation and `[1, H, W, 3]` for single image.
   */
  async generate(prompt: string, options: Text2ImageGenerateOptions = {}): Promise<Tensor> {
    if (!this.pipeline) throw new Error("Text2ImagePipeline is not initialized");

    const { callback, ...generationConfig } = options;

    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    return await generatePromise(prompt, generationConfig, callback);
  }

  /**
   * Get performance metrics from the latest generate() call.
   */
  getPerformanceMetrics(): Text2ImagePerfMetrics {
    if (!this.pipeline) throw new Error("Text2ImagePipeline is not initialized");
    return this.pipeline.getPerformanceMetrics();
  }

  /**
   * Get current image generation config.
   */
  getGenerationConfig(): ImageGenerationConfig {
    if (!this.pipeline) throw new Error("Text2ImagePipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update image generation config.
   */
  setGenerationConfig(config: ImageGenerationConfig): void {
    if (!this.pipeline) throw new Error("Text2ImagePipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
