// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Tensor } from "openvino-node";
import { Image2ImagePipeline as Image2ImagePipelineWrapper } from "../addon.js";
import { ImageGenerationPerfMetrics } from "../perfMetrics.js";
import {
  ImageGenerationConfig,
  ImageGenerationCallback,
  Image2ImagePipelineProperties,
} from "../utils.js";

export type Image2ImageGenerateOptions = ImageGenerationConfig & {
  /**
   * Callback invoked after each denoising step.
   * Return `true` to stop early, `false` to continue.
   */
  callback?: ImageGenerationCallback;
};

/**
 * Pipeline for image-to-image generation.
 *
 * Generates image tensor(s) for a single prompt conditioned on an input image.
 * Use the factory method Image2ImagePipeline() to create and
 * initialize an instance.
 */
export class Image2ImagePipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: Image2ImagePipelineWrapper | null = null;
  protected readonly properties: Image2ImagePipelineProperties;

  constructor(modelPath: string, device: string, properties: Image2ImagePipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    const pipeline = new Image2ImagePipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Generate image tensor(s) for a single prompt conditioned on the input image.
   *
   * The input `image` must be a `u8` Tensor with batched NHWC shape `[1, H, W, 3]`
   * (single image, RGB channels-last). Unbatched `[H, W, 3]` tensors are rejected.
   *
   * Batch image generation is supported through `generationConfig.num_images_per_prompt`.
   * Returned tensor shape is `[N, H, W, 3]` for batched generation and `[1, H, W, 3]` for single image.
   */
  async generate(
    prompt: string,
    image: Tensor,
    options: Image2ImageGenerateOptions = {},
  ): Promise<Tensor> {
    if (!this.pipeline) throw new Error("Image2ImagePipeline is not initialized");

    const { callback, ...generationConfig } = options;

    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    return await generatePromise(prompt, image, generationConfig, callback);
  }

  /**
   * Decode a latent image into an RGB image tensor using the VAE decoder.
   *
   * Useful inside a generation `callback` to obtain the intermediate image at a denoising step.
   * Returned tensor shape is `[N, H, W, 3]` with `u8` pixels.
   */
  async decode(latent: Tensor): Promise<Tensor> {
    if (!this.pipeline) throw new Error("Image2ImagePipeline is not initialized");
    const decodePromise = util.promisify(this.pipeline.decode.bind(this.pipeline));
    return await decodePromise(latent);
  }

  /**
   * Get performance metrics from the latest generate() call.
   */
  getPerformanceMetrics(): ImageGenerationPerfMetrics {
    if (!this.pipeline) throw new Error("Image2ImagePipeline is not initialized");
    return this.pipeline.getPerformanceMetrics();
  }

  /**
   * Get current image generation config.
   */
  getGenerationConfig(): ImageGenerationConfig {
    if (!this.pipeline) throw new Error("Image2ImagePipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update image generation config.
   */
  setGenerationConfig(config: ImageGenerationConfig): void {
    if (!this.pipeline) throw new Error("Image2ImagePipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
