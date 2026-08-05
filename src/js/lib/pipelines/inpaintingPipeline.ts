// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Tensor } from "openvino-node";
import { InpaintingPipeline as InpaintingPipelineWrapper } from "../addon.js";
import { ImageGenerationPerfMetrics } from "../perfMetrics.js";
import {
  ImageGenerationConfig,
  ImageGenerationCallback,
  InpaintingPipelineProperties,
} from "../utils.js";

export type InpaintingGenerateOptions = ImageGenerationConfig & {
  /**
   * Callback invoked after each denoising step.
   * Return `true` to stop early, `false` to continue.
   */
  callback?: ImageGenerationCallback;
};

/**
 * Pipeline for inpainting.
 *
 * Generates image tensor(s) for a single prompt conditioned on an input image
 * and a mask that marks the region to be replaced.
 * Use the factory method InpaintingPipeline() to create and
 * initialize an instance.
 */
export class InpaintingPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: InpaintingPipelineWrapper | null = null;
  protected readonly properties: InpaintingPipelineProperties;

  constructor(modelPath: string, device: string, properties: InpaintingPipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    const pipeline = new InpaintingPipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Generate image tensor(s) for a single prompt conditioned on the input image and mask.
   *
   * Both `image` and `mask` must be `u8` Tensors with batched NHWC shape `[1, H, W, 3]`
   * (single image, RGB channels-last). Unbatched `[H, W, 3]` tensors are rejected.
   * The mask marks the region to be replaced: white pixels are repainted, black pixels are kept.
   *
   * Batch image generation is supported through `generationConfig.num_images_per_prompt`.
   * Returned tensor shape is `[N, H, W, 3]` for batched generation and `[1, H, W, 3]` for single image.
   */
  async generate(
    prompt: string,
    image: Tensor,
    mask: Tensor,
    options: InpaintingGenerateOptions = {},
  ): Promise<Tensor> {
    if (!this.pipeline) throw new Error("InpaintingPipeline is not initialized");

    const { callback, ...generationConfig } = options;

    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    return await generatePromise(prompt, image, mask, generationConfig, callback);
  }

  /**
   * Decode a latent image into an RGB image tensor using the VAE decoder.
   *
   * Useful inside a generation `callback` to obtain the intermediate image at a denoising step.
   * Returned tensor shape is `[N, H, W, 3]` with `u8` pixels.
   */
  async decode(latent: Tensor): Promise<Tensor> {
    if (!this.pipeline) throw new Error("InpaintingPipeline is not initialized");
    const decodePromise = util.promisify(this.pipeline.decode.bind(this.pipeline));
    return await decodePromise(latent);
  }

  /**
   * Get performance metrics from the latest generate() call.
   */
  getPerformanceMetrics(): ImageGenerationPerfMetrics {
    if (!this.pipeline) throw new Error("InpaintingPipeline is not initialized");
    return this.pipeline.getPerformanceMetrics();
  }

  /**
   * Get current image generation config.
   */
  getGenerationConfig(): ImageGenerationConfig {
    if (!this.pipeline) throw new Error("InpaintingPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update image generation config.
   */
  setGenerationConfig(config: ImageGenerationConfig): void {
    if (!this.pipeline) throw new Error("InpaintingPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
