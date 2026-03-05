// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import {
  Text2VideoPipeline as Text2VideoPipelineWrapper,
  VideoGenerationPerfMetrics,
  Text2VideoResult,
  VideoGenerationConfig,
  Text2VideoGenerateOptions,
} from "../addon.js";

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
  constructor(modelPath: string, device: string, properties: Record<string, unknown> = {}) {
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
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");

    const { callback: _callback, ...generationConfig } = options;
    const innerGenerate = util.promisify(this.pipeline.generate.bind(this.pipeline));
    return await innerGenerate(prompt, generationConfig);
  }

  /**
   * Get the current video generation configuration.
   * @returns The current generation config.
   */
  getGenerationConfig(): VideoGenerationConfig {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Set video generation configuration.
   * @param config - Generation configuration parameters to set.
   */
  setGenerationConfig(config: VideoGenerationConfig): void {
    if (!this.pipeline) throw new Error("Text2VideoPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}

export type {
  VideoGenerationConfig,
  Text2VideoGenerateOptions,
  VideoGenerationPerfMetrics,
  Text2VideoResult,
};
