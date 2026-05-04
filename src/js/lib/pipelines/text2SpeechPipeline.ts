// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Tensor } from "openvino-node";
import { Text2SpeechPipeline as Text2SpeechPipelineWrapper } from "../addon.js";
import { Text2SpeechDecodedResults } from "../decodedResults.js";
import { SpeechGenerationConfig, Text2SpeechPipelineProperties } from "../utils.js";

/**
 * Options for Text2Speech generation methods.
 */
export type Text2SpeechGenerateOptions = {
  /** Optional speaker embedding tensor to condition voice characteristics. */
  speakerEmbedding?: Tensor;
  /** Additional generation properties passed to the native pipeline. */
  generationConfig?: SpeechGenerationConfig;
};

/**
 * Pipeline for text-to-speech synthesis.
 *
 * Converts text input into audio waveform tensors.
 * Use the factory method `PipelineFactory.Text2SpeechPipeline()` to create and
 * initialize an instance.
 */
export class Text2SpeechPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: Text2SpeechPipelineWrapper | null = null;
  protected readonly properties: Text2SpeechPipelineProperties;

  /**
   * Construct a Text2Speech pipeline from a folder containing model IRs.
   * @param modelPath - Path to the folder with model IRs.
   * @param device - Inference device (e.g. "CPU", "GPU").
   * @param properties - Device and pipeline properties.
   */
  constructor(modelPath: string, device: string, properties: Text2SpeechPipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    const pipeline = new Text2SpeechPipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Generate audio from a single text or batch of texts.
   * @param input - Text string or array of text strings to synthesize.
   * @param options - Optional speaker embedding and generation config.
   * @returns Decoded results with audio waveform tensors and perf metrics.
   */
  async generate(
    input: string | string[],
    options: Text2SpeechGenerateOptions = {},
  ): Promise<Text2SpeechDecodedResults> {
    if (!this.pipeline) throw new Error("Text2SpeechPipeline is not initialized");

    const { speakerEmbedding, generationConfig } = options;
    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const res = await generatePromise(input, speakerEmbedding, generationConfig ?? {});

    return new Text2SpeechDecodedResults(res.speeches, res.perfMetrics);
  }

  /**
   * Get current speech generation config.
   */
  getGenerationConfig(): SpeechGenerationConfig {
    if (!this.pipeline) throw new Error("Text2SpeechPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update speech generation config.
   */
  setGenerationConfig(config: SpeechGenerationConfig): void {
    if (!this.pipeline) throw new Error("Text2SpeechPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
