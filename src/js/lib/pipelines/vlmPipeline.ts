// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { VLMPipeline as VLMPipelineWrapper, type ChatHistory } from "../addon.js";
import { GenerationConfig, VLMPipelineProperties, TextStreamer } from "../utils.js";
import { VLMDecodedResults } from "../decodedResults.js";
import { Tokenizer } from "../tokenizer.js";
import type { Tensor } from "openvino-node";
import { PipelineInitialization, createTextStream } from "./pipelineUtils.js";

/**
 * Options for VLM generation methods.
 */
export type VLMGenerateOptions = {
  /** Array of image tensors to include in the prompt. */
  images?: Tensor[];
  /** Array of video frame tensors to include in the prompt. */
  videos?: Tensor[];
  /** Generation configuration parameters such as max_length, temperature, etc. */
  generationConfig?: GenerationConfig;
};

/**
 * This class is used for generation with Visual Language Models (VLMs)
 */
export class VLMPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: VLMPipelineWrapper | null = null;
  protected readonly properties: VLMPipelineProperties;
  private readonly initialization = new PipelineInitialization("VLMPipeline");

  /**
   * Construct a VLM pipeline from a folder containing tokenizer and model IRs.
   * @param modelPath - A folder to read tokenizer and model IRs.
   * @param device - Inference device. A tokenizer is always compiled for CPU.
   * @param properties - Device and pipeline properties.
   */
  constructor(modelPath: string, device: string, properties: VLMPipelineProperties) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Initialize the underlying native pipeline.
   * @returns Resolves when initialization is complete.
   */
  async init() {
    this.pipeline = await this.initialization.run(async () => {
      const pipeline = new VLMPipelineWrapper();
      const initialize = util.promisify(pipeline.init.bind(pipeline));
      await initialize(this.modelPath, this.device, this.properties);
      return pipeline;
    });
  }
  /**
   * Start a chat session with an optional system message.
   * @param systemMessage - Optional system message to initialize chat context.
   * @returns Resolves when chat session is started.
   * @deprecated startChat() / finishChat() API is deprecated and will be removed in the next major release.
   * Please, use generate() with ChatHistory argument.
   */
  async startChat(systemMessage: string = "") {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in the next major release.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    const startChatPromise = util.promisify(this.pipeline.startChat.bind(this.pipeline));
    const result = await startChatPromise(systemMessage);

    return result;
  }
  /**
   * Finish the current chat session and clear chat-related state.
   * @returns Resolves when chat session is finished.
   * @deprecated startChat() / finishChat() API is deprecated and will be removed in the next major release.
   * Please, use generate() with ChatHistory argument.
   */
  async finishChat() {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in the next major release.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    return result;
  }
  /**
   * Stream generation results as an async iterator of strings.
   * The iterator yields subword chunks during generation.
   * When generation finishes, the full decoded text is returned as the final
   * iterator value (`done: true`). This value is not available through
   * `for await...of`; call `next()` directly to read it.
   *
   * For custom streaming control, use {@link generate} with a streamer callback instead.
   *
   * @param inputs - Input prompt string or chat history. May contain image/video tags recognized by the model.
   * @param options - Optional parameters.
   * @param options.images - Array of image tensors to include in the prompt.
   * @param options.videos - Array of video frame tensors to include in the prompt.
   * @param options.generationConfig - Generation parameters.
   * @returns Async iterator producing subword chunks.
   */
  stream(
    inputs: string | ChatHistory,
    options: VLMGenerateOptions = {},
  ): AsyncIterableIterator<string> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const { images, videos, generationConfig } = options;

    const { pipeline } = this;
    return createTextStream((streamer, complete) => {
      pipeline.generate(inputs, images, videos, streamer, generationConfig, (error, result) =>
        complete(error, () =>
          new VLMDecodedResults(
            result.texts,
            result.scores,
            result.perfMetrics,
            result.parsed,
            result.finishReasons,
          ).toString(),
        ),
      );
    });
  }
  /**
   * Generate sequences for VLMs with optional streaming.
   *
   * For simple streaming use cases, consider using {@link stream}, which provides
   * a convenient async iterator interface.
   *
   * @param inputs - Input prompt string or chat history. May contain model-specific image/video tags.
   * @param options - Optional parameters.
   * @param options.images - Array of image tensors to include in the prompt.
   * @param options.videos - Array of video frame tensors to include in the prompt.
   * @param options.generationConfig - Generation configuration parameters (e.g., max_new_tokens, temperature).
   * @param options.streamer - Optional callback invoked for each generated subword chunk.
   * - Return a `StreamingStatus` flag to indicate whether generation should be stopped or cancelled
   * @returns Promise resolving to {@link VLMDecodedResults} containing texts, scores, and performance metrics.
   */
  async generate(
    inputs: string | ChatHistory,
    options: VLMGenerateOptions & { streamer?: TextStreamer } = {},
  ): Promise<VLMDecodedResults> {
    const { images, videos, generationConfig, streamer } = options;
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const innerGenerate = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const result = await innerGenerate(inputs, images, videos, streamer, generationConfig);

    return new VLMDecodedResults(
      result.texts,
      result.scores,
      result.perfMetrics,
      result.parsed,
      result.finishReasons,
    );
  }

  /**
   * Get the pipeline tokenizer instance.
   * @returns Tokenizer used by the pipeline.
   */
  getTokenizer(): Tokenizer {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    return this.pipeline.getTokenizer();
  }

  /**
   * Set the chat template used when formatting chat history and prompts.
   * @param chatTemplate - Chat template string.
   */
  setChatTemplate(chatTemplate: string): void {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    this.pipeline.setChatTemplate(chatTemplate);
  }

  /**
   * Set the generation parameters.
   * @param config - Generation configuration parameters.
   */
  setGenerationConfig(config: GenerationConfig): void {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }

  /**
   * Get the current generation config (model defaults).
   * @returns The current GenerationConfig object.
   */
  getGenerationConfig(): GenerationConfig {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }
}
