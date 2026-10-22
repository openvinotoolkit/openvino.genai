// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { ChatHistory, LLMPipeline as LLMPipelineWrapper } from "../addon.js";
import { GenerationConfig, TextStreamer, LLMPipelineProperties } from "../utils.js";
import { DecodedResults } from "../decodedResults.js";
import { Tokenizer } from "../tokenizer.js";
import { PipelineInitialization, createTextStream } from "./pipelineUtils.js";

/**
 * This class is used for generation with Large Language Models (LLMs)
 */
export class LLMPipeline {
  modelPath: string;
  device: string;
  pipeline: LLMPipelineWrapper | null = null;
  properties: LLMPipelineProperties;
  private readonly initialization = new PipelineInitialization("LLMPipeline");

  /**
   * Construct an LLM pipeline from a folder containing tokenizer and model IRs.
   * @param modelPath - A folder to read tokenizer and model IRs.
   * @param device - Inference device. A tokenizer is always compiled for CPU.
   * @param properties - Device and pipeline properties.
   */
  constructor(modelPath: string, device: string, properties: LLMPipelineProperties) {
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
      const pipeline = new LLMPipelineWrapper();
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
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

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
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    return result;
  }

  /**
   * Get the current generation config (model defaults).
   * @returns The current GenerationConfig object.
   */
  getGenerationConfig(): GenerationConfig {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Set the generation parameters.
   * @param config - Generation configuration parameters.
   */
  setGenerationConfig(config: GenerationConfig): void {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }

  /**
   * Stream generation results as an async iterator of strings.
   * The iterator yields subword chunks during generation.
   * When generation finishes, the full decoded text is returned as the final
   * iterator value (`done: true`). This value is not available through
   * `for await...of`; call `next()` directly to read it.
   *
   * For batch processing or custom streaming control, see {@link generate}.
   *
   * @param inputs - Input prompt string or chat history.
   * @param generationConfig - Generation configuration parameters.
   * @returns Async iterator producing subword chunks.
   *
   * @example
   * // Stream subword chunks to console
   * for await (const chunk of pipe.stream(prompt, { max_new_tokens: 100 })) {
   *   process.stdout.write(chunk);
   * }
   *
   * @throws {Error} If inputs is an array - use {@link generate} for batch processing
   */
  stream(inputs: string | ChatHistory, generationConfig: GenerationConfig = {}) {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    if (Array.isArray(inputs))
      throw new Error(
        "Streaming is not supported for array of inputs. Please use LLMPipeline.generate() method.",
      );
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");

    const { pipeline } = this;
    return createTextStream((streamer, complete) => {
      pipeline.generate(inputs, generationConfig, streamer, (error, result) =>
        complete(error, () =>
          new DecodedResults(
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
   * Generate text sequences with optional streaming.
   *
   * This method supports:
   * - Single prompt generation
   * - Batch generation (array of prompts)
   * - Chat history-based generation
   * - Optional custom streaming via callback
   *
   * For simple streaming use cases, consider using {@link stream}, which provides
   * a convenient async iterator interface.
   *
   * @param inputs - Input prompt string, array of prompts, or chat history.
   * @param generationConfig - Generation configuration parameters.
   * @param streamer - Optional callback invoked for each generated text chunk.
   * - Return a `StreamingStatus` flag to indicate whether generation should be stopped or cancelled
   * @returns Resolves with decoded results once generation finishes.
   *
   * @example
   * // Simple generation without streaming
   * const result = await pipe.generate("Hello", { max_new_tokens: 50 });
   * console.log(result.texts[0]);
   *
   * @example
   * // With custom streamer
   * const result = await pipe.generate(prompt, config, (chunk) => {
   *   process.stdout.write(chunk);
   *   return StreamingStatus.RUNNING;
   * });
   */
  async generate(
    inputs: string | string[] | ChatHistory,
    generationConfig: GenerationConfig = {},
    streamer?: TextStreamer,
  ): Promise<DecodedResults> {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");
    if (streamer !== undefined && typeof streamer !== "function")
      throw new Error("Streamer must be a function");

    const innerGenerate = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const result = await innerGenerate(inputs, generationConfig, streamer);

    return new DecodedResults(
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
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    return this.pipeline.getTokenizer();
  }
}
