// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { WhisperPipeline as WhisperPipelineWrapper } from "../addon.js";
import { WhisperDecodedResults } from "../decodedResults.js";
import { WhisperGenerationConfig, WhisperPipelineProperties, TextStreamer } from "../utils.js";
import { Tokenizer } from "../tokenizer.js";
import { PipelineInitialization, createTextStream } from "./pipelineUtils.js";

export type RawSpeechInput = Float32Array | number[];

/**
 * Options for Whisper generation methods.
 */
export type WhisperGenerateOptions = {
  /** Generation configuration (e.g. language, task, return_timestamps). */
  generationConfig?: WhisperGenerationConfig;
  /** Callback invoked with each decoded text chunk; return StreamingStatus to control generation. */
  streamer?: TextStreamer;
};

/**
 * Pipeline for automatic speech recognition using Whisper models.
 *
 * Expects raw audio normalized to approximately [-1, 1] at 16 kHz sample rate.
 * Use a WAV file or decode audio to Float32Array before calling generate().
 */
export class WhisperPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: WhisperPipelineWrapper | null = null;
  protected readonly properties: WhisperPipelineProperties;
  private readonly initialization = new PipelineInitialization("WhisperPipeline");

  /**
   * Construct a Whisper pipeline from a folder containing model IRs and tokenizer.
   * @param modelPath - Path to the folder with model IRs and tokenizer (e.g. openvino_encoder_model.xml, preprocessor_config.json).
   * @param device - Inference device (e.g. "CPU", "GPU").
   * @param properties - Device and pipeline properties (e.g. word_timestamps: true, CACHE_DIR: "cache").
   */
  constructor(modelPath: string, device: string, properties: WhisperPipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Load the pipeline. Must be called once before generate().
   */
  async init(): Promise<void> {
    this.pipeline = await this.initialization.run(async () => {
      const pipeline = new WhisperPipelineWrapper();
      const initialize = util.promisify(pipeline.init.bind(pipeline));
      await initialize(this.modelPath, this.device, this.properties);
      return pipeline;
    });
  }

  /**
   * Stream speech recognition results as an async iterator.
   * The iterator yields decoded text chunks during generation.
   * When generation finishes, the full decoded text is returned as the final
   * iterator value (`done: true`). This value is not available through
   * `for await...of`; call `next()` directly to read it.
   *
   * For custom streaming control, use {@link generate} with a streamer callback instead.
   *
   * @param rawSpeech - Audio samples as Float32Array or number[], normalized to ~[-1, 1], 16 kHz.
   * @param options - Optional generation config (e.g. language, task, return_timestamps).
   * @returns Async iterator that yields decoded text chunks as strings.
   */
  stream(
    rawSpeech: RawSpeechInput,
    options?: WhisperGenerateOptions,
  ): AsyncIterableIterator<string> {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    const generationConfig = options?.generationConfig ?? {};

    const { pipeline } = this;
    return createTextStream((streamer, complete) => {
      pipeline.generate(rawSpeech, generationConfig, streamer, (error, result) =>
        complete(error, () => result.texts[0] ?? ""),
      );
    });
  }

  /**
   * Run speech recognition with optional streaming.
   *
   * For simple streaming use cases, consider using {@link stream}, which provides
   * a convenient async iterator interface.
   *
   * @param rawSpeech - Audio samples as Float32Array or number[], normalized to ~[-1, 1], 16 kHz.
   * @param options - Optional parameters.
   * @param options.generationConfig - Generation config (e.g., language, task, return_timestamps).
   * @param options.streamer - Optional callback invoked for each decoded chunk.
   * - Return a `StreamingStatus` flag to indicate whether generation should be stopped or cancelled
   * @returns Decoded texts, scores, optional chunks with timestamps, and perf metrics.
   */
  async generate(
    rawSpeech: RawSpeechInput,
    options: WhisperGenerateOptions = {},
  ): Promise<WhisperDecodedResults> {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    const { generationConfig, streamer } = options;
    const generatePromise = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const res = await generatePromise(rawSpeech, generationConfig ?? {}, streamer);
    return new WhisperDecodedResults(res.texts, res.scores, res.perfMetrics, res.chunks, res.words);
  }

  /**
   * Get the pipeline tokenizer.
   */
  getTokenizer(): Tokenizer {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    return this.pipeline.getTokenizer();
  }

  /**
   * Get current generation config (language, task, return_timestamps, etc.).
   */
  getGenerationConfig(): Partial<WhisperGenerationConfig> {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Update generation config fields (e.g. language, task, return_timestamps).
   */
  setGenerationConfig(config: WhisperGenerationConfig): void {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
