// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { WhisperPipeline as WhisperPipelineWrapper } from "../addon.js";
import {
  WhisperDecodedResultChunk,
  WhisperDecodedResults,
  WhisperWordTiming,
} from "../decodedResults.js";
import { WhisperGenerationConfig, WhisperPipelineProperties, StreamingStatus } from "../utils.js";
import { Tokenizer } from "../tokenizer.js";
import { WhisperPerfMetrics } from "../perfMetrics.js";

export type RawSpeechInput = Float32Array | number[];

/**
 * Options for Whisper generation methods.
 */
export type WhisperGenerateOptions = {
  /** Generation configuration (e.g. language, task, return_timestamps). */
  generationConfig?: WhisperGenerationConfig;
  /** Callback invoked with each decoded text chunk; return StreamingStatus to control generation. */
  streamer?: (chunk: string) => StreamingStatus;
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
    const pipeline = new WhisperPipelineWrapper();
    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;
  }

  /**
   * Stream speech recognition results as an async iterator.
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

    let streamingStatus = StreamingStatus.RUNNING;
    let handledError: Error | null;
    const queue: { done: boolean; chunk: string }[] = [];
    type ResolveFunction = (arg: { value: string; done: boolean }) => void;
    let resolvePromise: ResolveFunction | null = null;
    let rejectPromise: ((reason?: unknown) => void) | null = null;

    const callback = (
      error: Error | null,
      result: {
        texts: string[];
        scores: number[];
        perfMetrics: WhisperPerfMetrics;
        chunks?: WhisperDecodedResultChunk[];
        words?: WhisperWordTiming[];
      },
    ) => {
      if (error) {
        if (rejectPromise) {
          rejectPromise(error);
          resolvePromise = null;
          rejectPromise = null;
        } else {
          handledError = error;
        }
      } else {
        const fullText = result.texts?.[0] ?? "";
        if (resolvePromise) {
          resolvePromise({ done: true, value: fullText });
          resolvePromise = null;
          rejectPromise = null;
        } else {
          queue.push({ done: true, chunk: fullText });
        }
      }
    };

    const streamer = (chunk: string): StreamingStatus => {
      if (resolvePromise) {
        resolvePromise({ done: false, value: chunk });
        resolvePromise = null;
        rejectPromise = null;
      } else {
        queue.push({ done: false, chunk });
      }
      return streamingStatus;
    };

    this.pipeline.generate(rawSpeech, generationConfig, streamer, callback);

    return {
      async next() {
        if (handledError) {
          const error = handledError;
          handledError = null;
          return Promise.reject(error);
        }
        const data = queue.shift();
        if (data) {
          return { value: data.chunk, done: data.done };
        }
        return new Promise<IteratorResult<string>>((resolve, reject) => {
          resolvePromise = resolve;
          rejectPromise = reject;
        });
      },
      async return() {
        streamingStatus = StreamingStatus.CANCEL;
        return { done: true, value: "" };
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
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
   * - Return `StreamingStatus.RUNNING` to continue or `StreamingStatus.CANCEL` to stop
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
   * Update generation config (e.g. language, task, return_timestamps).
   */
  setGenerationConfig(config: WhisperGenerationConfig): void {
    if (!this.pipeline) throw new Error("WhisperPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }
}
