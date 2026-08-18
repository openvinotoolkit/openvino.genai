// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Tensor } from "openvino-node";

import {
  PerfMetrics,
  VLMPerfMetrics,
  WhisperPerfMetrics,
  Text2SpeechPerfMetrics,
} from "./perfMetrics.js";
import { GenerationFinishReason } from "./utils.js";

/**
 * Structure to store resulting batched text outputs and scores for each batch.
 * @note The first num_return_sequences elements correspond to the first batch element.
 */
export class DecodedResults {
  /**
   * @param {string[]} texts - Vector of resulting sequences.
   * @param {number[]} scores - Scores for each sequence.
   * @param {PerfMetrics} perfMetrics - Performance metrics (tpot, ttft, etc.).
   * @param {Record<string, unknown>[]} parsed - The results of parsers processing for each sequence.
   * @param {GenerationFinishReason[]} finishReasons - Finish reasons for each sequence.
   */
  constructor(
    texts: string[],
    scores: number[],
    perfMetrics: PerfMetrics,
    parsed: Record<string, unknown>[],
    finishReasons: GenerationFinishReason[] = [],
  ) {
    this.texts = texts;
    this.scores = scores;
    this.perfMetrics = perfMetrics;
    this.parsed = parsed;
    this.finishReasons = finishReasons;
  }
  toString() {
    if (this.scores.length !== this.texts.length) {
      throw new Error("The number of scores and texts doesn't match in DecodedResults.");
    }
    if (this.texts.length === 0) {
      return "";
    }
    if (this.texts.length === 1) {
      return this.texts[0];
    }
    const lines = this.scores.map((score, i) => `${score.toFixed(6)}: ${this.texts[i]}`);
    return lines.join("\n");
  }
  texts: string[];
  scores: number[];
  perfMetrics: PerfMetrics;
  parsed: Record<string, unknown>[];
  finishReasons: GenerationFinishReason[];
}

/**
 * Structure to store VLM resulting batched text outputs and scores for each batch.
 * @note The first num_return_sequences elements correspond to the first batch element.
 */
export class VLMDecodedResults extends DecodedResults {
  /**
   * @param {string[]} texts - Vector of resulting sequences.
   * @param {number[]} scores - Scores for each sequence.
   * @param {VLMPerfMetrics} perfMetrics - VLM-specific performance metrics.
   * @param {Record<string, unknown>[]} parsed - The results of parsers processing for each sequence.
   * @param {GenerationFinishReason[]} finishReasons - Finish reasons for each sequence.
   */
  constructor(
    texts: string[],
    scores: number[],
    perfMetrics: VLMPerfMetrics,
    parsed: Record<string, unknown>[],
    finishReasons: GenerationFinishReason[] = [],
  ) {
    super(texts, scores, perfMetrics, parsed, finishReasons);
    this.perfMetrics = perfMetrics;
  }

  /** VLM specific performance metrics. */
  perfMetrics: VLMPerfMetrics;
}

/**
 * Talker-side performance metrics for Omni speech generation.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type OmniSpeechPerfMetrics = {
  /** Number of audio samples generated (waveform length in samples). */
  numGeneratedSamples: number;
  /** Total speech generation time in milliseconds. */
  generationTimeMs: number;
};

/**
 * Speech output of OmniPipeline.generate(): waveforms plus talker perf metrics.
 * `waveforms` is empty when `talkerSpeechConfig.return_audio` was false.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type OmniSpeechResult = {
  /**
   * Speech output waveforms, each `[1, 1, N_samples]` f32 PCM at 24 kHz.
   * Empty when speech generation was not requested.
   */
  waveforms: Tensor[];
  /** Talker-side performance metrics. */
  perfMetrics: OmniSpeechPerfMetrics;
};

/**
 * Result of OmniPipeline.generate(): VLM text results plus a speech result.
 * Extends {@link VLMDecodedResults} with a `speechResult` carrying waveforms and perf metrics.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export class OmniDecodedResults extends VLMDecodedResults {
  /**
   * @param {string[]} texts - Vector of resulting sequences.
   * @param {number[]} scores - Scores for each sequence.
   * @param {VLMPerfMetrics} perfMetrics - VLM-specific performance metrics.
   * @param {Record<string, unknown>[]} parsed - The results of parsers processing for each sequence.
   * @param {GenerationFinishReason[]} finishReasons - Finish reasons for each sequence.
   * @param {OmniSpeechResult} speechResult - Speech waveforms and talker perf metrics.
   */
  constructor(
    texts: string[],
    scores: number[],
    perfMetrics: VLMPerfMetrics,
    parsed: Record<string, unknown>[],
    finishReasons: GenerationFinishReason[],
    speechResult: OmniSpeechResult,
  ) {
    super(texts, scores, perfMetrics, parsed, finishReasons);
    this.speechResult = speechResult;
  }

  /** Speech waveforms and talker performance metrics. */
  speechResult: OmniSpeechResult;
}

/** Whisper decoded result chunk (when return_timestamps or word_timestamps is enabled). */
export type WhisperDecodedResultChunk = {
  text: string;
  startTs: number;
  endTs: number;
};

/** Word-level timing (when word_timestamps is enabled). */
export type WhisperWordTiming = {
  word: string;
  startTs: number;
  endTs: number;
  /** Word token identifiers as `BigInt64Array`. */
  tokenIds?: BigInt64Array;
};

/**
 * Result of WhisperPipeline.generate() with texts, scores, perf metrics, and optional timestamps.
 */
export class WhisperDecodedResults extends DecodedResults {
  constructor(
    texts: string[],
    scores: number[],
    perfMetrics: WhisperPerfMetrics,
    public chunks?: WhisperDecodedResultChunk[],
    public words?: WhisperWordTiming[],
  ) {
    super(texts, scores, perfMetrics, []);
    this.perfMetrics = perfMetrics;
  }

  /** Whisper-specific performance metrics. */
  override perfMetrics: WhisperPerfMetrics;
}

/**
 * Result of Text2SpeechPipeline.generate() with audio tensors and perf metrics.
 * Each element in `speeches` is an audio waveform tensor sampled at 16 kHz.
 */
export class Text2SpeechDecodedResults {
  constructor(speeches: Tensor[], perfMetrics: Text2SpeechPerfMetrics) {
    this.speeches = speeches;
    this.perfMetrics = perfMetrics;
  }

  speeches: Tensor[];
  perfMetrics: Text2SpeechPerfMetrics;
}
