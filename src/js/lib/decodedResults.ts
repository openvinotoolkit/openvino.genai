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
