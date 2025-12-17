// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PerfMetrics, VLMPerfMetrics } from "./perfMetrics.js";

/**
 * Structure to store resulting batched text outputs and scores for each batch.
 * @note The first num_return_sequences elements correspond to the first batch element.
 */
export class DecodedResults {
  /**
   * @param {string[]} texts - Vector of resulting sequences.
   * @param {number[]} scores - Scores for each sequence.
   * @param {PerfMetrics} perfMetrics - Performance metrics (tpot, ttft, etc.).
   */
  constructor(texts: string[], scores: number[], perfMetrics: PerfMetrics) {
    this.texts = texts;
    this.scores = scores;
    this.perfMetrics = perfMetrics;
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
    return lines.join('\n');
  }
  texts: string[];
  scores: number[];
  perfMetrics: PerfMetrics;
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
   */
  constructor(texts: string[], scores: number[], perfMetrics: VLMPerfMetrics) {
    super(texts, scores, perfMetrics);
    this.perfMetrics = perfMetrics;
  }

  /** VLM specific performance metrics. */
  perfMetrics: VLMPerfMetrics;
}
