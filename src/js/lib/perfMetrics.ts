// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/** Structure holding mean and standard deviation values. */
export type MeanStdPair = {
  mean: number;
  std: number;
};

/** Structure holding summary of statistical values */
export type SummaryStats = MeanStdPair & {
  min: number;
  max: number;
};

/** Structure with raw performance metrics for each generation before any statistics are calculated. */
export type RawMetrics = {
  /** Durations for each generate call in milliseconds. */
  generateDurations: number[];
  /** Durations for the tokenization process in milliseconds. */
  tokenizationDurations: number[];
  /** Durations for the detokenization process in milliseconds. */
  detokenizationDurations: number[];
  /** Times to the first token for each call in milliseconds. */
  timesToFirstToken: number[];
  /** Timestamps of generation every token or batch of tokens in milliseconds. */
  newTokenTimes: number[];
  /** Inference time for each token in milliseconds. */
  tokenInferDurations: number[];
  /** Batch sizes for each generate call. */
  batchSizes: number[];
  /** Total durations for each generate call in milliseconds. */
  durations: number[];
  /** Total inference duration for each generate call in microseconds. */
  inferenceDurations: number[];
  /** Time to compile the grammar in milliseconds. */
  grammarCompileTimes: number[];
};

/** Structure with raw performance metrics for VLM generation. */
export type VLMRawMetrics = {
  /** Durations for embedding preparation in milliseconds. */
  prepareEmbeddingsDurations: number[];
};

/**
 * Holds performance metrics for each generate call.
 *
 * PerfMetrics holds the following metrics with mean and standard deviations:
    - Time To the First Token (TTFT), ms
    - Time per Output Token (TPOT), ms/token
    - Inference time per Output Token (IPOT), ms/token
    - Generate total duration, ms
    - Inference duration, ms
    - Tokenization duration, ms
    - Detokenization duration, ms
    - Throughput, tokens/s
    - Load time, ms
    - Number of generated tokens
    - Number of tokens in the input prompt
    - Time to initialize grammar compiler for each backend, ms
    - Time to compile grammar, ms
 * Preferable way to access metrics is via getter methods. Getter methods calculate mean and std values from rawMetrics and return pairs.
 * If mean and std were already calculated, getter methods return cached values.
 */
export interface PerfMetrics {
  /** Returns the load time in milliseconds. */
  getLoadTime(): number;
  /** Returns the number of generated tokens. */
  getNumGeneratedTokens(): number;
  /** Returns the number of tokens in the input prompt. */
  getNumInputTokens(): number;
  /** Returns the mean and standard deviation of Time To the First Token (TTFT) in milliseconds. */
  getTTFT(): MeanStdPair;
  /** Returns the mean and standard deviation of Time Per Output Token (TPOT) in milliseconds. */
  getTPOT(): MeanStdPair;
  /** Returns the mean and standard deviation of Inference time Per Output Token in milliseconds. */
  getIPOT(): MeanStdPair;
  /** Returns the mean and standard deviation of throughput in tokens per second. */
  getThroughput(): MeanStdPair;
  /** Returns the mean and standard deviation of the time spent on model inference during generate call in milliseconds. */
  getInferenceDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of generate durations in milliseconds. */
  getGenerateDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of tokenization durations in milliseconds. */
  getTokenizationDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of detokenization durations in milliseconds. */
  getDetokenizationDuration(): MeanStdPair;
  /** Returns a map with the time to initialize the grammar compiler for each backend in milliseconds. */
  getGrammarCompilerInitTimes(): { [key: string]: number };
  /** Returns the mean, standard deviation, min, and max of grammar compile times in milliseconds. */
  getGrammarCompileTime(): SummaryStats;
  /** A structure of RawPerfMetrics type that holds raw metrics. */
  rawMetrics: RawMetrics;

  /** Adds the metrics from another PerfMetrics object to this one.
   * @returns The current PerfMetrics instance.
   */
  add(other: PerfMetrics): this;
}

/**
 * Holds performance metrics for each VLM generate call.
 *
 * VLMPerfMetrics extends PerfMetrics with VLM-specific metrics:
 *  - Prepare embeddings duration, ms
 */
export interface VLMPerfMetrics extends PerfMetrics {
  /** Returns the mean and standard deviation of embeddings preparation duration in milliseconds. */
  getPrepareEmbeddingsDuration(): MeanStdPair;
  /** VLM specific raw metrics */
  vlmRawMetrics: VLMRawMetrics;

  /** Adds the metrics from another VLMPerfMetrics object to this one.
   * @returns The current VLMPerfMetrics instance.
   */
  add(other: VLMPerfMetrics): this;
}
