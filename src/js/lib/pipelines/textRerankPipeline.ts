// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import {
  TextRerankPipelineConfig,
  TextRerankResults,
  TextRerankPipeline as TextRerankPipelineWrapper,
} from "../addon.js";

/**
 * Options for initializing TextRerankPipeline.
 */
export type TextRerankPipelineOptions = {
  /** Device to run the model on (e.g., "CPU", "GPU"). */
  device?: string;
  /** Optional pipeline configuration. */
  config?: TextRerankPipelineConfig;
  /** Plugin properties. */
  ovProperties?: object;
};

/**
 * Text rerank pipeline for reranking documents based on query relevance.
 */
export class TextRerankPipeline {
  modelPath: string;
  device: string;
  config: TextRerankPipelineConfig;
  ovProperties: object;
  pipeline: TextRerankPipelineWrapper | null = null;

  /**
   * Constructs a TextRerankPipeline from xml/bin files, tokenizer and configuration in the same directory.
   * @param modelPath - Path to the directory containing model xml/bin files and tokenizer.
   * @param options - Pipeline initialization options.
   */
  constructor(modelPath: string, options: TextRerankPipelineOptions) {
    this.modelPath = modelPath;
    this.device = options.device || "CPU";
    this.config = options.config || {};
    this.ovProperties = options.ovProperties || {};
  }

  /**
   * Initializes the pipeline.
   * @throws {Error} If the pipeline is already initialized.
   */
  async init() {
    if (this.pipeline) throw new Error("TextRerankPipeline is already initialized");

    const pipeline = new TextRerankPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.config, this.ovProperties);
    this.pipeline = pipeline;
  }

  /**
   * Reranks a vector of documents based on the query.
   * @param query - The query string.
   * @param documents - Array of document strings to rerank.
   * @returns A promise that resolves to an array of tuples containing document indices and their relevance scores,
   *          sorted by score in descending order.
   * @throws {Error} If the pipeline is not initialized.
   */
  async rerank(query: string, documents: string[]): Promise<TextRerankResults> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const rerank = util.promisify(this.pipeline.rerank.bind(this.pipeline));
    const result = await rerank(query, documents);

    return result;
  }
}
