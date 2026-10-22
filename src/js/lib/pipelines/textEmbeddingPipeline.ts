// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import {
  TextEmbeddingPipelineWrapper,
  EmbeddingResult,
  EmbeddingResults,
  TextEmbeddingConfig,
  TextEmbeddingPipeline as TextEmbeddingPipelineWrap,
} from "../addon.js";
import { PipelineInitialization } from "./pipelineUtils.js";

export class TextEmbeddingPipeline {
  modelPath: string;
  device: string;
  config: TextEmbeddingConfig;
  ovProperties: object;
  pipeline: TextEmbeddingPipelineWrapper | null = null;
  isInitialized = false;
  private readonly initialization = new PipelineInitialization("TextEmbeddingPipeline");

  constructor(
    modelPath: string,
    device: string,
    config: TextEmbeddingConfig,
    ovProperties?: object,
  ) {
    this.modelPath = modelPath;
    this.device = device;
    this.config = config;
    this.ovProperties = ovProperties || {};
  }

  async init() {
    this.pipeline = await this.initialization.run(async () => {
      const pipeline = new TextEmbeddingPipelineWrap();
      const initialize = util.promisify(pipeline.init.bind(pipeline));
      await initialize(this.modelPath, this.device, this.config, this.ovProperties);
      return pipeline;
    });
    this.isInitialized = true;
  }

  embedDocumentsSync(texts: string[]): EmbeddingResults {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    return this.pipeline.embedDocumentsSync(texts);
  }

  embedQuerySync(text: string): EmbeddingResult {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    return this.pipeline.embedQuerySync(text);
  }

  async embedDocuments(documents: string[]): Promise<EmbeddingResults> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const embedDocuments = util.promisify(this.pipeline.embedDocuments.bind(this.pipeline));
    const result = await embedDocuments(documents);

    return result;
  }

  async embedQuery(text: string): Promise<EmbeddingResult> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const embedQuery = util.promisify(this.pipeline.embedQuery.bind(this.pipeline));
    const result = await embedQuery(text);

    return result;
  }
}
