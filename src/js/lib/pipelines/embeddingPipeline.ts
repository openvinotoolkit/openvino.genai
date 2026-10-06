// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import type { Tensor } from "openvino-node";
import {
  EmbeddingPipelineWrapper,
  EmbeddingPipelineProperties,
  EmbedResult,
  VideoMetadata,
  EmbeddingPipeline as EmbeddingPipelineWrap,
} from "../addon.js";

/**
 * Options for {@link EmbeddingPipeline} embedding methods.
 */
export type EmbedOptions = {
  /** Array of image tensors to embed. */
  images?: Tensor[];
  /** Array of video frame tensors to embed. */
  videos?: Tensor[];
  /** Metadata for each provided video controlling frame sampling. */
  videosMetadata?: VideoMetadata[];
  /**
   * Instruction describing how the input should be encoded.
   * If the model has a chat template, the prompt is added to the system message; otherwise it is prepended to the text.
   */
  embedding_prompt?: string;
  /** Additional embed-time properties. */
  [key: string]: unknown;
};

/**
 * Multimodal embedding pipeline computing embeddings for text, images and videos.
 */
export class EmbeddingPipeline {
  modelPath: string;
  device: string;
  properties: EmbeddingPipelineProperties;
  pipeline: EmbeddingPipelineWrapper | null = null;

  constructor(modelPath: string, device: string, properties: EmbeddingPipelineProperties = {}) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  async init() {
    if (this.pipeline) throw new Error("EmbeddingPipeline is already initialized");

    this.pipeline = new EmbeddingPipelineWrap();

    const initPromise = util.promisify(this.pipeline.init.bind(this.pipeline));
    await initPromise(this.modelPath, this.device, this.properties);
  }

  async embed(text: string | string[], options: EmbedOptions = {}): Promise<EmbedResult> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const { images = [], videos = [], videosMetadata = [], ...properties } = options;
    const embed = util.promisify(this.pipeline.embed.bind(this.pipeline));

    return embed(text, images, videos, videosMetadata, properties);
  }
}
