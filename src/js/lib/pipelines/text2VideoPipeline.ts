// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { Text2VideoPipeline as Text2VideoPipelineWrapper } from "../addon.js";
import type { Tensor } from "../addon.js";

/**
 * Pipeline for text-to-video generation.
 *
 * Uses Text2ImagePipeline as a placeholder until the native
 * Text2VideoPipeline C++ API is available.
 */
export class Text2VideoPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected readonly properties: Record<string, unknown>;
  protected pipeline: Text2VideoPipelineWrapper | null = null;

  constructor(
    modelPath: string,
    device: string,
    properties: Record<string, unknown> = {},
  ) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  async init() {
    const pipeline = new Text2VideoPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);

    this.pipeline = pipeline;
  }

  async generate(
    prompt: string,
    config: Record<string, unknown> = {},
  ): Promise<Tensor> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");

    const generateFn = (
      p: string,
      c: Record<string, unknown>,
      cb: (err: Error | null, result: Tensor) => void,
    ) => {
      this.pipeline!.generate(p, cb, c);
    };

    const generatePromise = util.promisify(generateFn);
    const result = await generatePromise(prompt, config);

    return result;
  }
}
