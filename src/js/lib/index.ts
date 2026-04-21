// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LLMPipeline as LLM } from "./pipelines/llmPipeline.js";
import { VLMPipeline as VLM } from "./pipelines/vlmPipeline.js";
import { TextEmbeddingPipeline as Embedding } from "./pipelines/textEmbeddingPipeline.js";
import {
  TextRerankPipeline as TextRerank,
  TextRerankPipelineOptions,
} from "./pipelines/textRerankPipeline.js";
import { WhisperPipeline as Whisper } from "./pipelines/whisperPipeline.js";
import { Text2ImagePipeline as Text2Image } from "./pipelines/text2ImagePipeline.js";
import { Text2SpeechPipeline as Text2Speech } from "./pipelines/text2SpeechPipeline.js";
import {
  LLMPipelineProperties,
  VLMPipelineProperties,
  WhisperPipelineProperties,
  Text2ImagePipelineProperties,
  Text2SpeechPipelineProperties,
} from "./utils.js";

class PipelineFactory {
  static async LLMPipeline(modelPath: string, device?: string): Promise<any>;
  static async LLMPipeline(
    modelPath: string,
    device: string,
    properties?: LLMPipelineProperties,
  ): Promise<any>;
  static async LLMPipeline(
    modelPath: string,
    device?: string,
    properties: LLMPipelineProperties = {},
  ) {
    if (device === undefined) device = "CPU";
    if (typeof device !== "string") {
      throw new Error(
        "The second argument must be a device string. If you want to pass LLMPipelineProperties, please use the third argument.",
      );
    }

    const pipeline = new LLM(modelPath, device, properties);
    await pipeline.init();
    return pipeline;
  }

  static async VLMPipeline(
    modelPath: string,
    device: string = "CPU",
    properties: VLMPipelineProperties = {},
  ) {
    const pipeline = new VLM(modelPath, device, properties);
    await pipeline.init();

    return pipeline;
  }

  static async TextEmbeddingPipeline(modelPath: string, device = "CPU", config = {}) {
    const pipeline = new Embedding(modelPath, device, config);
    await pipeline.init();

    return pipeline;
  }

  static async TextRerankPipeline(modelPath: string, options: TextRerankPipelineOptions = {}) {
    const pipeline = new TextRerank(modelPath, options);
    await pipeline.init();

    return pipeline;
  }

  static async WhisperPipeline(
    modelPath: string,
    device: string = "CPU",
    properties: WhisperPipelineProperties = {},
  ) {
    const pipeline = new Whisper(modelPath, device, properties);
    await pipeline.init();

    return pipeline;
  }

  static async Text2ImagePipeline(
    modelPath: string,
    device: string = "CPU",
    properties: Text2ImagePipelineProperties = {},
  ) {
    const pipeline = new Text2Image(modelPath, device, properties);
    await pipeline.init();

    return pipeline;
  }

  static async Text2SpeechPipeline(
    modelPath: string,
    device: string = "CPU",
    properties: Text2SpeechPipelineProperties = {},
  ) {
    const pipeline = new Text2Speech(modelPath, device, properties);
    await pipeline.init();

    return pipeline;
  }
}

export const {
  LLMPipeline,
  VLMPipeline,
  TextEmbeddingPipeline,
  TextRerankPipeline,
  WhisperPipeline,
  Text2ImagePipeline,
  Text2SpeechPipeline,
} = PipelineFactory;
export {
  DecodedResults,
  VLMDecodedResults,
  WhisperDecodedResults,
  Text2SpeechDecodedResults,
} from "./decodedResults.js";
export type { WhisperDecodedResultChunk, WhisperWordTiming } from "./decodedResults.js";
export {
  PerfMetrics,
  VLMPerfMetrics,
  WhisperPerfMetrics,
  Text2ImagePerfMetrics,
  Text2SpeechPerfMetrics,
} from "./perfMetrics.js";
export * from "./utils.js";
export * from "./addon.js";
export type { TokenizedInputs, EncodeOptions, DecodeOptions } from "./tokenizer.js";
export type { ChatMessage, ExtraContext, ToolDefinition } from "./chatHistory.js";
export type { Parser, ReasoningParserOptions } from "./parsers.js";
