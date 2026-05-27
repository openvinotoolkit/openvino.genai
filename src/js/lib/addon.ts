// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "module";
import { platform } from "node:os";
import { join, dirname, resolve } from "node:path";
import { Tensor } from "openvino-node";
import type { ChatHistory as IChatHistory } from "./chatHistory.js";
import type { Tokenizer as ITokenizer } from "./tokenizer.js";
import { addon as ovAddon } from "openvino-node";
import {
  IReasoningParser,
  IDeepSeekR1ReasoningParser,
  IPhi4ReasoningParser,
  ILlama3PythonicToolParser,
  ILlama3JsonToolParser,
} from "./parsers.js";
import {
  GenerationConfig,
  GenerationFinishReason,
  StreamingStatus,
  VLMPipelineProperties,
  LLMPipelineProperties,
  WhisperGenerationConfig,
  WhisperPipelineProperties,
  SpeechGenerationConfig,
  ImageGenerationConfig,
  Text2ImageCallback,
  Text2ImagePipelineProperties,
  Text2SpeechPipelineProperties,
} from "./utils.js";
import {
  VLMPerfMetrics,
  PerfMetrics,
  WhisperPerfMetrics,
  Text2ImagePerfMetrics,
  Text2SpeechPerfMetrics,
} from "./perfMetrics.js";
import type { WhisperDecodedResultChunk, WhisperWordTiming } from "./decodedResults.js";

export type EmbeddingResult = Float32Array | Int8Array | Uint8Array;
export type EmbeddingResults = Float32Array[] | Int8Array[] | Uint8Array[];
export type TextRerankResult = [index: number, score: number];
export type TextRerankResults = TextRerankResult[];
/**
 * Pooling strategy
 */
export enum PoolingType {
  /** First token embeddings */
  CLS,
  /** The average of all token embeddings */
  MEAN,
}
export type TextEmbeddingConfig = {
  /** Maximum length of tokens passed to the embedding model */
  max_length?: number;
  /** If 'true', model input tensors are padded to the maximum length */
  pad_to_max_length?: boolean;
  /** Side to use for padding "left" or "right" */
  padding_side?: "left" | "right";
  /**
   * Batch size of embedding model.
   * Useful for database population. If set, the pipeline will fix model shape for inference optimization.
   * Number of documents passed to pipeline should be equal to batch_size.
   * For query embeddings, batch_size should be set to 1 or not set.
   */
  batch_size?: number;
  /** Pooling strategy applied to model output tensor */
  pooling_type?: PoolingType;
  /** If 'true', L2 normalization is applied to embeddings */
  normalize?: boolean;
  /** Instruction to use for embedding a query */
  query_instruction?: string;
  /** Instruction to use for embedding a document */
  embed_instruction?: string;
};

export interface TextEmbeddingPipelineWrapper {
  new (): TextEmbeddingPipelineWrapper;
  init(
    modelPath: string,
    device: string,
    config: TextEmbeddingConfig,
    ovProperties: object,
    callback: (err: Error | null) => void,
  ): void;
  embedQuery(text: string, callback: (err: Error | null, value: EmbeddingResult) => void): void;
  embedDocuments(
    documents: string[],
    callback: (err: Error | null, value: EmbeddingResults) => void,
  ): void;
  embedQuerySync(text: string): EmbeddingResult;
  embedDocumentsSync(documents: string[]): EmbeddingResults;
}

/**
 * Configuration parameters for TextRerankPipeline.
 */
export type TextRerankPipelineConfig = {
  /**
   * Number of documents to return sorted by score.
   * @defaultValue 3
   */
  top_n?: number;
  /** Maximum length of tokens passed to the embedding model. */
  max_length?: number;
  /** If 'true', model input tensors are padded to the maximum length. */
  pad_to_max_length?: boolean;
  /** Side to use for padding "left" or "right". */
  padding_side?: "left" | "right";
};

export interface TextRerankPipeline {
  new (): TextRerankPipeline;
  init(
    modelPath: string,
    device: string,
    config: TextRerankPipelineConfig,
    ovProperties: object,
    callback: (err: Error | null) => void,
  ): void;
  rerank(
    query: string,
    documents: string[],
    callback: (err: Error | null, value: TextRerankResults) => void,
  ): void;
}

export interface LLMPipeline {
  new (): LLMPipeline;
  init(
    modelPath: string,
    device: string,
    ovProperties: LLMPipelineProperties,
    callback: (err: Error | null) => void,
  ): void;
  generate(
    inputs: string | string[] | IChatHistory,
    generationConfig: GenerationConfig,
    streamer: ((chunk: string) => StreamingStatus) | undefined,
    callback: (
      err: Error | null,
      result: {
        texts: string[];
        scores: number[];
        perfMetrics: PerfMetrics;
        parsed: Record<string, unknown>[];
        finishReasons: GenerationFinishReason[];
      },
    ) => void,
  ): void;
  startChat(systemMessage: string, callback: (err: Error | null) => void): void;
  finishChat(callback: (err: Error | null) => void): void;
  getTokenizer(): ITokenizer;
  getGenerationConfig(): GenerationConfig;
  setGenerationConfig(config: GenerationConfig): void;
}

export interface WhisperPipeline {
  new (): WhisperPipeline;
  init(
    modelPath: string,
    device: string,
    properties: WhisperPipelineProperties,
    callback: (err: Error | null) => void,
  ): void;
  generate(
    rawSpeech: Float32Array | number[],
    generationConfig: WhisperGenerationConfig,
    streamer: ((chunk: string) => StreamingStatus) | undefined,
    callback: (
      err: Error | null,
      result: {
        texts: string[];
        scores: number[];
        perfMetrics: WhisperPerfMetrics;
        chunks?: WhisperDecodedResultChunk[];
        words?: WhisperWordTiming[];
      },
    ) => void,
  ): void;
  getTokenizer(): ITokenizer;
  getGenerationConfig(): Partial<WhisperGenerationConfig>;
  setGenerationConfig(config: WhisperGenerationConfig): void;
}

export interface VLMPipeline {
  new (): VLMPipeline;
  init(
    modelPath: string,
    device: string,
    ovProperties: VLMPipelineProperties,
    callback: (err: Error | null) => void,
  ): void;
  generate(
    inputs: string | IChatHistory,
    images: Tensor[] | undefined,
    videos: Tensor[] | undefined,
    streamer: ((chunk: string) => StreamingStatus) | undefined,
    generationConfig: GenerationConfig | undefined,
    callback: (
      err: Error | null,
      result: {
        texts: string[];
        scores: number[];
        perfMetrics: VLMPerfMetrics;
        parsed: Record<string, unknown>[];
        finishReasons: GenerationFinishReason[];
      },
    ) => void,
  ): void;
  startChat(systemMessage: string, callback: (err: Error | null) => void): void;
  finishChat(callback: (err: Error | null) => void): void;
  getTokenizer(): ITokenizer;
  setChatTemplate(template: string): void;
  setGenerationConfig(config: GenerationConfig): void;
  getGenerationConfig(): GenerationConfig;
}

export interface Text2ImagePipeline {
  new (): Text2ImagePipeline;
  init(
    modelPath: string,
    device: string,
    properties: Text2ImagePipelineProperties,
    callback: (err: Error | null) => void,
  ): void;
  generate(
    prompt: string,
    properties: ImageGenerationConfig,
    streamer: Text2ImageCallback | undefined,
    callback: (err: Error | null, result: Tensor) => void,
  ): void;
  getPerformanceMetrics(): Text2ImagePerfMetrics;
  getGenerationConfig(): ImageGenerationConfig;
  setGenerationConfig(config: ImageGenerationConfig): void;
}

export interface Text2SpeechPipeline {
  new (): Text2SpeechPipeline;
  init(
    modelPath: string,
    device: string,
    properties: Text2SpeechPipelineProperties,
    callback: (err: Error | null) => void,
  ): void;
  generate(
    inputs: string | string[],
    speakerEmbedding: Tensor | undefined,
    properties: SpeechGenerationConfig,
    callback: (
      err: Error | null,
      result: {
        speeches: Tensor[];
        perfMetrics: Text2SpeechPerfMetrics;
      },
    ) => void,
  ): void;
  getGenerationConfig(): SpeechGenerationConfig;
  setGenerationConfig(config: SpeechGenerationConfig): void;
}

interface OpenVINOGenAIAddon {
  TextRerankPipeline: TextRerankPipeline;
  TextEmbeddingPipeline: TextEmbeddingPipelineWrapper;
  LLMPipeline: LLMPipeline;
  VLMPipeline: VLMPipeline;
  WhisperPipeline: WhisperPipeline;
  Text2ImagePipeline: Text2ImagePipeline;
  Text2SpeechPipeline: Text2SpeechPipeline;
  ChatHistory: IChatHistory;
  Tokenizer: ITokenizer;
  ReasoningParser: IReasoningParser;
  DeepSeekR1ReasoningParser: IDeepSeekR1ReasoningParser;
  Phi4ReasoningParser: IPhi4ReasoningParser;
  Llama3PythonicToolParser: ILlama3PythonicToolParser;
  Llama3JsonToolParser: ILlama3JsonToolParser;
  setOpenvinoAddon: (ovAddon: any) => void;
}

// We need to use delayed import to get an updated Path if required
function getGenAIAddon(): OpenVINOGenAIAddon {
  const require = createRequire(import.meta.url);
  const ovPath = require.resolve("openvino-node");
  if (platform() == "win32") {
    // Find the openvino binaries that are required for openvino-genai-node
    const pathToOpenVino = join(dirname(ovPath), "../bin");
    if (!process.env.PATH?.includes("openvino-node")) {
      process.env.PATH += ";" + resolve(pathToOpenVino);
    }
  }

  return require("../bin/genai_node_addon.node");
}

const addon = getGenAIAddon();
addon.setOpenvinoAddon(ovAddon);

export const {
  TextEmbeddingPipeline,
  TextRerankPipeline,
  LLMPipeline,
  VLMPipeline,
  WhisperPipeline,
  Text2ImagePipeline,
  Text2SpeechPipeline,
  ChatHistory,
  Tokenizer,
  ReasoningParser,
  DeepSeekR1ReasoningParser,
  Phi4ReasoningParser,
  Llama3PythonicToolParser,
  Llama3JsonToolParser,
} = addon;
export type ChatHistory = IChatHistory;
export type Tokenizer = ITokenizer;
export type ReasoningParser = IReasoningParser;
export type DeepSeekR1ReasoningParser = IDeepSeekR1ReasoningParser;
export type Phi4ReasoningParser = IPhi4ReasoningParser;
export type Llama3PythonicToolParser = ILlama3PythonicToolParser;
export type Llama3JsonToolParser = ILlama3JsonToolParser;
