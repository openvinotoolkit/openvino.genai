// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import util from "node:util";
import { OmniPipeline as OmniPipelineWrapper, type ChatHistory } from "../addon.js";
import {
  GenerationConfig,
  OmniPipelineProperties,
  OmniTalkerSpeechConfig,
  StreamingStatus,
} from "../utils.js";
import { OmniDecodedResults } from "../decodedResults.js";
import type { Tensor } from "openvino-node";

/**
 * Options for Omni generation.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type OmniGenerateOptions = {
  /** Array of image tensors to include in the prompt. */
  images?: Tensor[];
  /** Array of video frame tensors to include in the prompt. */
  videos?: Tensor[];
  /** Array of audio tensors to include in the prompt. */
  audios?: Tensor[];
  /** Generation configuration for the thinker text decode (max_new_tokens, temperature, etc.). */
  textConfig?: GenerationConfig;
  /** Generation configuration for the talker speech output (return_audio, speaker, etc.). */
  talkerSpeechConfig?: OmniTalkerSpeechConfig;
  /**
   * Optional callback invoked for each generated text subword chunk.
   * Return a {@link StreamingStatus} flag to continue, stop, or cancel generation.
   */
  streamer?: (chunk: string) => StreamingStatus;
  /**
   * Optional callback invoked for each generated audio chunk.
   * The chunk is a `[1, 1, N_samples]` f32 PCM tensor at 24 kHz.
   * Return a {@link StreamingStatus} flag to continue, stop, or cancel generation.
   */
  speechStreamer?: (audioChunk: Tensor) => StreamingStatus;
};

/**
 * This class is used for generation with omni models (e.g. Qwen3-Omni) that
 * produce both text and speech from text, image, video, and audio inputs.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export class OmniPipeline {
  protected readonly modelPath: string;
  protected readonly device: string;
  protected pipeline: OmniPipelineWrapper | null = null;
  protected readonly properties: OmniPipelineProperties;

  /**
   * Construct an Omni pipeline from a folder containing tokenizer and model IRs.
   * @param modelPath - A folder to read tokenizer and model IRs.
   * @param device - Inference device. A tokenizer is always compiled for CPU.
   * @param properties - Device and pipeline properties. Speech output uses the
   * continuous-batching backend, which is the default on CPU and GPU (NPU is not
   * supported for speech output).
   */
  constructor(modelPath: string, device: string, properties: OmniPipelineProperties) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Initialize the underlying native pipeline.
   * @returns Resolves when initialization is complete.
   */
  async init() {
    const pipeline = new OmniPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    await initPromise(this.modelPath, this.device, this.properties);

    this.pipeline = pipeline;
  }

  /**
   * Generate text and (optionally) speech with optional streaming.
   *
   * @param inputs - Input prompt string or chat history. May contain model-specific image/video/audio tags.
   * @param options - Optional parameters.
   * @param options.images - Array of image tensors to include in the prompt.
   * @param options.videos - Array of video frame tensors to include in the prompt.
   * @param options.audios - Array of audio tensors to include in the prompt.
   * @param options.textConfig - Generation config for the thinker text decode.
   * @param options.talkerSpeechConfig - Generation config for the talker speech output.
   * @param options.streamer - Optional callback invoked for each generated text subword chunk.
   * @param options.speechStreamer - Optional callback invoked for each generated audio chunk.
   * @returns Promise resolving to {@link OmniDecodedResults} with texts, perf metrics, and speech result.
   */
  async generate(
    inputs: string | ChatHistory,
    options: OmniGenerateOptions = {},
  ): Promise<OmniDecodedResults> {
    if (!this.pipeline) throw new Error("Pipeline is not initialized");
    const { images, videos, audios, textConfig, talkerSpeechConfig, streamer, speechStreamer } =
      options;
    const innerGenerate = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const result = await innerGenerate(
      inputs,
      images,
      videos,
      audios,
      streamer,
      speechStreamer,
      textConfig,
      talkerSpeechConfig,
    );

    return new OmniDecodedResults(
      result.texts,
      result.scores,
      result.perfMetrics,
      result.parsed,
      result.finishReasons,
      result.speechResult,
    );
  }
}
