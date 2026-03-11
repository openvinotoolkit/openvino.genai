import util from "node:util";
import { ChatHistory, LLMPipeline as LLMPipelineWrapper } from "../addon.js";
import { GenerationConfig, StreamingStatus, LLMPipelineProperties } from "../utils.js";
import { DecodedResults } from "../decodedResults.js";
import { Tokenizer } from "../tokenizer.js";

/**
 * This class is used for generation with Large Language Models (LLMs)
 */
export class LLMPipeline {
  modelPath: string;
  device: string;
  pipeline: LLMPipelineWrapper | null = null;
  properties: LLMPipelineProperties;

  /**
   * Construct an LLM pipeline from a folder containing tokenizer and model IRs.
   * @param modelPath - A folder to read tokenizer and model IRs.
   * @param device - Inference device. A tokenizer is always compiled for CPU.
   * @param properties - Device and pipeline properties.
   */
  constructor(modelPath: string, device: string, properties: LLMPipelineProperties) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  /**
   * Initialize the underlying native pipeline.
   * @returns Resolves when initialization is complete.
   */
  async init() {
    if (this.pipeline) throw new Error("LLMPipeline is already initialized");

    const pipeline = new LLMPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    const result = await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;

    return result;
  }

  /**
   * Start a chat session with an optional system message.
   * @param systemMessage - Optional system message to initialize chat context.
   * @returns Resolves when chat session is started.
   * @deprecated startChat is deprecated and will be removed in future releases. Please, use generate() with ChatHistory argument.
   */
  async startChat(systemMessage: string = "") {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in the next major release.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    const startChatPromise = util.promisify(this.pipeline.startChat.bind(this.pipeline));
    const result = await startChatPromise(systemMessage);

    return result;
  }

  /**
   * Finish the current chat session and clear chat-related state.
   * @returns Resolves when chat session is finished.
   * @deprecated finishChat is deprecated and will be removed in future releases. Please, use generate() with ChatHistory argument.
   */
  async finishChat() {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in the next major release.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    return result;
  }

  /**
   * Get the current generation config (model defaults).
   * @returns The current GenerationConfig object.
   */
  getGenerationConfig(): GenerationConfig {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    return this.pipeline.getGenerationConfig();
  }

  /**
   * Set generation configuration parameters.
   * @param config - Generation configuration parameters.
   */
  setGenerationConfig(config: GenerationConfig): void {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    this.pipeline.setGenerationConfig(config);
  }

  /**
   * Stream generation results as an async iterator of strings.
   * The iterator yields subword chunks.
   * @param inputs - Input prompt string or chat history.
   * @param generationConfig - Generation configuration parameters.
   * @returns Async iterator producing subword chunks.
   */
  stream(inputs: string | ChatHistory, generationConfig: GenerationConfig = {}) {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    if (Array.isArray(inputs))
      throw new Error(
        "Streaming is not supported for array of inputs. Please use LLMPipeline.generate() method.",
      );
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");

    let streamingStatus: StreamingStatus = StreamingStatus.RUNNING;
    const queue: { done: boolean; subword: string }[] = [];
    type ResolveFunction = (arg: { value: string; done: boolean }) => void;
    type RejectFunction = (reason?: unknown) => void;
    let resolvePromise: ResolveFunction | null;
    let rejectPromise: RejectFunction | null;

    const callback = (
      error: Error | null,
      result: {
        texts: string[];
        scores: number[];
        perfMetrics: any;
        parsed: Record<string, unknown>[];
      },
    ) => {
      if (error) {
        if (rejectPromise) {
          rejectPromise(error);
          // Reset promises
          resolvePromise = null;
          rejectPromise = null;
        } else {
          throw error;
        }
      } else {
        const decodedResult = new DecodedResults(
          result.texts,
          result.scores,
          result.perfMetrics,
          result.parsed,
        );
        const fullText = decodedResult.toString();
        if (resolvePromise) {
          // Fulfill pending request
          resolvePromise({ done: true, value: fullText });
          // Reset promises
          resolvePromise = null;
          rejectPromise = null;
        } else {
          // Add data to queue if no pending promise
          queue.push({ done: true, subword: fullText });
        }
      }
    };

    const streamer = (chunk: string): StreamingStatus => {
      if (resolvePromise) {
        // Fulfill pending request
        resolvePromise({ done: false, value: chunk });
        // Reset promises
        resolvePromise = null;
        rejectPromise = null;
      } else {
        // Add data to queue if no pending promise
        queue.push({ done: false, subword: chunk });
      }
      return streamingStatus;
    };

    this.pipeline.generate(inputs, generationConfig, streamer, callback);

    return {
      async next() {
        // If there is data in the queue, return it
        // Otherwise, return a promise that will resolve when data is available
        const data = queue.shift();

        if (data !== undefined) {
          return { value: data.subword, done: data.done };
        }

        return new Promise((resolve: ResolveFunction, reject: RejectFunction) => {
          resolvePromise = resolve;
          rejectPromise = reject;
        });
      },
      async return() {
        streamingStatus = StreamingStatus.CANCEL;

        return { done: true, value: "" };
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  }

  /**
   * Generate sequences for LLMs.
   * @param inputs - Input prompt string, array of prompts, or chat history.
   * @param generationConfig - Generation configuration parameters.
   * @param streamer - Optional streamer callback called for each chunk.
   * @returns Resolves with decoded results once generation finishes.
   */
  async generate(
    inputs: string | string[] | ChatHistory,
    generationConfig: GenerationConfig = {},
    streamer?: (chunk: string) => StreamingStatus,
  ): Promise<DecodedResults> {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");
    if (streamer !== undefined && typeof streamer !== "function")
      throw new Error("Streamer must be a function");

    const innerGenerate = util.promisify(this.pipeline.generate.bind(this.pipeline));
    const result = await innerGenerate(inputs, generationConfig, streamer);

    return new DecodedResults(result.texts, result.scores, result.perfMetrics, result.parsed);
  }

  /**
   * Get the pipeline tokenizer instance.
   * @returns Tokenizer used by the pipeline.
   */
  getTokenizer(): Tokenizer {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    return this.pipeline.getTokenizer();
  }
}
