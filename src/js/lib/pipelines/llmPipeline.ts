import util from "node:util";
import { ChatHistory, LLMPipeline as LLMPipelineWrapper } from "../addon.js";
import { GenerationConfig, StreamingStatus, LLMPipelineProperties } from "../utils.js";
import { DecodedResults } from "../decodedResults.js";
import { Tokenizer } from "../tokenizer.js";

export class LLMPipeline {
  modelPath: string;
  device: string;
  pipeline: LLMPipelineWrapper | null = null;
  properties: LLMPipelineProperties;

  constructor(modelPath: string, device: string, properties: LLMPipelineProperties) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  async init() {
    if (this.pipeline) throw new Error("LLMPipeline is already initialized");

    const pipeline = new LLMPipelineWrapper();

    const initPromise = util.promisify(pipeline.init.bind(pipeline));
    const result = await initPromise(this.modelPath, this.device, this.properties);
    this.pipeline = pipeline;

    return result;
  }

  async startChat(systemMessage: string = "") {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    const startChatPromise = util.promisify(this.pipeline.startChat.bind(this.pipeline));
    const result = await startChatPromise(systemMessage);

    return result;
  }
  async finishChat() {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    return result;
  }

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
        subword: string;
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

  getTokenizer(): Tokenizer {
    if (!this.pipeline) throw new Error("LLMPipeline is not initialized");
    return this.pipeline.getTokenizer();
  }
}
