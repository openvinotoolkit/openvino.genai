import util from "node:util";
import { ChatHistory, LLMPipeline as LLMPipelineWrap } from "../addon.js";
import { GenerationConfig, StreamingStatus, LLMPipelineProperties } from "../utils.js";
import { DecodedResults } from "../decodedResults.js";
import { Tokenizer } from "../tokenizer.js";

export type ResolveFunction = (arg: { value: string; done: boolean }) => void;
export type Options = {
  disableStreamer?: boolean;
  max_new_tokens?: number;
};

export class LLMPipeline {
  modelPath: string;
  device: string;
  pipeline: any | null = null;
  properties: LLMPipelineProperties;
  isInitialized = false;
  isChatStarted = false;

  constructor(modelPath: string, device: string, properties: LLMPipelineProperties) {
    this.modelPath = modelPath;
    this.device = device;
    this.properties = properties;
  }

  async init() {
    if (this.isInitialized) throw new Error("LLMPipeline is already initialized");

    this.pipeline = new LLMPipelineWrap();

    const initPromise = util.promisify(this.pipeline.init.bind(this.pipeline));
    const result = await initPromise(this.modelPath, this.device, this.properties);

    this.isInitialized = true;

    return result;
  }

  async startChat(systemMessage: string = "") {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in future releases.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (this.isChatStarted) throw new Error("Chat is already started");

    const startChatPromise = util.promisify(this.pipeline.startChat.bind(this.pipeline));
    const result = await startChatPromise(systemMessage);

    this.isChatStarted = true;

    return result;
  }
  async finishChat() {
    console.warn(
      "DEPRECATION WARNING: startChat() / finishChat() API is deprecated and will be removed in future releases.",
      "Please, use generate() with ChatHistory argument.",
    );
    if (!this.isChatStarted) throw new Error("Chat is not started");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    this.isChatStarted = false;

    return result;
  }

  stream(inputs: string | ChatHistory, generationConfig: GenerationConfig = {}) {
    if (!this.isInitialized) throw new Error("Pipeline is not initialized");

    if (Array.isArray(inputs))
      throw new Error(
        "Streaming is not supported for array of inputs. Please use LLMPipeline.generate() method.",
      );
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");

    let streamingStatus: StreamingStatus = StreamingStatus.RUNNING;
    const queue: { isDone: boolean; subword: string }[] = [];
    let resolvePromise: ResolveFunction | null;

    // Callback function that C++ will call when a chunk is ready
    function chunkOutput(isDone: boolean, subword: string) {
      if (resolvePromise) {
        // Fulfill pending request
        resolvePromise({ value: subword, done: isDone });
        resolvePromise = null; // Reset promise resolver
      } else {
        // Add data to queue if no pending promise
        queue.push({ isDone, subword });
      }

      return streamingStatus;
    }

    this.pipeline.generate(inputs, chunkOutput, generationConfig);

    return {
      async next() {
        // If there is data in the queue, return it
        // Otherwise, return a promise that will resolve when data is available
        const data = queue.shift();

        if (data !== undefined) {
          const { isDone, subword } = data;

          return { value: subword, done: isDone };
        }

        return new Promise((resolve: ResolveFunction) => (resolvePromise = resolve));
      },
      async return() {
        streamingStatus = StreamingStatus.CANCEL;

        return { done: true };
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  }

  async generate(
    inputs: string | string[] | ChatHistory,
    generationConfig: GenerationConfig = {},
    callback: (chunk: string) => void | undefined,
  ) {
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");
    if (callback !== undefined && typeof callback !== "function")
      throw new Error("Callback must be a function");

    const options: { disableStreamer?: boolean } = {};
    if (!callback) {
      options["disableStreamer"] = true;
    }

    return new Promise((resolve: (value: DecodedResults) => void) => {
      const chunkOutput = (isDone: boolean, result: string | any) => {
        if (isDone) {
          const decodedResults = new DecodedResults(
            result.texts,
            result.scores,
            result.perfMetrics,
            result.parsed,
          );
          resolve(decodedResults);
        } else if (callback && typeof result === "string") {
          return callback(result);
        }

        return StreamingStatus.RUNNING;
      };
      this.pipeline.generate(inputs, chunkOutput, generationConfig, options);
    });
  }

  getTokenizer(): Tokenizer {
    return this.pipeline.getTokenizer();
  }
}
