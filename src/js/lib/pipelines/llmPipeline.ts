/* eslint-disable max-len */
import util from "node:util";
import addon from "../addon.js";
import { GenerationConfig, StreamingStatus } from "../utils.js";

export type ResolveFunction = (arg: { value: string; done: boolean }) => void;
export type Options = {
  disableStreamer?: boolean;
  max_new_tokens?: number;
};

interface Tokenizer {
  /** Embeds input prompts with special tags for a chat scenario. */
  applyChatTemplate(
    chatHistory: { role: string; content: string }[],
    addGenerationPrompt: boolean,
    chatTemplate?: string,
  ): string;
  getBosToken(): string;
  getBosTokenId(): number;
  getEosToken(): string;
  getEosTokenId(): number;
  getPadToken(): string;
  getPadTokenId(): number;
}

/** Structure with raw performance metrics for each generation before any statistics are calculated. */
export type RawMetrics = {
  /** Durations for each generate call in milliseconds. */
  generateDurations: number[];
  /** Durations for the tokenization process in milliseconds. */
  tokenizationDurations: number[];
  /** Durations for the detokenization process in milliseconds. */
  detokenizationDurations: number[];
  /** Times to the first token for each call in milliseconds. */
  timesToFirstToken: number[];
  /** Timestamps of generation every token or batch of tokens in milliseconds. */
  newTokenTimes: number[];
  /** Inference time for each token in milliseconds. */
  tokenInferDurations: number[];
  /** Batch sizes for each generate call. */
  batchSizes: number[];
  /** Total durations for each generate call in milliseconds. */
  durations: number[];
  /** Total inference duration for each generate call in microseconds. */
  inferenceDurations: number[];
  /** Time to compile the grammar in milliseconds. */
  grammarCompileTimes: number[];
};

export type MeanStdPair = {
  mean: number;
  std: number;
};

/**
 * Holds performance metrics for each generate call.
 *
 * PerfMetrics holds fields with mean and standard deviations for the following metrics:
    - Time To the First Token (TTFT), ms
    - Time per Output Token (TPOT), ms/token
    - Generate total duration, ms
    - Tokenization duration, ms
    - Detokenization duration, ms
    - Throughput, tokens/s
 * Additional fields include:
    - Load time, ms
    - Number of generated tokens
    - Number of tokens in the input prompt
 */
export interface PerfMetrics {
  /** Returns the load time in milliseconds. */
  getLoadTime(): number;
  /** Returns the number of generated tokens. */
  getNumGeneratedTokens(): number;
  /** Returns the number of tokens in the input prompt. */
  getNumInputTokens(): number;
  /** Returns the mean and standard deviation of Time To the First Token (TTFT) in milliseconds. */
  getTTFT(): MeanStdPair;
  /** Returns the mean and standard deviation of Time Per Output Token (TPOT) in milliseconds. */
  getTPOT(): MeanStdPair;
  /** Returns the mean and standard deviation of Inference time Per Output Token in milliseconds. */
  getIPOT(): MeanStdPair;
  /** Returns the mean and standard deviation of throughput in tokens per second. */
  getThroughput(): MeanStdPair;
  /** Returns the mean and standard deviation of inference durations in milliseconds. */
  getInferenceDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of generate durations in milliseconds. */
  getGenerateDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of tokenization durations in milliseconds. */
  getTokenizationDuration(): MeanStdPair;
  /** Returns the mean and standard deviation of detokenization durations in milliseconds. */
  getDetokenizationDuration(): MeanStdPair;
  /** A structure of RawPerfMetrics type that holds raw metrics. */
  rawMetrics: RawMetrics;
}

export class DecodedResults {
  constructor(texts: string[], scores: number[], perfMetrics: PerfMetrics) {
    this.texts = texts;
    this.scores = scores;
    this.perfMetrics = perfMetrics;
  }
  toString() {
    if (this.scores.length !== this.texts.length) {
      throw new Error("The number of scores and texts doesn't match in DecodedResults.");
    }
    if (this.texts.length === 0) {
      return "";
    }
    if (this.texts.length === 1) {
      return this.texts[0];
    }
    let result = "";
    for (let i = 0; i < this.texts.length - 1; ++i) {
      result += `${this.scores[i].toFixed(6)}: ${this.texts[i]}\n`;
    }
    result += `${this.scores[this.scores.length - 1].toFixed(
      6,
    )}: ${this.texts[this.texts.length - 1]}`;

    return result;
  }
  texts: string[];
  scores: number[];
  perfMetrics: PerfMetrics;
}

export class LLMPipeline {
  modelPath: string | null = null;
  device: string | null = null;
  pipeline: any | null = null;
  isInitialized = false;
  isChatStarted = false;

  constructor(modelPath: string, device: string) {
    this.modelPath = modelPath;
    this.device = device;
  }

  async init() {
    if (this.isInitialized) throw new Error("LLMPipeline is already initialized");

    this.pipeline = new addon.LLMPipeline();

    const initPromise = util.promisify(this.pipeline.init.bind(this.pipeline));
    const result = await initPromise(this.modelPath, this.device);

    this.isInitialized = true;

    return result;
  }

  async startChat() {
    if (this.isChatStarted) throw new Error("Chat is already started");

    const startChatPromise = util.promisify(this.pipeline.startChat.bind(this.pipeline));
    const result = await startChatPromise();

    this.isChatStarted = true;

    return result;
  }
  async finishChat() {
    if (!this.isChatStarted) throw new Error("Chat is not started");

    const finishChatPromise = util.promisify(this.pipeline.finishChat.bind(this.pipeline));
    const result = await finishChatPromise();

    this.isChatStarted = false;

    return result;
  }

  stream(prompt: string, generationConfig: GenerationConfig = {}) {
    if (!this.isInitialized) throw new Error("Pipeline is not initialized");

    if (typeof prompt !== "string") throw new Error("Prompt must be a string");
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

    this.pipeline.generate(prompt, chunkOutput, generationConfig);

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
    prompt: string | string[],
    generationConfig: GenerationConfig = {},
    callback: (chunk: string) => void | undefined,
  ) {
    if (
      typeof prompt !== "string" &&
      !(Array.isArray(prompt) && prompt.every((item) => typeof item === "string"))
    )
      throw new Error("Prompt must be a string or string[]");
    if (typeof generationConfig !== "object") throw new Error("Options must be an object");
    if (callback !== undefined && typeof callback !== "function")
      throw new Error("Callback must be a function");

    const options: { disableStreamer?: boolean } = {};
    if (!callback) {
      options["disableStreamer"] = true;
    }
    const returnDecoded = generationConfig["return_decoded_results"] || false;

    return new Promise((resolve: (value: string | DecodedResults) => void) => {
      const chunkOutput = (isDone: boolean, result: string | any) => {
        if (isDone && returnDecoded) {
          const decodedResults = new DecodedResults(
            result.texts,
            result.scores,
            result.perfMetrics,
          );
          resolve(decodedResults);
        } else if (isDone && !returnDecoded) {
          console.warn(
            "DEPRECATION WARNING: Starting in version 2026.0.0,",
            "LLMPipeline.generate() will return DecodedResults by default.\n",
            'To use the new behavior now, set "return_decoded_results": true',
            "in GenerationConfig.",
          );
          resolve(result.subword);
        } else if (callback && typeof result === "string") {
          return callback(result);
        }

        return StreamingStatus.RUNNING;
      };
      this.pipeline.generate(prompt, chunkOutput, generationConfig, options);
    });
  }

  getTokenizer(): Tokenizer {
    return this.pipeline.getTokenizer();
  }
}
