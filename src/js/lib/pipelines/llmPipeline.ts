import util from 'node:util';
import addon from '../addon.js';
import { GenerationConfig, StreamingStatus } from '../utils.js';

export type ResolveFunction = (arg: { value: string, done: boolean }) => void;
export type Options = {
  disableStreamer?: boolean,
  'max_new_tokens'?: number
};

interface Tokenizer {
  applyChatTemplate(
    chatHistory: {'role': string, 'content': string}[],
    addGenerationPrompt: boolean,
    chatTemplate?: string
  ): string;
  getBosToken(): string;
  getBosTokenId(): number;
  getEosToken(): string;
  getEosTokenId(): number;
  getPadToken(): string;
  getPadTokenId(): number;
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
    if (this.isInitialized)
      throw new Error('LLMPipeline is already initialized');

    this.pipeline = new addon.LLMPipeline();

    const initPromise = util.promisify(this.pipeline.init.bind(this.pipeline));
    const result = await initPromise(this.modelPath, this.device);

    this.isInitialized = true;

    return result;
  }

  async startChat() {
    if (this.isChatStarted)
      throw new Error('Chat is already started');

    const startChatPromise = util.promisify(
      this.pipeline.startChat.bind(this.pipeline),
    );
    const result = await startChatPromise();

    this.isChatStarted = true;

    return result;
  }
  async finishChat() {
    if (!this.isChatStarted)
      throw new Error('Chat is not started');

    const finishChatPromise = util.promisify(
      this.pipeline.finishChat.bind(this.pipeline),
    );
    const result = await finishChatPromise();

    this.isChatStarted = false;

    return result;
  }

  stream(
    prompt: string,
    generationConfig: GenerationConfig = {},
  ) {
    if (!this.isInitialized)
      throw new Error('Pipeline is not initialized');

    if (typeof prompt !== 'string')
      throw new Error('Prompt must be a string');
    if (typeof generationConfig !== 'object')
      throw new Error('Options must be an object');

    let streamingStatus: StreamingStatus = StreamingStatus.RUNNING;
    const queue: { isDone: boolean; subword: string; }[] = [];
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

        return new Promise(
          (resolve: ResolveFunction) => (resolvePromise = resolve),
        );
      },
      async return() {
        streamingStatus = StreamingStatus.CANCEL;

        return { done: true };
      },
      [Symbol.asyncIterator]() { return this; },
    };
  }

  async generate(
    prompt: string | string[],
    generationConfig: GenerationConfig = {},
    callback: (chunk: string) => void | undefined,
  ) {
    if (typeof prompt !== 'string'
      && !(Array.isArray(prompt)
        && prompt.every(item => typeof item === 'string')))
      throw new Error('Prompt must be a string or string[]');
    if (typeof generationConfig !== 'object')
      throw new Error('Options must be an object');
    if (callback !== undefined && typeof callback !== 'function')
      throw new Error('Callback must be a function');

    const options: { 'disableStreamer'?: boolean } = {};
    if (!callback) {
      options['disableStreamer'] = true;
    }

    return new Promise(
      (resolve: (value: string) => void) => {
        const chunkOutput = (isDone: boolean, subword: string) => {
          if (isDone) {
            resolve(subword);
          } else if (callback) {
            return callback(subword);
          }

          return StreamingStatus.RUNNING;
        };
        this.pipeline.generate(prompt, chunkOutput, generationConfig, options);
      },
    );
  }

  getTokenizer(): Tokenizer {
    return this.pipeline.getTokenizer();
  }
}
