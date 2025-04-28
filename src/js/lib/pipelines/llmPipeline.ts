import util from 'node:util';
import addon from '../addon.js';

export type ResolveFunction = (arg: { value: string, done: boolean }) => void;
export type Options = {[key: string]: string | boolean | number | BigInt};

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

  stream(prompt: string, generationOptions: Options = {}) {
    if (!this.isInitialized)
      throw new Error('Pipeline is not initialized');

    if (typeof prompt !== 'string')
      throw new Error('Prompt must be a string');
    if (typeof generationOptions !== 'object')
      throw new Error('Options must be an object');

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
    }

    this.pipeline.generate(prompt, chunkOutput, generationOptions);

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
      [Symbol.asyncIterator]() { return this; },
    };
  }

  async generate(
    prompt: string | string[],
    options: Options,
    callback: (chunk: string)=>void | undefined,
  ) {
    if (typeof prompt !== 'string'
      && !(Array.isArray(prompt)
      && prompt.every(item => typeof item === 'string')))
      throw new Error('Prompt must be a string or string[]');
    if (typeof options !== 'object')
      throw new Error('Options must be an object');
    if (callback !== undefined && typeof callback !== 'function')
      throw new Error('Callback must be a function');

    if (!callback) {
      options['disableStreamer'] = true;
    }

    return new Promise(
      (resolve: (value: string) => void) => {
        const chunkOutput = (isDone: boolean, subword: string) => {
          if (isDone) {
            resolve(subword);
          } else if (callback) callback(subword);
        };
        this.pipeline.generate(prompt, chunkOutput, options);
      },
    );
  }
}
