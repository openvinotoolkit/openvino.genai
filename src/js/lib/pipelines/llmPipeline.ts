import util from 'node:util';
import addon from '../addon.js';

export type ResolveFunction = (arg: { value: string, done: boolean }) => void;
export type Options = {[key: string]: string};

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

  static castOptionsToString(options: Options) {
    const castedOptions: Options = {};

    for (const key in options)
      castedOptions[key] = String(options[key]);

    return castedOptions;
  }

  getAsyncGenerator(prompt: string, generationOptions: Options = {}) {
    if (!this.isInitialized)
      throw new Error('Pipeline is not initialized');

    if (typeof prompt !== 'string')
      throw new Error('Prompt must be a string');
    if (typeof generationOptions !== 'object')
      throw new Error('Options must be an object');

    const castedOptions = LLMPipeline.castOptionsToString(generationOptions);

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

    this.pipeline.generate(prompt, chunkOutput, castedOptions);

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
    prompt: string,
    generationOptions: Options,
    generationCallback: (chunk: string)=>void | undefined,
  ) {

    if (generationCallback !== undefined
      && typeof generationCallback !== 'function')
      throw new Error('Generation callback must be a function');

    const g = this.getAsyncGenerator(prompt, generationOptions);
    const result = [];

    for await (const chunk of g) {
      result.push(chunk);

      if (generationCallback) generationCallback(chunk);
    }

    return result.join('');
  }
}
