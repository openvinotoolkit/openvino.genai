import { createRequire } from 'module'
import path from 'path';
import util from 'node:util';
import { platform } from 'node:os';
import getGenAIAddon from './bindings.cjs';

let ovPath;
const require = createRequire(import.meta.url);
try {
  ovPath = require.resolve('openvino-node');
} catch (error) {
  if (error.code === 'MODULE_NOT_FOUND') {
    console.error(error.message);
  }
  throw error;
}

if (platform() == "win32") {
  // Find the openvino libraries that are required for genai-node
  const pathToOpenVino = path.join(path.dirname(ovPath), '../bin');
  if (!process.env.PATH.includes('openvino-node')) {
    process.env.PATH += ';' + path.resolve(pathToOpenVino);
  }
}

const addon = getGenAIAddon();

class LLMPipeline {
  modelPath = null;
  device = null;
  pipeline = null;
  isInitialized = false;
  isChatStarted = false;

  constructor(modelPath, device) {
    this.modelPath = modelPath;
    this.device = device;
  }

  async init() {
    if (this.isInitialized)
      throw new Error('Pipeline is already initialized');

    this.pipeline = new addon.LLMPipeline();

    const init = util.promisify(this.pipeline.init.bind(this.pipeline));
    const result = await init(this.modelPath, this.device);

    this.isInitialized = true;

    return result;
  }

  async startChat() {
    if (this.isChatStarted)
      throw new Error('Chat is already started');

    const startChatPromise = util.promisify(
      this.pipeline.startChat.bind(this.pipeline)
    );
    const result = await startChatPromise();

    this.isChatStarted = true;

    return result;
  }
  async finishChat() {
    if (!this.isChatStarted)
      throw new Error('Chat is not started');

    const finishChatPromise = util.promisify(
      this.pipeline.finishChat.bind(this.pipeline)
    );
    const result = await finishChatPromise();

    this.isChatStarted = false;

    return result;
  }

  static castOptionsToString(options) {
    const castedOptions = {};

    for (const key in options)
      castedOptions[key] = String(options[key]);

    return castedOptions;
  }

  getAsyncGenerator(prompt, generationOptions = {}) {
    if (!this.isInitialized)
      throw new Error('Pipeline is not initialized');

    if (typeof prompt !== 'string')
      throw new Error('Prompt must be a string');
    if (typeof generationOptions !== 'object')
      throw new Error('Options must be an object');

    const castedOptions = LLMPipeline.castOptionsToString(generationOptions);

    const queue = [];
    let resolvePromise;

    // Callback function that C++ will call when a chunk is ready
    function chunkOutput(isDone, subword) {
      if (resolvePromise) {
        resolvePromise({ value: subword, done: isDone }); // Fulfill pending request
        resolvePromise = null;  // Reset promise resolver
      } else {
        queue.push({ isDone, subword }); // Add data to queue if no pending promise
      }
    }

    this.pipeline.generate(prompt, chunkOutput, castedOptions);

    return {
      async next() {
        // If there is data in the queue, return it
        // Otherwise, return a promise that will resolve when data is available
        if (queue.length > 0) {
          const { isDone, subword } = queue.shift();

          return { value: subword, done: isDone };
        }

        return new Promise((resolve) => (resolvePromise = resolve));
      },
      [Symbol.asyncIterator]() { return this; }
    };
  }

  async generate(prompt, generationOptions, generationCallback) {
    const options = generationOptions || {};

    if (generationCallback !== undefined && typeof generationCallback !== 'function')
      throw new Error('Generation callback must be a function');

    const g = this.getAsyncGenerator(prompt, options);
    const result = [];

    for await (const chunk of g) {
      result.push(chunk);

      if (generationCallback) generationCallback(chunk);
    }

    return result.join('');
  }
}

class Pipeline {
  static async LLMPipeline(modelPath, device = 'CPU') {
    const pipeline = new LLMPipeline(modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}


export {
  addon,
  Pipeline,
};
