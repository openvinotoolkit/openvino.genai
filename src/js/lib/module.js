import util from 'node:util';

import addon from './bindings.cjs';

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

  async generate(prompt, generationCallback, options = {}) {
    if (!this.isInitialized)
      throw new Error('Pipeline is not initialized');

    if (typeof prompt !== 'string')
      throw new Error('Prompt must be a string');
    if (typeof generationCallback !== 'function')
      throw new Error('Generation callback must be a function');
    if (typeof options !== 'object')
      throw new Error('Options must be an object');

    let result = '';
    const castedOptions = {};

    for (const key in options) castedOptions[key] = String(options[key]);

    const promise = new Promise((resolve, reject) => {
      const generationCallbackDecorator = function(isDone, chunk) {
        if (isDone) return resolve(result);

        result += chunk;

        try {
          generationCallback(chunk);
        } catch (err) {
          reject(err);
        }
      };

      try {
        this.pipeline.generate(prompt, generationCallbackDecorator, castedOptions);
      } catch (err) {
        reject(err);
      }
    });

    return promise;
  }
}

const availablePipelines = { LLMPipeline: LLMPipeline };

class Pipeline {
  static async create(pipelineType, modelPath, device = 'CPU') {
    if (!Object.keys(availablePipelines).includes(pipelineType))
      throw new Error(`Pipeline type: '${pipelineType}' doesn't support`);

    const pipeline = new availablePipelines[pipelineType](modelPath, device);
    await pipeline.init();

    return pipeline;
  }
}

const availablePipelinesKeys = Object.keys(availablePipelines);

export {
  addon,
  Pipeline,
  availablePipelinesKeys as availablePipelines,
};
