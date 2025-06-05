import { LLMPipeline } from '../dist/index.js';

import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import { models } from './models.js';

const MODEL_PATH = process.env.MODEL_PATH
  || `./tests/models/${models[0].split('/')[1]}`;

describe('module', async () => {
  let pipeline = null;

  await before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await pipeline.startChat();
  });

  await after(async () => {
    await pipeline.finishChat();
  });

  await it('should generate non empty string', async () => {
    const result = await pipeline.generate(
      'Type something in English',
      // eslint-disable-next-line camelcase
      { temperature: '0', max_new_tokens: '4' },
      () => {},
    );

    assert.ok(result.length > 0);
  });
});

describe('corner cases', async () => {
  it('should throw an error if pipeline is already initialized', async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await assert.rejects(
      async () => await pipeline.init(),
      {
        name: 'Error',
        message: 'LLMPipeline is already initialized',
      },
    );
  });

  it('should throw an error if chat is already started', async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await pipeline.startChat();

    await assert.rejects(
      () => pipeline.startChat(),
      {
        name: 'Error',
        message: 'Chat is already started',
      },
    );
  });

  it('should throw an error if chat is not started', async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await assert.rejects(
      () => pipeline.finishChat(),
      {
        name: 'Error',
        message: 'Chat is not started',
      },
    );
  });
});

describe('generation parameters validation', () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await pipeline.startChat();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it('should throw an error if temperature is not a number', async () => {
    await assert.rejects(
      async () => await pipeline.generate(),
      {
        name: 'Error',
        message: 'Prompt must be a string or string[]',
      },
    );
  });

  it(
    'should throw an error if generationCallback is not a function',
    async () => {
      const pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

      await pipeline.startChat();

      await assert.rejects(
        async () => await pipeline.generate('prompt', {}, false),
        {
          name: 'Error',
          message: 'Callback must be a function',
        },
      );
    });

  it(
    'should throw an error if options specified but not an object',
    async () => {
      await assert.rejects(
        async () => await pipeline.generate('prompt', 'options', () => {}),
        {
          name: 'Error',
          message: 'Options must be an object',
        },
      );
    });

  it('should perform generation with default options', async () => {
    try {
      // eslint-disable-next-line camelcase
      await pipeline.generate('prompt', { max_new_tokens: 1 });
    } catch(error) {
      assert.fail(error);
    }

    assert.ok(true);
  });

  it('should return a string as generation result', async () => {
    // eslint-disable-next-line camelcase
    const reply = await pipeline.generate('prompt', { max_new_tokens: 1 });

    assert.strictEqual(typeof reply, 'string');
  });

  it('should call generationCallback with string chunk', async () => {
    // eslint-disable-next-line camelcase
    await pipeline.generate('prompt', { max_new_tokens: 1 }, (chunk) => {
      assert.strictEqual(typeof chunk, 'string');
    });
  });

  it('should convert Set', async () => {
    const generationConfig = {
      'max_new_tokens': 100,
      'stop_strings': new Set(['1', '2', '3', '4', '5']),
      'include_stop_str_in_output': true,
    };
    const result = await pipeline.generate('continue: 1 2 3', generationConfig);
    assert.strictEqual(typeof result, 'string');
  });
});
