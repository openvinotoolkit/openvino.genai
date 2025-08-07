import { LLMPipeline } from '../dist/index.js';

import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import { models } from './models.js';
import { hrtime } from 'node:process';

const MODEL_PATH = process.env.MODEL_PATH
  || `./tests/models/${models.LLM.split('/')[1]}`;

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
      () => { },
    );

    assert.ok(result.length > 0);
    assert.strictEqual(typeof result, 'string');
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
        async () => await pipeline.generate('prompt', 'options', () => { }),
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

describe('LLMPipeline.generate()', () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');
    await pipeline.startChat();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it('generate(prompt, config) return_decoded_results', async () => {
    const config = {
      'max_new_tokens': 5,
      'return_decoded_results': true,
    };
    const reply = await pipeline.generate('prompt', config);
    assert.strictEqual(typeof reply, 'object');
    assert.ok(Array.isArray(reply.texts));
    assert.ok(reply.texts.every(text => typeof text === 'string'));
    assert.ok(reply.perfMetrics !== undefined);

    const configStr = {
      'max_new_tokens': 5,
      'return_decoded_results': false,
    };
    const replyStr = await pipeline.generate('prompt', configStr);
    assert.strictEqual(typeof replyStr, 'string');
    assert.strictEqual(replyStr, reply.toString());
  });

  it('DecodedResults.perfMetrics', async () => {
    const config = {
      'max_new_tokens': 20,
      'return_decoded_results': true,
    };
    const prompt = 'The Sky is blue because';
    const start = hrtime.bigint();
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');
    await pipeline.startChat();
    const res = await pipeline.generate(prompt, config);
    const elapsed = Number(hrtime.bigint() - start) / 1e6;

    const { perfMetrics } = res;
    const loadTime = perfMetrics.getLoadTime();
    assert.ok(loadTime >= 0 && loadTime <= elapsed);

    const numGeneratedTokens = perfMetrics.getNumGeneratedTokens();
    assert.ok(numGeneratedTokens > 0);
    assert.ok(numGeneratedTokens <= config.max_new_tokens);

    const numInputTokens = perfMetrics.getNumInputTokens();
    assert.ok(numInputTokens > 0 && typeof numInputTokens === 'number');

    // assert.ok(perfMetrics.get_num_generated_tokens() !== undefined);
    // assert.ok(perfMetrics.get_generate_duration() !== undefined);
    // assert.ok(perfMetrics.get_tokenization_duration() !== undefined);
    // assert.ok(perfMetrics.get_detokenization_duration() !== undefined);
    // assert.ok(perfMetrics.get_ttft() !== undefined);
    // assert.ok(perfMetrics.get_tpot() !== undefined);
    // assert.ok(perfMetrics.get_throughput() !== undefined);
  });
});

describe('stream()', () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');
  });

  it('stream() with max_new_tokens', async () => {
    const streamer = pipeline.stream(
      'Print hello world',
      {
        'max_new_tokens': 5,
      },
    );
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
    }
    assert.ok(chunks.length < 5);
  });

  it('stream() with stop_strings', async () => {
    const streamer = pipeline.stream(
      'Print hello world',
      {
        'stop_strings': new Set(['world']),
        'include_stop_str_in_output': true,
      },
    );
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
    }
    assert.ok(chunks[chunks.length - 1].includes('world'));
  });

  it('early break of stream', async () => {
    const streamer = pipeline.stream('Print hello world');
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
      if (chunks.length >= 5) {
        break;
      }
    }
    assert.equal(chunks.length, 5);
  });
});
