import { LLMPipeline } from '../dist/index.js';

import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import { models } from './models.js';

const MODEL_PATH = process.env.MODEL_PATH
  || `./tests/models/${models.LLM.split('/')[1]}`;

describe('tokenizer', async () => {
  let pipeline = null;
  let tokenizer = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, 'CPU');

    await pipeline.startChat();
    tokenizer = pipeline.getTokenizer();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it('applyChatTemplate return string', async () => {
    const template = tokenizer.applyChatTemplate([{
      role: 'user',
      content: 'continue: 1 2 3',
    }],
    false,
    );
    assert.strictEqual(typeof template, 'string');
  });

  it('applyChatTemplate with unknown property', async () => {
    const testValue = '1234567890';
    const template = tokenizer.applyChatTemplate([{
      role: 'user',
      content: 'continue: 1 2 3',
      unknownProp: testValue,
    }],
    false,
    );
    assert.ok(!template.includes(testValue));
  });

  it('applyChatTemplate with true addGenerationPrompt', async () => {
    const template = tokenizer.applyChatTemplate([{
      role: 'user',
      content: 'continue: 1 2 3',
    }],
    true,
    );
    assert.ok(template.includes('assistant'));
  });

  it('applyChatTemplate with missed role', async () => {
    assert.throws(() => tokenizer.applyChatTemplate([{
      content: 'continue: 1 2 3',
    }],
    false,
    ),
    );
  });

  it('applyChatTemplate with missed content', async () => {
    assert.throws(() => tokenizer.applyChatTemplate([{
      role: 'user',
    }],
    false,
    ));
  });

  it('applyChatTemplate with missed addGenerationPrompt', async () => {
    assert.throws(() => tokenizer.applyChatTemplate([{
      role: 'user',
      content: 'continue: 1 2 3',
    }],
    ));
  });

  it('applyChatTemplate with incorrect type of  history', async () => {
    assert.throws(
      () => tokenizer.applyChatTemplate('prompt',
        false,
      ),
    );
  });

  it('applyChatTemplate with unknown property', async () => {
    const testValue = '1234567890';
    const template = tokenizer.applyChatTemplate([{
      role: 'user',
      content: 'continue: 1 2 3',
      unknownProp: testValue,
    }],
    false,
    );
    assert.ok(!template.includes(testValue));
  });

  it('applyChatTemplate use custom chatTemplate', async () => {
    const prompt = 'continue: 1 2 3';
    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% endfor %}`;
    const template = tokenizer.applyChatTemplate([{
      role: 'user',
      content: prompt,
    }],
    false,
    chatTemplate);
    assert.strictEqual(template, `${prompt}\n`);
  });

  it('getBosToken return string', async () => {
    const token = tokenizer.getBosToken();
    assert.strictEqual(typeof token, 'string');
  });

  it('getBosTokenId return number', async () => {
    const token = tokenizer.getBosTokenId();
    assert.strictEqual(typeof token, 'number');
  });

  it('getEosToken return string', async () => {
    const token = tokenizer.getEosToken();
    assert.strictEqual(typeof token, 'string');
  });

  it('getEosTokenId return number', async () => {
    const token = tokenizer.getEosTokenId();
    assert.strictEqual(typeof token, 'number');
  });

  it('getPadToken return string', async () => {
    const token = tokenizer.getPadToken();
    assert.strictEqual(typeof token, 'string');
  });

  it('getPadTokenId return number', async () => {
    const token = tokenizer.getPadTokenId();
    assert.strictEqual(typeof token, 'number');
  });
});
