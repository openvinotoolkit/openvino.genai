import addon from '../lib/bindings.cjs';

import assert from 'node:assert';
import { describe, it, before, after } from 'node:test';
import { models } from './models.js';

const MODEL_PATH = process.env.MODEL_PATH
  || `./tests/models/${models[0].split('/')[1]}`;

describe('bindings', () => {
  let pipeline = null;

  before((_, done) => {
    pipeline = new addon.LLMPipeline();

    pipeline.init(MODEL_PATH, 'CPU', (err) => {
      if (err) {
        console.error(err);
        process.exit(1);
      }

      pipeline.startChat((err) => {
        if (err) {
          console.error(err);
          process.exit(1);
        }

        done();
      });
    });
  });

  after((_, done) => {
    pipeline.finishChat((err) => {
      if (err) {
        console.error(err);
        process.exit(1);
      }

      done();
    });
  });

  it('should generate string result', (_, done) => {
    let output = '';

    pipeline.generate('Say Hello', (isDone, chunk) => {
      if (!isDone) {
        output += chunk;

        return;
      }
    }, { temperature: '0', max_new_tokens: '4' });

    assert.ok(output);
    done();
  });

  it('should generate "Hello world"', (_, done) => {
    let output = '';

    pipeline.generate('Type "Hello world!" in English', (isDone, chunk) => {
      if (!isDone) {
        output += chunk;

        return;
      }

      assert.ok(output.includes('Hello world!'));
      done();
    }, { temperature: '0', max_new_tokens: '4' });
  });
});
