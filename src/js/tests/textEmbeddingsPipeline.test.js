import { describe, it } from 'node:test';
import { TextEmbeddingPipeline } from '../dist/index.js';
import { isFloat32Array } from 'util/types';
import assert from 'node:assert/strict';

const EMBEDDING_MODEL_PATH = process.env.EMBEDDING_MODEL_PATH;

if (!EMBEDDING_MODEL_PATH) throw new Error(
    'Set the path to the model directory in the EMBEDDING_MODEL_PATH environment variable.'
);

describe('TextEmbeddingPipeline', async () => {
    let pipeline = await TextEmbeddingPipeline(EMBEDDING_MODEL_PATH, 'CPU');

    await it('async embed query', async () => {
        const result = pipeline.embedQuery("test");
        assert.ok(result instanceof Promise, "result should be Promise");

        const embed_result = await result;
        assert.ok(isFloat32Array(embed_result));
    });

    await it('async embed documents', async () => {
        const result = pipeline.embedDocuments(["Hello", "World"]);
        assert.ok(result instanceof Promise, "result should be Promise");

        const embed_result = await result;
        assert.ok(embed_result instanceof Array)
        assert.strictEqual(embed_result.length, 2)
        assert.ok(isFloat32Array(embed_result[0]));
    });

    it('sync embed query', () => {
        const embed_result = pipeline.embedQuerySync("test");
        assert.ok(isFloat32Array(embed_result));
    });

    it('sync embed documents', () => {
        const embed_result = pipeline.embedDocumentsSync(["Hello", "World"]);
        assert.ok(embed_result instanceof Array)
        assert.strictEqual(embed_result.length, 2)
        assert.ok(isFloat32Array(embed_result[0]));
    });
});
