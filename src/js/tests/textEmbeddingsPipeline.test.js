import { describe, it, before } from "node:test";
import { TextEmbeddingPipeline, PoolingType } from "../dist/index.js";
import { isFloat32Array } from "util/types";
import assert from "node:assert/strict";
import { models } from "./models.js";

const EMBEDDING_MODEL_PATH =
  process.env.EMBEDDING_MODEL_PATH || `./tests/models/${models.Embedding.split("/")[1]}`;

if (!EMBEDDING_MODEL_PATH)
  throw new Error(
    "Set the path to the model directory in the " + "EMBEDDING_MODEL_PATH environment variable.",
  );

describe("TextEmbeddingPipeline", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await TextEmbeddingPipeline(EMBEDDING_MODEL_PATH, "CPU");
  });

  it("async embed query", async () => {
    const result = pipeline.embedQuery("test");
    assert.ok(result instanceof Promise, "result should be Promise");

    const embedResult = await result;
    assert.ok(isFloat32Array(embedResult));
  });

  it("async embed documents", async () => {
    const result = pipeline.embedDocuments(["Hello", "World"]);
    assert.ok(result instanceof Promise, "result should be Promise");

    const embedResult = await result;
    assert.ok(embedResult instanceof Array);
    assert.strictEqual(embedResult.length, 2);
    assert.ok(isFloat32Array(embedResult[0]));
  });

  it("sync embed query", () => {
    const embedResult = pipeline.embedQuerySync("test");
    assert.ok(isFloat32Array(embedResult));
  });

  it("sync embed documents", () => {
    const embedResult = pipeline.embedDocumentsSync(["Hello", "World"]);
    assert.ok(embedResult instanceof Array);
    assert.strictEqual(embedResult.length, 2);
    assert.ok(isFloat32Array(embedResult[0]));
  });

  it("test TextEmbeddingPipeline config param", async () => {
    const pipelineWithConfig = await TextEmbeddingPipeline(EMBEDDING_MODEL_PATH, "CPU", {
      pooling_type: PoolingType.MEAN,
      normalize: false,
    });
    assert.ok(pipelineWithConfig instanceof Object);
  });
});
