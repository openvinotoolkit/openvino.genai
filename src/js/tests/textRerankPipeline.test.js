// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { TextRerankPipeline } from "../dist/index.js";
import { models } from "./models.js";

const RERANK_MODEL_PATH =
  process.env.RERANK_MODEL_PATH || `./tests/models/${models.TestReranking.split("/")[1]}`;

const docs = [
  "The temperature in London today is 15 degrees Celsius.",
  "London has sunny weather this afternoon.",
  "Paris weather forecast shows rain tomorrow.",
  "OpenVINO accelerates deep learning inference.",
];
const query = "What is the weather in London?";

describe("TextRerankPipeline", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await TextRerankPipeline(RERANK_MODEL_PATH, { device: "CPU" });
  });

  it("async rerank returns scores", async () => {
    const result = await pipeline.rerank(query, docs);
    assert.ok(Array.isArray(result), "rerank result should be an array");
    assert.ok(result.length > 0, "rerank result should not be empty");

    const [first] = result;
    assert.ok(Array.isArray(first), "each item should be a tuple [index, score]");
    assert.strictEqual(first.length, 2, "tuple should have index and score");
    assert.strictEqual(typeof first[0], "number", "index should be a number");
    assert.strictEqual(typeof first[1], "number", "score should be a number");
  });

  it("throws when rerank is already in progress", async () => {
    await assert.rejects(() => {
      const result1 = pipeline.rerank(query, docs);
      const result2 = pipeline.rerank(query, docs);
      return Promise.all([result1, result2]);
    }, /Another reranking is already in progress/);
  });

  it("throws with incorrect query type", async () => {
    await assert.rejects(pipeline.rerank(123, docs), /Passed argument must be of type String./);
  });

  it("throws on non-array documents", async () => {
    await assert.rejects(
      pipeline.rerank(query, "not-an-array"),
      /Passed argument must be of type Array or TypedArray./,
    );
  });

  it("throws when documents contain non-string entries", async () => {
    await assert.rejects(
      pipeline.rerank(query, ["ok", 123]),
      /Passed array must contain only strings./,
    );
  });
});

describe("TextRerankPipeline initialization", () => {
  it("respects top_n in config", async () => {
    const limited = await TextRerankPipeline(RERANK_MODEL_PATH, {
      device: "CPU",
      config: { top_n: 2 },
    });
    const result = await limited.rerank(query, docs);
    assert.strictEqual(result.length, 2, "should return only top_n items");
  });

  it("throws when pipeline is initialized twice", async () => {
    const pipeline = await TextRerankPipeline(RERANK_MODEL_PATH);
    await assert.rejects(pipeline.init(), /TextRerankPipeline is already initialized/);
  });
});
