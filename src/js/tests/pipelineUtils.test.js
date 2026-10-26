// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createTextStream } from "../dist/pipelines/pipelineUtils.js";
import { StreamingStatus } from "../dist/utils.js";

function controlledStream() {
  let emit;
  let complete;
  const iterator = createTextStream((streamer, completion) => {
    emit = streamer;
    complete = completion;
  });
  return { iterator, emit, complete };
}

describe("Buffered text stream iterator", { timeout: 5000 }, () => {
  it("retains an error delivered before the first read and reports it once", async () => {
    const { iterator, complete } = controlledStream();
    const error = new Error("generation failed before next()");
    complete(error);

    await assert.rejects(iterator.next(), (received) => received === error);
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
  });

  it("drains buffered chunks before delivering a buffered error", async () => {
    const { iterator, emit, complete } = controlledStream();
    const error = new Error("generation failed after streaming");
    emit("first");
    emit("second");
    complete(error);

    assert.deepEqual(await iterator.next(), { done: false, value: "first" });
    assert.deepEqual(await iterator.next(), { done: false, value: "second" });
    await assert.rejects(iterator.next(), (received) => received === error);
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
  });

  it("serves concurrent pending reads in FIFO order", async () => {
    const { iterator, emit, complete } = controlledStream();
    const reads = [iterator.next(), iterator.next(), iterator.next()];
    assert.equal(emit("first"), StreamingStatus.RUNNING);
    assert.equal(emit("second"), StreamingStatus.RUNNING);
    complete(null, () => "firstsecond");

    assert.deepEqual(await Promise.all(reads), [
      { done: false, value: "first" },
      { done: false, value: "second" },
      { done: true, value: "firstsecond" },
    ]);
  });

  it("settles every pending and future read and delivers the final text only once", async () => {
    const { iterator, complete } = controlledStream();
    const reads = Array.from({ length: 4 }, () => iterator.next());
    complete(null, () => "full text");

    assert.deepEqual(await Promise.all(reads), [
      { done: true, value: "full text" },
      { done: true, value: "" },
      { done: true, value: "" },
      { done: true, value: "" },
    ]);
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
  });

  for (const started of [false, true]) {
    it(`return() waits for completion ${started ? "after streaming" : "before the first chunk"}`, async () => {
      let active = false;
      let emit;
      let complete;
      const start = (streamer, completion) => {
        assert.equal(active, false, "Previous generation must finish before reuse");
        active = true;
        emit = streamer;
        complete = () => {
          active = false;
          completion(null, () => "completed");
        };
      };
      const iterator = createTextStream(start);
      if (started) {
        emit("first");
        assert.deepEqual(await iterator.next(), { done: false, value: "first" });
      }
      const reads = [iterator.next(), iterator.next()];
      let returned = false;
      const closing = iterator.return().then((result) => {
        returned = true;
        return result;
      });

      await Promise.resolve();
      assert.equal(returned, false);
      assert.equal(active, true);
      assert.equal(emit("ignored"), StreamingStatus.CANCEL);
      assert.deepEqual(await Promise.all(reads), [
        { done: true, value: "" },
        { done: true, value: "" },
      ]);
      assert.deepEqual(await iterator.next(), { done: true, value: "" });
      complete();
      assert.deepEqual(await closing, { done: true, value: "" });
      assert.equal(active, false);
      assert.deepEqual(await iterator.next(), { done: true, value: "" });

      const reused = createTextStream(start);
      complete();
      assert.deepEqual(await reused.next(), { done: true, value: "completed" });
    });
  }

  it("propagates an error arriving while return() is waiting for completion", async () => {
    const { iterator, complete } = controlledStream();
    const error = new Error("generation failed while cancelling");
    const closing = assert.rejects(iterator.return(), (received) => received === error);
    complete(error);

    await closing;
    assert.deepEqual(await iterator.next(), { done: true, value: "" });
  });
});
