// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { StreamingStatus, type TextStreamer } from "../utils.js";

export class PipelineInitialization {
  private initializing = false;
  private initialized = false;

  constructor(private readonly name: string) {}

  async run<T>(initialize: () => Promise<T>): Promise<T> {
    if (this.initialized) throw new Error(`${this.name} is already initialized`);
    if (this.initializing) throw new Error(`${this.name} is already initializing`);
    this.initializing = true;
    try {
      const pipeline = await initialize();
      this.initialized = true;
      return pipeline;
    } finally {
      this.initializing = false;
    }
  }
}

type PendingRead = {
  resolve: (result: IteratorResult<string>) => void;
  reject: (error: unknown) => void;
};

export function createTextStream(
  start: (
    streamer: TextStreamer,
    complete: (error: Error | null, text: () => string) => void,
  ) => void,
): AsyncIterableIterator<string> {
  const reads: PendingRead[] = [];
  const chunks: string[] = [];
  let completed = false;
  let cancelled = false;
  let error: unknown;
  let failed = false;
  let terminalDelivered = false;
  let fullText = "";
  let finish!: () => void;
  const completion = new Promise<void>((resolve) => {
    finish = resolve;
  });

  function pump() {
    if (cancelled) {
      chunks.length = 0;
      for (const read of reads.splice(0)) read.resolve({ done: true, value: "" });
      return;
    }
    while (reads.length > 0) {
      const chunk = chunks.shift();
      if (chunk !== undefined) {
        reads.shift()!.resolve({ done: false, value: chunk });
      } else if (completed) {
        const read = reads.shift()!;
        if (!terminalDelivered && failed) {
          read.reject(error);
        } else {
          read.resolve({ done: true, value: terminalDelivered ? "" : fullText });
        }
        terminalDelivered = true;
      } else {
        break;
      }
    }
  }

  const streamer: TextStreamer = (text) => {
    if (cancelled) return StreamingStatus.CANCEL;
    chunks.push(text);
    pump();
    return StreamingStatus.RUNNING;
  };

  start(streamer, (generationError, text) => {
    if (generationError) {
      error = generationError;
      failed = true;
    } else {
      try {
        fullText = text();
      } catch (conversionError) {
        error = conversionError;
        failed = true;
      }
    }
    completed = true;
    finish();
    pump();
  });

  async function cancel(): Promise<IteratorResult<string>> {
    cancelled = true;
    pump();
    await completion;
    if (failed && !terminalDelivered) {
      terminalDelivered = true;
      throw error;
    }
    return { done: true, value: "" };
  }

  return {
    next() {
      return new Promise<IteratorResult<string>>((resolve, reject) => {
        reads.push({ resolve, reject });
        pump();
      });
    },
    return: cancel,
    async throw(reason: unknown) {
      await cancel();
      throw reason;
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
}
