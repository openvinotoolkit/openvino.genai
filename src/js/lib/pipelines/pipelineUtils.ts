// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
