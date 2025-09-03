// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export class SchedulerConfig {
  /** a maximum number of tokens to batch
   * (in contrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
   * When ContinuousBatching is invoked from LLMPipeline (client scenario) by default max_num_batched_tokens is not limited.
   */
  max_num_batched_tokens: number = 256;

  /** total number of KV blocks available to scheduler logic */
  num_kv_blocks: number = 0;

  /** total size of KV cache in GB
   * When both num_kv_blocks and cache_size are set, num_kv_blocks is used.
   * When both num_kv_blocks and cache_size are equal to zero dynamic KV-cache allocation is turned on.
   */
  cache_size: number = 0;

  /** whether to split prompt / generate to different scheduling phases
   * Allows to process prompt partially in case when batch size is limited.
   * If dynamic_split_fuse is turned off any prompt that is longer than batch size will lead to error.
   */
  dynamic_split_fuse: boolean = true;

  constructor(init?: Partial<SchedulerConfig>) {
    if (init) {
      Object.assign(this, init);
    }
  }
}
