/* eslint-disable @typescript-eslint/no-misused-new */
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Base type for parsers that process complete text content at the end of generation.
 */
export type Parser = {
  /**
   * Parse complete text content at the end of generate call.
   *
   * This method processes the entire text content and extracts or modifies
   * information as needed. The results are stored in the provided message.
   *
   * @param message Message containing the text to parse and to store results
   */
  parse: (message: Record<string, unknown>) => void;
};

export interface ReasoningParserOptions {
  expectOpenTag?: boolean;
  keepOriginalContent?: boolean;
  openTag?: string;
  closeTag?: string;
}

export interface IReasoningParser extends Parser {
  new (options?: ReasoningParserOptions): IReasoningParser;
}

export interface IDeepSeekR1ReasoningParser extends IReasoningParser {
  new (): IDeepSeekR1ReasoningParser;
}
