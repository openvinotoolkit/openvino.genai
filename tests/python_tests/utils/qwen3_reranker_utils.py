# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


QWEN3_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def qwen3_reranker_format_queries(query, instruction=None):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"


def qwen3_reranker_format_document(document):
    return f"<Document>: {document}{QWEN3_RERANKER_SUFFIX}"


def qwen3_reranker_format_document_no_suffix(document):
    # Suffix is reserved and appended at the token level after truncation (mirrors GenAI's encode_qwen3).
    return f"<Document>: {document}"
