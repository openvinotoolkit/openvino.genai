# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_JS_DIR
from test_utils import run_sample


class TestTextEmbeddingPipeline:
    @pytest.mark.rag
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["bge-small-en-v1.5"], indirect=True)
    def test_sample_text_embedding_pipeline(self, convert_model):
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "text_embeddings")
        cpp_command = [cpp_sample, convert_model, "Document 1", "Document 2"]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "rag/text_embeddings.py")
        py_command = [sys.executable, py_script, convert_model, "Document 1", "Document 2"]
        py_result = run_sample(py_command)

        # Run JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "rag/text_embeddings.js")
        js_command = ["node", js_sample, convert_model, "Document 1", "Document 2"]
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"


class TestTextRerankPipeline:
    @pytest.mark.rag
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["ms-marco-TinyBERT-L2-v2"], indirect=True)
    def test_sample_text_rerank_pipeline(self, convert_model):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "rag/text_rerank.py")
        document_1 = "Intel Core Ultra processors incorporate an AI-optimized\
architecture that supports new user experiences and the\
next wave of commercial applications."
        document_2 = "Intel Core Ultra processors are designed to\
provide enhanced performance and efficiency for a wide\
range of computing tasks."
        py_command = [sys.executable, py_script, convert_model, "What are the main features of Intel Core Ultra processors?", document_1, document_2]
        py_result = run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "text_rerank")

        cpp_command = [cpp_sample, convert_model, "What are the main features of Intel Core Ultra processors?", document_1, document_2]
        cpp_result = run_sample(cpp_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
