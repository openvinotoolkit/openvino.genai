# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

import pytest

from whowhatbench import model_loaders

GENAI_LOADERS = [
    "load_visual_text_genai_pipeline",
    "load_image2image_genai_pipeline",
    "load_inpainting_genai_pipeline",
    "load_embedding_genai_pipeline",
    "load_reranking_genai_pipeline",
]


def error_messages(caplog):
    # getMessage() applies the %-arguments, so a format string without a placeholder fails here.
    return [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]


@pytest.mark.parametrize("loader_name", GENAI_LOADERS)
def test_missing_openvino_genai_reports_the_import_error(monkeypatch, caplog, loader_name):
    monkeypatch.setitem(sys.modules, "openvino_genai", None)
    with pytest.raises(ImportError) as import_error:
        import openvino_genai  # noqa: F401

    with pytest.raises(SystemExit):
        getattr(model_loaders, loader_name)("model_dir")

    (message,) = error_messages(caplog)
    assert "Failed to import openvino_genai package" in message
    assert str(import_error.value) in message


def test_inpainting_fallback_reports_why_the_first_load_failed(monkeypatch, caplog):
    class FakeInpaintingPipeline:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls += 1
            if cls.calls == 1:
                raise ValueError("custom pipeline needs trust_remote_code")
            return object()

    monkeypatch.setattr("optimum.intel.openvino.OVPipelineForInpainting", FakeInpaintingPipeline, raising=False)
    monkeypatch.setattr(model_loaders, "disable_diffusers_model_progress_bar", lambda model: None)

    model_loaders.load_inpainting_model("model_id")

    assert FakeInpaintingPipeline.calls == 2
    (message,) = error_messages(caplog)
    assert "Failed to load inpainting pipeline" in message
    assert "custom pipeline needs trust_remote_code" in message
