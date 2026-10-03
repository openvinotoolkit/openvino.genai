# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline loading for the tool_call_benchmark."""

import os

import openvino_genai


def load_pipeline(model_path, device="CPU", ov_config=None):
    """Load an OpenVINO GenAI pipeline from a local IR directory.

    Uses VLMPipeline for multi-part exports (``openvino_language_model.xml``,
    e.g. the OpenVINO org preconverted VLM repositories) and LLMPipeline
    for single-file optimum exports (``openvino_model.xml``).

    :param model_path: directory with the exported OpenVINO IR.
    :param device: OpenVINO device, e.g. "CPU" or "GPU".
    :param ov_config: optional dict of OpenVINO properties.
    :return: LLMPipeline or VLMPipeline instance.
    """
    vlm = os.path.exists(os.path.join(model_path, "openvino_language_model.xml"))
    kwargs = {}
    if ov_config:
        kwargs["properties"] = ov_config
    if vlm:
        return openvino_genai.VLMPipeline(model_path, device, **kwargs)
    return openvino_genai.LLMPipeline(model_path, device, **kwargs)
