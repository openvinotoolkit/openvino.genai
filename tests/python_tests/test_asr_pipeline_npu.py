# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from utils.patch_pyav_for_servercore import install_av_stub_module_for_windows

install_av_stub_module_for_windows()

import numpy as np  # noqa: E402
import openvino_genai as ov_genai  # noqa: E402
import pytest  # noqa: E402

from utils.asr_utils.qwen3_asr import check_qwen3_asr_package, qwen3_asr_split_model_path  # noqa: E402
from utils.constants import NPUW_CPU_PROPERTIES  # noqa: E402
from utils.ov_genai_pipelines import should_skip_npuw_tests  # noqa: E402

pytestmark = pytest.mark.skipif(**should_skip_npuw_tests())


def test_qwen3_asr_split_npu():
    check_qwen3_asr_package()
    model_path = Path(qwen3_asr_split_model_path())
    assert (model_path / "openvino_audio_encoder_model.xml").is_file()

    properties = {
        "DEVICE_PROPERTIES": {"NPU": NPUW_CPU_PROPERTIES},
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 16,
    }
    with pytest.raises(RuntimeError, match="supports only the SDPA attention backend"):
        ov_genai.ASRPipeline(model_path, "NPU", ATTENTION_BACKEND="PA", **properties)

    pipe = ov_genai.ASRPipeline(model_path, "NPU", **properties)
    audio = np.zeros(66 * 16000, dtype=np.float32)
    result = pipe.generate(audio, language="English", max_new_tokens=4, ignore_eos=True)
    assert len(result.texts) == 1
    assert len(result.perf_metrics.asr_raw_metrics.encode_inference_durations) == 2
    assert result.perf_metrics.get_num_generated_tokens() == 8
