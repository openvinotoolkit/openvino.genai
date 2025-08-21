# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile

import numpy as np
import pytest

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

rng = np.random.default_rng(34231)


class TestTextToSpeechSample:
    def setup_class(self):
        # Create temporary binary file containing speaker embedding
        self.temp_speaker_embedding_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        # Generate 512 random float32 values
        data = rng.random(512, dtype=np.float32)
        # Write to file
        data.tofile(self.temp_speaker_embedding_file)
        self.temp_speaker_embedding_file.close()

    def teardown_class(self):
        # Remove temporary file
        if os.path.exists(self.temp_speaker_embedding_file.name):
            os.remove(self.temp_speaker_embedding_file.name)

    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.precommit
    @pytest.mark.parametrize("convert_model", ["tiny-random-SpeechT5ForTextToSpeech"], indirect=True)
    @pytest.mark.parametrize("input_prompt", ["Hello everyone"])
    def test_sample_text_to_speech(self, convert_model, input_prompt):
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'text2speech')
        cpp_command = [cpp_sample, convert_model, input_prompt, self.temp_speaker_embedding_file.name]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "speech_generation/text2speech.py")
        py_command = [sys.executable, py_script, convert_model, input_prompt,
                      "--speaker_embedding_file_path", self.temp_speaker_embedding_file.name]
        py_result = run_sample(py_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout, "C++ sample text2speech must be successfully completed"
        assert "Text successfully converted to audio file" in py_result.stdout, "Python sample text2speech must be successfully completed"


    @pytest.mark.speech_generation
    @pytest.mark.samples
    @pytest.mark.precommit
    @pytest.mark.parametrize("convert_model", ["tiny-random-SpeechT5ForTextToSpeech"], indirect=True)
    @pytest.mark.parametrize("input_prompt", ["Test text to speech without speaker embedding file"])
    def test_sample_text_to_speech_no_speaker_embedding_file(self, convert_model, input_prompt):
        # Run C++ sample
        # Example: text2speech spt5_model_dir "Hello everyone" --speaker_embedding_file_path xvector.bin
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'text2speech')
        cpp_command = [cpp_sample, convert_model, input_prompt]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "speech_generation/text2speech.py")
        py_command = [sys.executable, py_script, convert_model, input_prompt]
        py_result = run_sample(py_command)

        assert "Text successfully converted to audio file" in cpp_result.stdout, "C++ sample text2speech must be successfully completed"
        assert "Text successfully converted to audio file" in py_result.stdout, "Python sample text2speech must be successfully completed"
