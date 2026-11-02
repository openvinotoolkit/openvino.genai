# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class SpeechBenchmarkGenAITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench = Path(__file__).resolve().parents[2] / "tools" / "llm_bench"
        path_patch = mock.patch.object(sys, "path", [str(bench), *sys.path])
        path_patch.start()
        cls.addClassCleanup(path_patch.stop)
        cls.ov_utils = importlib.import_module("llm_bench_utils.ov_utils")
        importlib.import_module("llm_bench_utils.pt_utils")
        cls.speech = importlib.import_module("task.speech_to_text_generation")

    def test_genai_factory_does_not_initialize_hf_processor(self):
        module = self.ov_utils
        use_case = SimpleNamespace(ov_cls=module.UseCaseSpeech2Text.ov_cls, model_type="qwen3-asr")
        expected = (object(), object(), 0.0, True)
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(module, "is_genai_available", return_value=True),
                mock.patch.object(module.Qwen3ASROptimumPipeline, "init_model", side_effect=AssertionError("HF initialized")),
                mock.patch.object(module.AutoProcessor, "from_pretrained", side_effect=AssertionError("HF processor loaded")),
                mock.patch.object(module.AutoTokenizer, "from_pretrained", side_effect=AssertionError("HF tokenizer loaded")),
                mock.patch.object(module, "create_genai_speech_2_txt_model", return_value=expected) as factory,
            ):
                actual = module.create_speech_2_txt_model(
                    directory, "NPU", None, genai=True, config={}, use_case=use_case
                )
                self.assertEqual(actual, expected)
                self.assertEqual(factory.call_args.args, (Path(directory), "NPU", None))

    def test_genai_factory_returns_native_tokenizer(self):
        pipeline = mock.Mock()
        tokenizer = object()
        pipeline.get_tokenizer.return_value = tokenizer
        constructor = mock.Mock(return_value=pipeline)
        module = SimpleNamespace(ASRPipeline=constructor)
        with mock.patch.dict(sys.modules, {"openvino_genai": module}):
            actual = self.ov_utils.create_genai_speech_2_txt_model(
                Path("exported-model"), "npu", None, config={}
            )
        self.assertIs(actual[0], pipeline)
        self.assertIs(actual[1], tokenizer)
        self.assertTrue(actual[3])
        constructor.assert_called_once_with(Path("exported-model"), "NPU")

    def test_native_tokenizer_output_count(self):
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(size=7))
        self.assertEqual(self.speech._get_output_token_size(tokenizer, "transcript", True), 7)
        tokenizer.encode.assert_called_once_with("transcript")
        tokenizer.assert_not_called()

    def test_optimum_processor_output_count(self):
        token_row = mock.Mock()
        token_row.numel.return_value = 5
        tokenizer = mock.Mock(return_value={"input_ids": [token_row]})
        processor = SimpleNamespace(tokenizer=tokenizer)
        self.assertEqual(self.speech._get_output_token_size(processor, "transcript", False), 5)
        tokenizer.assert_called_once_with("transcript", return_tensors="pt")

    def test_optimum_tokenizer_output_count(self):
        token_row = SimpleNamespace(numel=lambda: 3)
        tokenizer = mock.Mock(spec=["__call__"], return_value={"input_ids": [token_row]})
        self.assertEqual(self.speech._get_output_token_size(tokenizer, "transcript", False), 3)
        tokenizer.assert_called_once_with("transcript", return_tensors="pt")


if __name__ == "__main__":
    unittest.main()
