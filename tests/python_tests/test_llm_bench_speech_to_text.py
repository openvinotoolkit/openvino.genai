# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import utils.patch_pyav_for_servercore as patch_pyav_for_servercore

patch_pyav_for_servercore.install_av_stub_module_for_windows()


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
                mock.patch.object(
                    module.Qwen3ASROptimumPipeline, "init_model", side_effect=AssertionError("HF initialized")
                ),
                mock.patch.object(
                    module.AutoProcessor, "from_pretrained", side_effect=AssertionError("HF processor loaded")
                ),
                mock.patch.object(
                    module.AutoTokenizer, "from_pretrained", side_effect=AssertionError("HF tokenizer loaded")
                ),
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
            actual = self.ov_utils.create_genai_speech_2_txt_model(Path("exported-model"), "npu", None, config={})
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

    def test_sensevoice_uses_reported_output_count_without_processor(self):
        self.assertEqual(self.speech._get_output_token_size(None, "transcript", False, 6), 6)

    def test_missing_processor_and_output_count_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not report the generated token count"):
            self.speech._get_output_token_size(None, "transcript", False)

    def test_native_qwen_npu_default_is_scoped_to_genai_and_device(self):
        classify = self.speech._uses_native_npu_qwen3_asr_decoder
        self.assertTrue(classify("NPU", True, "qwen3-asr"))
        self.assertTrue(classify("npu", True, "qwen3-asr"))
        self.assertFalse(classify("CPU", True, "qwen3-asr"))
        self.assertFalse(classify("NPU.0", True, "qwen3-asr"))
        self.assertFalse(classify("NPU", False, "qwen3-asr"))
        self.assertFalse(classify("NPU", True, "whisper"))

    def _run_native_generation(
        self, model_type, infer_count, native_qwen_npu, configured_max, tokenizer=None, token_count=1, iter_data=None
    ):
        config = SimpleNamespace(max_new_tokens=configured_max, max_length=777)
        metrics = SimpleNamespace(
            get_ttft=lambda: SimpleNamespace(mean=1),
            raw_metrics=SimpleNamespace(
                m_new_token_times=[1, 2],
                token_infer_durations=[1, 1],
                sampling_durations=[1, 1],
            ),
            asr_raw_metrics=SimpleNamespace(encode_inference_durations=[1], decode_inference_durations=[1]),
            get_tokenization_duration=lambda: SimpleNamespace(mean=1),
            get_features_extraction_duration=lambda: SimpleNamespace(mean=1),
            get_detokenization_duration=lambda: SimpleNamespace(mean=1),
            get_num_generated_tokens=lambda: token_count,
        )
        pipe = mock.Mock()
        pipe.get_generation_config.return_value = config
        pipe.generate.return_value = SimpleNamespace(texts=["transcript"], perf_metrics=metrics)
        if tokenizer is None:
            tokenizer = mock.Mock()
            tokenizer.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(size=1))
        memory = mock.Mock()
        memory.iter_stop_and_collect_data.return_value = {}
        input_param = {
            "pipe": pipe,
            "raw_speech": [0.0],
            "iter_idx": 0,
            "speech_idx": 0,
            "processor": tokenizer,
            "use_genai": True,
            "native_qwen3_asr_npu": native_qwen_npu,
            "speech_param": {},
            "mem_consumption": memory,
        }
        args = {"use_case": SimpleNamespace(model_type=model_type), "infer_count": infer_count}
        with (
            mock.patch.object(self.speech.metrics_print, "print_metrics"),
            mock.patch.object(self.speech.metrics_print, "print_generated"),
        ):
            result_data = iter_data if iter_data is not None else []
            self.speech.run_speech_2_txt_generation(input_param, args, {0: {}}, result_data)
        self.assertIs(pipe.generate.call_args.args[1], config)
        return config

    def test_sensevoice_detokenizer_only_uses_ctc_output_count(self):
        for token_count in (0, 7):
            with self.subTest(token_count=token_count):
                tokenizer = mock.Mock()
                tokenizer.encode.side_effect = AssertionError("SenseVoice must not encode its transcript")
                iter_data = []
                self._run_native_generation("sense-voice", None, False, 128, tokenizer, token_count, iter_data)
                tokenizer.encode.assert_not_called()
                self.assertEqual(iter_data[0]["output_size"], token_count)
                self.assertEqual(iter_data[0]["throughput_token_count"], token_count)

    def test_other_native_asr_models_still_count_retokenized_text(self):
        for model_type in ("qwen3-asr", "fun-asr", "whisper"):
            with self.subTest(model_type=model_type):
                tokenizer = mock.Mock()
                tokenizer.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(size=3))
                iter_data = []
                self._run_native_generation(model_type, None, False, 128, tokenizer, 7, iter_data)
                tokenizer.encode.assert_called_once_with("transcript")
                self.assertEqual(iter_data[0]["output_size"], 3)

    def test_qwen_omni_speech_uses_visual_language_pipeline(self):
        visual_language = importlib.import_module("task.visual_language_generation")
        args = {"is_omni_model": True, "infer_count": None}
        memory = mock.Mock()
        expected = object()
        with (
            mock.patch.object(self.speech, "get_speech_files", return_value=[{"media": "audio.wav"}]),
            mock.patch.object(self.speech.FW_UTILS["ov"], "create_speech_2_txt_model") as create_asr,
            mock.patch.object(
                visual_language, "run_visual_language_generation_benchmark", return_value=expected
            ) as run_vlm,
        ):
            result = self.speech.run_speech_2_txt_benchmark(Path("model"), "ov", "CPU", args, 1, memory)
        self.assertIs(result, expected)
        create_asr.assert_not_called()
        self.assertEqual(args["infer_count"], self.speech.DEFAULT_OUTPUT_TOKEN_SIZE)
        self.assertEqual(
            run_vlm.call_args.kwargs["input_list"],
            [{"prompt": self.speech.DEFAULT_SPEECH_PROMPT, "audio": "audio.wav"}],
        )

    def test_qwen_npu_omitted_limit_preserves_pipeline_configuration(self):
        unlimited = (1 << 64) - 1
        config = self._run_native_generation("qwen3-asr", None, True, unlimited)
        self.assertEqual(config.max_new_tokens, unlimited)
        self.assertEqual(config.max_length, 777)
        configured = self._run_native_generation("qwen3-asr", None, True, 128)
        self.assertEqual(configured.max_new_tokens, 128)
        self.assertEqual(configured.max_length, 777)

    def test_qwen_npu_explicit_limit_still_overrides_model_configuration(self):
        config = self._run_native_generation("qwen3-asr", 256, True, 128)
        self.assertEqual(config.max_new_tokens, 256)
        self.assertEqual(config.max_length, 777)

    def test_other_native_backends_keep_existing_implicit_limits(self):
        self.assertEqual(self._run_native_generation("qwen3-asr", None, False, 128).max_new_tokens, 1000)
        self.assertEqual(self._run_native_generation("fun-asr", None, False, 128).max_new_tokens, 1000)
        self.assertEqual(self._run_native_generation("whisper", None, False, 128).max_new_tokens, 400)

    def test_reference_backend_keeps_default_and_explicit_limits(self):
        memory = mock.Mock()
        memory.iter_stop_and_collect_data.return_value = {}
        token_row = SimpleNamespace(numel=lambda: 1)
        tokenizer = mock.Mock(return_value={"input_ids": [token_row]})
        pipe = mock.Mock(return_value={"text": "transcript", "num_generated_tokens": 1})
        hook = mock.Mock()
        hook.get_time_list.return_value = [0.1]
        hook.get_time_infer_list.return_value = [0.1]
        input_param = {
            "pipe": pipe,
            "raw_speech": [0.0],
            "iter_idx": 0,
            "speech_idx": 0,
            "processor": SimpleNamespace(tokenizer=tokenizer),
            "use_genai": False,
            "native_qwen3_asr_npu": False,
            "speech_param": {},
            "mem_consumption": memory,
        }
        args = {"use_case": SimpleNamespace(model_type="qwen3-asr"), "infer_count": None}
        with (
            mock.patch.object(self.speech, "asr_hook", hook),
            mock.patch.object(self.speech.model_utils, "setup_gen_config_use_custom_args", return_value={}),
            mock.patch.object(self.speech.metrics_print, "print_metrics"),
            mock.patch.object(self.speech.metrics_print, "print_generated"),
        ):
            self.speech.run_speech_2_txt_generation(input_param, args, {0: {}}, [])
            self.assertEqual(pipe.call_args.kwargs["generate_kwargs"]["max_new_tokens"], 1000)
            args["infer_count"] = 256
            self.speech.run_speech_2_txt_generation(input_param, args, {0: {}}, [])
            self.assertEqual(pipe.call_args.kwargs["generate_kwargs"]["max_new_tokens"], 256)

    def test_speech_benchmark_passes_native_npu_classification_to_generation(self):
        memory = mock.Mock()
        use_case = SimpleNamespace(model_type="qwen3-asr")
        args = {"prompt_index": None, "use_case": use_case, "infer_count": None}
        speech_files = [{"media": "audio.wav"}]
        observed = []

        def capture(input_param, *_):
            observed.append(input_param["native_qwen3_asr_npu"])

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(self.speech, "get_speech_files", return_value=speech_files),
                mock.patch.object(
                    self.speech.FW_UTILS["ov"],
                    "create_speech_2_txt_model",
                    return_value=(object(), object(), 0.0, True),
                ),
                mock.patch.object(self.speech.model_utils, "init_timestamp", side_effect=lambda *_: {0: {0: {}}}),
                mock.patch.object(self.speech.model_utils, "read_wav", return_value=[0.0]),
                mock.patch.object(self.speech.metrics_print, "print_average"),
                mock.patch.object(self.speech, "run_speech_2_txt_generation", side_effect=capture),
            ):
                self.speech.run_speech_2_txt_benchmark(Path(directory), "ov", "NPU", args, 0, memory)
                self.speech.run_speech_2_txt_benchmark(Path(directory), "ov", "CPU", args, 0, memory)
                (Path(directory) / "openvino_audio_encoder_model.xml").touch()
                self.speech.run_speech_2_txt_benchmark(Path(directory), "ov", "NPU", args, 0, memory)
        self.assertEqual(observed, [True, False, True])

    def test_vlm_genai_keeps_upstream_config_loading(self):
        module = self.ov_utils
        expected = object()
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(module.AutoConfig, "from_pretrained") as config,
                mock.patch.object(module, "is_genai_available", return_value=True),
                mock.patch.object(module, "create_genai_image_text_gen_model", return_value=expected) as factory,
            ):
                actual = module.create_image_text_gen_model(directory, "CPU", None, genai=True, config={})
                self.assertIs(actual, expected)
                config.assert_called_once_with(Path(directory), trust_remote_code=False)
                factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
