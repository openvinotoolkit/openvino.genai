# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the public OmniPipeline API.

Public surface under test:

    openvino_genai.OmniPipeline
    openvino_genai.OmniTalkerSpeechConfig
    openvino_genai.OmniSpeechStreamerBase

The pybind variant alias OmniSpeechStreamerVariant is intentionally not in
the smoke import — variant aliases are not re-exported in __init__.py per the
existing AudioStreamerVariant convention.

Two tiers, following the other pipeline suites:

1. Model-free tests — imports, config defaults, field round-trips, and dependency
   injection of user-defined VLMPipelineBase / TalkerBase children. The DI tests
   drive the real C++ OmniPipeline composition path against Python-defined mocks,
   exercising virtual dispatch, the text vs speech branching, and the constructor
   capability guards end to end.

2. Real-model tests against `optimum-intel-internal-testing/tiny-random-qwen3-omni`,
   covering the path-based constructor, text-only and speech generation, the
   ChatHistory overload, and the speaker APIs.

The real-model tier currently skips: the tiny checkpoint is a full omni model
(talker_config, code2wav_config, enable_audio_output), but exporting it needs
support that the pinned dependencies do not have yet. With transformers==5.0.0 the
export raises `AttributeError: 'Qwen3OmniMoeTalkerCodePredictorConfig' object has no
attribute 'use_sliding_window'` — the same failure `test_vlm_pipeline.py`'s
`test_qwen3_omni_vision_preprocess_modes_equivalence` xfails on. Beyond that, the
pinned optimum-intel exports no talker/code2wav submodels at all, so `Talker` has
nothing to load. `omni_model_path` detects both cases — plus transformers older than
4.57, which cannot export Qwen3-Omni at all — and skips with the reason; the tests
activate on their own once either dependency catches up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

import openvino_genai as ov_genai
from utils.constants import get_ov_cache_converted_models_dir

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.omni

OMNI_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-omni"

# Written by the Talker export; without them OmniPipeline's path ctor cannot build the speech stage.
TALKER_ARTIFACTS = (
    "openvino_talker_model.xml",
    "openvino_code_predictor_model.xml",
    "openvino_code2wav_model.xml",
)


class TestOmniPipelineImports:
    """Smoke tests: do the new public symbols exist at all?"""

    def test_omni_pipeline_imports(self) -> None:
        """All new public symbols must be importable from the top-level module."""
        from openvino_genai import (  # noqa: F401
            OmniPipeline,
            OmniSpeechStreamerBase,
            OmniTalkerSpeechConfig,
        )

    def test_old_symbol_is_gone(self) -> None:
        """The pre-migration OmniSpeechGenerationConfig symbol must not exist anymore.

        The rename to OmniTalkerSpeechConfig is a clean break with no deprecation
        alias. If this attribute survives, somebody re-exported the old name against
        the migration contract.
        """
        assert not hasattr(ov_genai, "OmniSpeechGenerationConfig"), (
            "OmniSpeechGenerationConfig was renamed to OmniTalkerSpeechConfig and must not be re-exported as an alias."
        )


class TestOmniTalkerSpeechConfig:
    """OmniTalkerSpeechConfig — defaults, validate(), AnyMap update, MRO shape."""

    def test_no_generation_config_inheritance(self) -> None:
        """Standalone struct: MRO must NOT contain GenerationConfig.

        The whole point of the migration is to break the historical
        `OmniSpeechGenerationConfig : public GenerationConfig` inheritance so a future
        omni model with a non-LLM talker doesn't drag GenerationConfig fields it
        doesn't use. pybind11 inserts its own `pybind11_object` into the MRO above
        `object`; that's a wrapping artifact, not the load-bearing inheritance check.
        """
        assert ov_genai.GenerationConfig not in ov_genai.OmniTalkerSpeechConfig.__mro__, (
            "OmniTalkerSpeechConfig must NOT inherit from GenerationConfig — that's the whole point of the migration."
        )

        cfg = ov_genai.OmniTalkerSpeechConfig()
        assert not isinstance(cfg, ov_genai.GenerationConfig), (
            "OmniTalkerSpeechConfig instances must not be GenerationConfig instances."
        )

    def test_defaults(self) -> None:
        """Default ctor exposes the speech-side fields with sane defaults."""
        cfg = ov_genai.OmniTalkerSpeechConfig()

        assert cfg.return_audio is True, "return_audio default must be True"
        assert cfg.speaker == "", "speaker default must be empty (model default)"
        assert cfg.audio_chunk_frames == 4, "audio_chunk_frames default must be 4"
        assert cfg.rng_seed == 0, "rng_seed default must be 0"
        # talker_*/cp_* sampling overrides are std::optional<T> — exposed as None when unset.
        assert cfg.talker_temperature is None
        assert cfg.talker_top_k is None
        assert cfg.talker_repetition_penalty is None
        assert cfg.cp_temperature is None
        assert cfg.cp_top_k is None

    def test_direct_field_assignment(self) -> None:
        """Direct field assignment sets the speech-side fields."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = False
        cfg.speaker = "any_voice_id"
        cfg.audio_chunk_frames = 2
        cfg.rng_seed = 7
        cfg.talker_temperature = 0.7
        cfg.talker_top_k = 20
        cfg.cp_temperature = 0.5
        cfg.cp_top_k = 10

        assert cfg.return_audio is False
        assert cfg.speaker == "any_voice_id"
        assert cfg.audio_chunk_frames == 2
        assert cfg.rng_seed == 7
        assert cfg.talker_temperature == pytest.approx(0.7)
        assert cfg.talker_top_k == 20
        assert cfg.cp_temperature == pytest.approx(0.5)
        assert cfg.cp_top_k == 10

    def test_max_new_tokens_field(self) -> None:
        """OmniTalkerSpeechConfig carries its own max_new_tokens (talker AR cap).

        Independent of GenerationConfig.max_new_tokens (which caps the thinker text
        decode). Both can be set simultaneously to different values when the caller
        constructs typed configs explicitly.
        """
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.max_new_tokens = 128
        assert cfg.max_new_tokens == 128


class TestOmniPipelineAccessors:
    """OmniPipeline getter/setter surface — methods must exist with the right signatures.

    No model is loaded here, so we only assert presence and signatures via the unbound
    method handle. End-to-end get/set behavior against injected pipelines is covered by
    TestOmniPipelineDependencyInjection.
    """

    def test_methods_exist(self) -> None:
        assert hasattr(ov_genai.OmniPipeline, "get_talker"), "OmniPipeline.get_talker() missing from public surface"

        # Speaker APIs live on the Talker, accessed via get_talker()
        for method in ("list_speakers", "get_speaker_embedding"):
            assert hasattr(ov_genai.TalkerBase, method), f"TalkerBase.{method}() missing from public surface"

    def test_speech_config_accessors_live_on_base(self) -> None:
        """get/set_speech_config are part of the TalkerBase interface every backend implements.

        TalkerBase declares them pure virtual, so the accessors and the property-bag generate()
        overload that seeds from them are part of the contract for every backend, not just the
        default Qwen3-Omni Talker, which stores the config itself.
        """
        for method in ("get_speech_config", "set_speech_config"):
            assert hasattr(ov_genai.TalkerBase, method), f"TalkerBase.{method}() missing from public surface"
            assert hasattr(ov_genai.Talker, method), f"Talker.{method}() missing from public surface"

    def test_talker_blob_ctor_signature(self) -> None:
        """Talker exposes the ModelsMap/device_mapping blob constructor (slide 10 spec).

        Calling with a bogus models_map must raise (missing submodels), not TypeError — that
        proves the overload resolves and reaches C++ construction rather than being absent.
        The disk-path constructor stays available alongside it.
        """
        empty_models_map: dict[str, object] = {}
        empty_device_mapping: dict[str, str] = {}
        with pytest.raises(Exception) as exc_info:
            ov_genai.Talker(empty_models_map, ov_genai.OmniTalkerSpeechConfig(), ".", empty_device_mapping)
        # Must not be a signature-resolution failure — the overload has to exist.
        assert not isinstance(exc_info.value, TypeError), f"blob ctor overload did not resolve: {exc_info.value}"


class RecordingTalker(ov_genai.TalkerBase):
    """User-defined talker that counts invocations. Result fields are read-only from Python, so
    dispatch is proven by the call count rather than by round-tripping audio.
    """

    def __init__(self, speakers: list[str] | None = None) -> None:
        super().__init__()
        self.generate_calls = 0
        self.last_return_audio: bool | None = None
        self._speakers = speakers if speakers is not None else ["default_voice"]
        self._speech_config = ov_genai.OmniTalkerSpeechConfig()

    def generate(
        self,
        vlm_result: ov_genai.VLMDecodedResults,
        talker_speech_config: ov_genai.OmniTalkerSpeechConfig,
        speech_streamer: ov_genai.OmniSpeechStreamerBase | None = None,
    ) -> ov_genai.TalkerResults:
        self.generate_calls += 1
        self.last_return_audio = talker_speech_config.return_audio
        return ov_genai.TalkerResults()

    # TalkerBase declares these pure virtual, so a Python backend has to store the config itself.
    def get_speech_config(self) -> ov_genai.OmniTalkerSpeechConfig:
        return self._speech_config

    def set_speech_config(self, config: ov_genai.OmniTalkerSpeechConfig) -> None:
        self._speech_config = config

    def list_speakers(self) -> list[str]:
        return self._speakers

    def get_speaker_embedding(self, name: str) -> ov.Tensor:
        return ov.Tensor(np.zeros((1, 1, 4), dtype=np.float32))


class RecordingVLM(ov_genai.VLMPipelineBase):
    """User-defined VLM (thinker) that counts generate() calls. Capability queries default to True
    so the ctor accepts the mock; tests flip them to exercise its guard assertions.
    """

    def __init__(self, audio_output: bool = True, hidden_states: bool = True) -> None:
        super().__init__()
        self.generate_calls = 0
        self.last_prompt: str | ov_genai.ChatHistory | None = None
        self.last_images: list[ov.Tensor] = []
        self.last_videos: list[ov.Tensor] = []
        self.last_audios: list[ov.Tensor] = []
        self.last_videos_metadata: list[ov_genai.VideoMetadata] = []
        self._audio_output = audio_output
        self._hidden_states = hidden_states

    def generate(  # noqa: PLR0913
        self,
        prompt: str | ov_genai.ChatHistory,
        images: list[ov.Tensor] | None = None,
        videos: list[ov.Tensor] | None = None,
        audios: list[ov.Tensor] | None = None,
        videos_metadata: list[ov_genai.VideoMetadata] | None = None,
        generation_config: ov_genai.GenerationConfig | None = None,
        streamer: ov_genai.StreamerBase | Callable[[str], bool] | None = None,
    ) -> ov_genai.VLMDecodedResults:
        self.generate_calls += 1
        self.last_prompt = prompt
        self.last_images = list(images or [])
        self.last_videos = list(videos or [])
        self.last_audios = list(audios or [])
        self.last_videos_metadata = list(videos_metadata or [])
        return ov_genai.VLMDecodedResults()

    def get_tokenizer(self) -> ov_genai.Tokenizer:
        raise RuntimeError("RecordingVLM.get_tokenizer() is not needed for the prompt-based path")

    def set_chat_template(self, chat_template: str) -> None:
        pass

    def get_generation_config(self) -> ov_genai.GenerationConfig:
        return ov_genai.GenerationConfig()

    def set_generation_config(self, config: ov_genai.GenerationConfig) -> None:
        pass

    def supports_hidden_states_collection(self) -> bool:
        return self._hidden_states

    def is_audio_output_enabled(self) -> bool:
        return self._audio_output


class TestCustomTalkerSubclass:
    """Users can define a TalkerBase child in Python and have its methods dispatched from C++."""

    def test_subclass_is_instantiable(self) -> None:
        """TalkerBase must expose a constructor so Python subclasses can be instantiated."""
        talker = RecordingTalker()
        assert isinstance(talker, ov_genai.TalkerBase)

    def test_overrides_are_dispatched(self) -> None:
        """Overridden list_speakers / get_speaker_embedding must call back into Python."""
        talker = RecordingTalker(speakers=["alice", "bob"])
        assert talker.list_speakers() == ["alice", "bob"]
        embedding = talker.get_speaker_embedding("alice")
        assert embedding.shape == [1, 1, 4]


class TestCustomVLMSubclass:
    """Users can define a VLMPipelineBase child in Python and have its methods dispatched from C++."""

    def test_subclass_is_instantiable(self) -> None:
        vlm = RecordingVLM()
        assert isinstance(vlm, ov_genai.VLMPipelineBase)

    def test_capability_overrides_are_dispatched(self) -> None:
        vlm = RecordingVLM(audio_output=False, hidden_states=False)
        assert vlm.is_audio_output_enabled() is False
        assert vlm.supports_hidden_states_collection() is False


@dataclass(frozen=True)
class InjectedOmni:
    """The DI pipeline together with the mocks it was built from, so tests can read the recorded calls."""

    pipeline: ov_genai.OmniPipeline
    vlm: RecordingVLM
    talker: RecordingTalker


@pytest.fixture
def injected_omni() -> InjectedOmni:
    """Function-scoped on purpose: the mocks carry per-test call counters that must start at zero."""
    vlm, talker = RecordingVLM(), RecordingTalker()
    return InjectedOmni(ov_genai.OmniPipeline(vlm, talker), vlm, talker)


def _talker_speech_config(return_audio: bool) -> ov_genai.OmniTalkerSpeechConfig:
    config = ov_genai.OmniTalkerSpeechConfig()
    config.return_audio = return_audio
    return config


class TestOmniPipelineDependencyInjection:
    """OmniPipeline(vlm, talker) accepts user-defined children and orchestrates them via C++."""

    def test_construct_from_python_children(self, injected_omni: InjectedOmni) -> None:
        """The DI constructor accepts a Python-defined VLM and Talker and hands them back verbatim."""
        assert injected_omni.pipeline.get_vlm() is injected_omni.vlm, (
            "get_vlm() must return the exact injected instance"
        )
        assert injected_omni.pipeline.get_talker() is injected_omni.talker, (
            "get_talker() must return the exact injected instance"
        )

    def test_speech_path_invokes_both_stages(self, injected_omni: InjectedOmni) -> None:
        """With return_audio=True, generate() drives the VLM then the talker via virtual dispatch."""
        result = injected_omni.pipeline.generate(
            "describe this", talker_speech_config=_talker_speech_config(return_audio=True)
        )

        assert isinstance(result, ov_genai.OmniDecodedResults)
        assert injected_omni.vlm.generate_calls == 1, "the injected VLM must be driven exactly once"
        assert injected_omni.talker.generate_calls == 1, "the injected talker must be driven exactly once"
        assert injected_omni.vlm.last_prompt == "describe this", "the prompt must reach the Python VLM unchanged"
        assert injected_omni.talker.last_return_audio is True

    def test_text_only_path_skips_talker(self, injected_omni: InjectedOmni) -> None:
        """With return_audio=False, the talker must not be invoked at all."""
        injected_omni.pipeline.generate("just text", talker_speech_config=_talker_speech_config(return_audio=False))

        assert injected_omni.vlm.generate_calls == 1
        assert injected_omni.talker.generate_calls == 0, "text-only generation must short-circuit the talker"

    def test_media_arguments_do_not_leak_between_calls(self, injected_omni: InjectedOmni) -> None:
        """Media passed to one generate() call must not reappear in the next call that omits it.

        The binding declares images/videos/audios/videos_metadata with empty-list defaults, which
        pybind11 materializes once per overload; a call that omits them must still see an empty
        sequence rather than whatever the previous call supplied.
        """
        media = ov.Tensor(np.zeros((1, 4, 4, 3), dtype=np.uint8))
        text_only = _talker_speech_config(return_audio=False)
        injected_omni.pipeline.generate(
            "with media",
            images=[media],
            videos=[media],
            videos_metadata=[ov_genai.VideoMetadata()],
            audios=[media],
            talker_speech_config=text_only,
        )
        assert len(injected_omni.vlm.last_images) == 1, "the first call must reach the VLM with its media"

        injected_omni.pipeline.generate("without media", talker_speech_config=text_only)

        assert injected_omni.vlm.generate_calls == 2
        assert len(injected_omni.vlm.last_images) == 0, "images leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_videos) == 0, "videos leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_audios) == 0, "audios leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_videos_metadata) == 0, "videos_metadata leaked from the previous call"

    def test_rejects_model_without_audio_output(self) -> None:
        """The constructor must reject a VLM whose is_audio_output_enabled() reports False."""
        vlm = RecordingVLM(audio_output=False)
        with pytest.raises(RuntimeError, match="is_audio_output_enabled"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())

    def test_rejects_backend_without_hidden_states(self) -> None:
        """The constructor must reject a VLM that cannot collect hidden states the talker needs."""
        vlm = RecordingVLM(hidden_states=False)
        with pytest.raises(RuntimeError, match="supports_hidden_states_collection"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())


def _export_tiny_omni_model(target_dir: Path) -> None:
    """Export the tiny Qwen3-Omni checkpoint to OpenVINO IR under ``target_dir``."""
    import openvino
    import openvino_tokenizers
    import transformers
    from huggingface_hub import snapshot_download
    from optimum.intel import OVModelForVisualCausalLM
    from utils.network import retry_request

    model_cached = snapshot_download(OMNI_MODEL_ID)  # required to avoid HF rate limits
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = retry_request(
        lambda: transformers.AutoProcessor.from_pretrained(
            model_cached,
            trust_remote_code=True,
            **align_with_optimum_cli,
        )
    )
    # No trust_remote_code here: test_vlm_pipeline.py grants it per model, and Qwen3-Omni is not on
    # that list — the architecture is native to transformers >= 4.57, so the export needs no custom code.
    model = retry_request(
        lambda: OVModelForVisualCausalLM.from_pretrained(
            model_cached, compile=False, device="CPU", export=True, load_in_8bit=False
        )
    )

    tokenizer = processor.tokenizer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = processor.chat_template
    tokenizer.save_pretrained(target_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, target_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, target_dir / "openvino_detokenizer.xml")

    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


@pytest.fixture(scope="module")
def omni_model_path() -> Path:
    """Path to an exported tiny Qwen3-Omni model, or skip when the pinned deps cannot produce one.

    Three distinct gaps are reported separately so a future failure points at the right dependency:
    a transformers version that cannot export Qwen3-Omni at all (checked before any download), the
    export itself raising, and the export succeeding without any talker submodels. Only the known
    transformers 5.0.x config bug is turned into a skip — any other export failure propagates so a
    real regression cannot hide behind a skipped test.
    """
    from optimum.utils.import_utils import is_transformers_version
    from utils.atomic_download import AtomicDownloadManager

    # Same gate as test_vlm_pipeline.py's _maybe_skip_unsupported_model_export, applied before the
    # snapshot download so an unsupported environment costs nothing.
    if is_transformers_version("<", "4.57.0"):
        pytest.skip(
            "ValueError: The current version of Transformers does not allow for the export of Qwen3-Omni. "
            "Minimum required is 4.57.0."
        )

    model_dir = get_ov_cache_converted_models_dir() / OMNI_MODEL_ID.replace("/", "_")
    manager = AtomicDownloadManager(model_dir)

    if not manager.is_complete() and not (model_dir / "openvino_language_model.xml").exists():
        try:
            manager.execute(_export_tiny_omni_model)
        except AttributeError as error:
            # Transformers 5.0 generated this config with an uninitialized use_sliding_window
            # reference, and the optimum-intel revision pinned by CI predates its workaround.
            # Same failure test_vlm_pipeline.py's Qwen3-Omni preprocessing test xfails on.
            message = str(error)
            if not (
                is_transformers_version(">=", "5.0")
                and is_transformers_version("<", "5.1")
                and "Qwen3OmniMoeTalkerCodePredictorConfig" in message
                and "use_sliding_window" in message
            ):
                raise
            logger.info("Tiny Qwen3-Omni export hit the known transformers 5.0.x config bug: %s", message)
            pytest.skip(f"Cannot export {OMNI_MODEL_ID} with the pinned dependencies: AttributeError: {message}")

    missing = [name for name in TALKER_ARTIFACTS if not (model_dir / name).exists()]
    if missing:
        pytest.skip(
            f"{OMNI_MODEL_ID} exported without the talker stage (missing {', '.join(missing)}); "
            "the pinned optimum-intel has no Qwen3-Omni talker/code2wav export support."
        )

    return model_dir


@pytest.fixture(scope="module")
def omni_pipe(omni_model_path: Path) -> ov_genai.OmniPipeline:
    """Pipeline built once per module — loading the model per test dominates the suite runtime.

    Reused across the real-model tests the way test_vlm_pipeline.py reuses ov_pipe_model. Safe to
    share: none of these tests enters chat mode or mutates the stored configs, so no state carries
    between them.
    """
    return ov_genai.OmniPipeline(str(omni_model_path), "CPU")


def _text_config(max_new_tokens: int = 10) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    return config


class TestOmniPipelineRealModel:
    """OmniPipeline driven by an exported tiny Qwen3-Omni checkpoint.

    Covers what the mock-based DI tests cannot: the path-based constructor, the real
    thinker/talker composition, and the speaker APIs backed by actual model data.
    """

    def test_constructs_from_path(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """The path-based ctor builds both stages from a single model directory."""
        assert omni_pipe.get_vlm() is not None, "path ctor must build a VLM stage"
        assert omni_pipe.get_talker() is not None, "path ctor must build a talker stage"

    def test_generate_text_only(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """With return_audio=False the pipeline decodes text and emits no waveform."""
        result = omni_pipe.generate(
            "Describe this.",
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=False),
        )

        assert isinstance(result, ov_genai.OmniDecodedResults)
        assert len(result.texts) == 1, "greedy decode must produce exactly one sequence"
        assert result.speech_result.waveforms == [], "return_audio=False must not produce waveforms"

    def test_generate_with_speech(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """With return_audio=True the talker produces a finite, non-empty waveform."""
        result = omni_pipe.generate(
            "Describe this.",
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=True),
        )

        assert len(result.speech_result.waveforms) == 1, "return_audio=True must produce one waveform"
        waveform = np.array(result.speech_result.waveforms[0].data, dtype=np.float32).reshape(-1)
        assert waveform.size > 0, "waveform must not be empty"
        assert np.isfinite(waveform).all(), "waveform must contain no NaN or Inf"

    def test_generate_from_chat_history(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """The ChatHistory overload accepts structured turns and leaves the caller's history intact."""
        history = ov_genai.ChatHistory()
        history.append({"role": "user", "content": "Describe this."})
        messages_before = history.get_messages()

        result = omni_pipe.generate(
            history,
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=False),
        )

        assert len(result.texts) == 1
        assert history.get_messages() == messages_before, "ChatHistory messages should not be mutated after generate."

    def test_speaker_apis(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """list_speakers() reports the checkpoint's voices and each resolves to an embedding."""
        speakers = omni_pipe.get_talker().list_speakers()

        assert speakers, "the checkpoint declares speakers, so list_speakers() must not be empty"
        embedding = omni_pipe.get_talker().get_speaker_embedding(speakers[0])
        assert embedding.get_size() > 0, f"speaker {speakers[0]!r} must resolve to a non-empty embedding"
