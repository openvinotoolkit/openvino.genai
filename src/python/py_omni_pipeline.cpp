// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <optional>
#include <variant>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "omni/talker_speech_config_utils.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::ChatHistory;
using ov::genai::GenerationConfig;
using ov::genai::OmniDecodedResults;
using ov::genai::OmniPipeline;
using ov::genai::OmniTalkerSpeechConfig;
using ov::genai::Talker;
using ov::genai::TalkerBase;
using ov::genai::VideoMetadata;
using ov::genai::VLMDecodedResults;
using ov::genai::VLMPipeline;
using ov::genai::VLMPipelineBase;

namespace {

auto omni_talker_speech_config_docstring = R"(
    OmniTalkerSpeechConfig

    Standalone speech-side generation config for the Qwen3-Omni talker. Does NOT inherit
    from GenerationConfig — the thinker text decode is steered by a separate
    GenerationConfig argument to OmniPipeline.generate. This struct only carries fields
    the talker actually consumes:

    :param return_audio: Enable speech output. Default True. Set False to short-circuit
        the talker and produce text only.
    :type return_audio: bool

    :param speaker: Speaker identity — either a name (str) looked up in
        `talker_config.speaker_id`, or an explicit embedding tensor
        ([1, 1, talker_hidden_size], f32). Empty string selects the model's default.
    :type speaker: str | openvino.Tensor

    :param audio_chunk_frames: Number of codec frames accumulated before streaming each
        audio chunk. Must be >= 1. Each frame is 80ms of audio at 24 kHz (1920 samples).
    :type audio_chunk_frames: int

    :param max_new_tokens: Cap on talker AR steps. Independent of
        `text_config.max_new_tokens` (which caps the thinker text decode). The talker
        pipeline takes the min of this value and the model's
        `talker_config.talker_max_new_tokens`.
    :type max_new_tokens: int

    :param rng_seed: RNG seed for deterministic talker + CodePredictor sampling.
    :type rng_seed: int

    :param talker_temperature, talker_top_k, talker_repetition_penalty: Talker sampling
        overrides. None = keep the checkpoint default loaded from generation_config.json.
    :type talker_temperature: float | None
    :type talker_top_k: int | None
    :type talker_repetition_penalty: float | None

    :param cp_temperature, cp_top_k, cp_repetition_penalty: CodePredictor sampling
        overrides. Same semantics as talker_*.
    :type cp_temperature: float | None
    :type cp_top_k: int | None
    :type cp_repetition_penalty: float | None
)";

auto omni_pipeline_docstring = R"(
    OmniPipeline — Qwen3-Omni text + speech pipeline.

    Composes a VLM pipeline (text generation with hidden-state collection) with a Qwen3-Omni
    speech pipeline (Talker + CodePredictor + Code2Wav). Each `generate` call takes two
    configs: a `GenerationConfig text_config` (thinker) and an `OmniTalkerSpeechConfig
    talker_speech_config` (talker + speech). Speech generation is gated per-call by
    `talker_speech_config.return_audio`.

    Two construction paths:

      - Path-based: OmniPipeline(models_path, device, **properties) loads VLM and speech
        models from a single directory.

      - DI: OmniPipeline(vlm_pipeline, talker) reuses an externally-loaded VLMPipeline
        and a TalkerBase subclass for independent device choices or custom backends.

    Both ctors enforce that the loaded model is Qwen3-Omni capable (model_type == QWEN3_OMNI
    and enable_audio_output) — non-Omni models throw at construction time.
)";

auto omni_generate_prompt_docstring = R"(
    Generate text and (optionally) speech from a flat prompt.

    :param prompt: Input prompt
    :type prompt: str

    :param images: image tensors to be prepended to the prompt
    :type images: list[ov.Tensor]

    :param videos: video tensors to be prepended to the prompt
    :type videos: list[ov.Tensor]

    :param videos_metadata: metadata for each video (fps, frames_indices). Must be empty or have the same length as videos.
    :type videos_metadata: list[VideoMetadata]

    :param audios: audio tensors to be prepended to the prompt
    :type audios: list[ov.Tensor]

    :param text_config: thinker text-decode config. None = use the VLM's default
        GenerationConfig loaded from generation_config.json.
    :type text_config: GenerationConfig | None

    :param talker_speech_config: talker + speech-output config. None = a default-
        constructed OmniTalkerSpeechConfig (return_audio=True, model-default speaker).
    :type talker_speech_config: OmniTalkerSpeechConfig | None

    :param streamer: optional streamer for text tokens.
    :type streamer: Callable[[str], bool] | StreamerBase | None

    :param speech_streamer: optional callback or OmniSpeechStreamerBase to receive audio chunks
        during speech generation. Lambda receives ov.Tensor [1, 1, N_samples] and returns
        StreamingStatus (or bool/None).
    :type speech_streamer: Callable[[ov.Tensor], StreamingStatus | bool | None] | OmniSpeechStreamerBase | None

    :return: OmniDecodedResults with `speech_result.waveforms` populated when
        `talker_speech_config.return_audio` is True.
    :rtype: OmniDecodedResults
)";

auto omni_generate_history_docstring = R"(
    Generate text and (optionally) speech from a chat history. Same parameter semantics as the
    prompt overload.

    :param history: Chat history
    :type history: ChatHistory

    :param videos_metadata: metadata for each video (fps, frames_indices). Must be empty or have the same length as videos.
    :type videos_metadata: list[VideoMetadata]
)";

py::object call_omni_generate(OmniPipeline& pipe,
                              const std::string& prompt,
                              const std::vector<ov::Tensor>& images,
                              const std::vector<ov::Tensor>& videos,
                              const std::vector<VideoMetadata>& videos_metadata,
                              const std::vector<ov::Tensor>& audios,
                              const GenerationConfig& text_config,
                              const OmniTalkerSpeechConfig& talker_speech_config,
                              const pyutils::PyBindStreamerVariant& py_streamer,
                              const pyutils::PyBindOmniSpeechStreamerVariant& py_speech_streamer) {
    auto streamer = pyutils::pystreamer_to_streamer(py_streamer);
    auto speech_streamer = pyutils::py_speech_streamer_to_streamer(py_speech_streamer);
    OmniDecodedResults res;
    {
        py::gil_scoped_release rel;
        res = pipe.generate(prompt,
                            images,
                            videos,
                            videos_metadata,
                            audios,
                            text_config,
                            talker_speech_config,
                            streamer,
                            speech_streamer);
    }
    return py::cast(res);
}

py::object call_omni_generate_history(OmniPipeline& pipe,
                                      const ChatHistory& history,
                                      const std::vector<ov::Tensor>& images,
                                      const std::vector<ov::Tensor>& videos,
                                      const std::vector<VideoMetadata>& videos_metadata,
                                      const std::vector<ov::Tensor>& audios,
                                      const GenerationConfig& text_config,
                                      const OmniTalkerSpeechConfig& talker_speech_config,
                                      const pyutils::PyBindStreamerVariant& py_streamer,
                                      const pyutils::PyBindOmniSpeechStreamerVariant& py_speech_streamer) {
    auto streamer = pyutils::pystreamer_to_streamer(py_streamer);
    auto speech_streamer = pyutils::py_speech_streamer_to_streamer(py_speech_streamer);
    OmniDecodedResults res;
    {
        py::gil_scoped_release rel;
        res = pipe.generate(history,
                            images,
                            videos,
                            videos_metadata,
                            audios,
                            text_config,
                            talker_speech_config,
                            streamer,
                            speech_streamer);
    }
    return py::cast(res);
}

}  // namespace

void init_omni_pipeline(py::module_& m) {
    py::class_<TalkerBase, std::shared_ptr<TalkerBase>>(m, "TalkerBase",
        R"(Abstract speech-output backend for OmniPipeline.

        Subclass to plug a custom talker into OmniPipeline. The default implementation
        is Talker. Subclasses must override generate(), list_speakers(), and
        get_speaker_embedding().)")
        .def("list_speakers", &TalkerBase::list_speakers)
        .def("get_speaker_embedding", &TalkerBase::get_speaker_embedding, py::arg("name"));

    py::class_<Talker, TalkerBase, std::shared_ptr<Talker>>(m, "Talker",
        R"(Default OmniPipeline talker for the Qwen3-Omni Talker + CodePredictor + Code2Wav stack.

        Loads the speech submodels from a directory containing
        openvino_talker_model.xml, openvino_code_predictor_model.xml,
        openvino_code2wav_model.xml, plus the talker text-embedding and projection
        submodels and config.json.)")
        .def(py::init([](const std::filesystem::path& model_dir,
                         const std::string& device,
                         const py::kwargs& kwargs) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                 py::gil_scoped_release rel;
                 return std::make_shared<Talker>(model_dir, device, properties);
             }),
             py::arg("model_dir"),
             "folder with Qwen3-Omni speech submodels + config.json",
             py::arg("device"),
             "device on which inference will be done",
             R"(
                Talker constructor.
                model_dir (os.PathLike): Folder with Qwen3-Omni speech submodels + config.json.
                device (str): Device to run inference on (e.g., CPU, GPU).
                kwargs: Device properties.
             )");

    py::class_<OmniTalkerSpeechConfig>(m, "OmniTalkerSpeechConfig", omni_talker_speech_config_docstring)
        .def(py::init<>())
        .def(py::init<std::filesystem::path>(),
             py::arg("models_path"),
             "folder with config.json (talker_config) for default speaker resolution")
        .def_readwrite("return_audio", &OmniTalkerSpeechConfig::return_audio)
        .def_property(
            "speaker",
            [](const OmniTalkerSpeechConfig& self) -> py::object {
                if (std::holds_alternative<ov::Tensor>(self.speaker)) {
                    return py::cast(std::get<ov::Tensor>(self.speaker));
                }
                return py::cast(std::get<std::string>(self.speaker));
            },
            [](OmniTalkerSpeechConfig& self, py::object value) {
                if (py::isinstance<py::str>(value)) {
                    self.speaker = value.cast<std::string>();
                } else {
                    self.speaker = value.cast<ov::Tensor>();
                }
            },
            "Speaker identity: a name (str) looked up in talker_config.speaker_id, "
            "or an explicit embedding tensor ([1, 1, talker_hidden_size], f32).")
        .def_property(
            "speaker_embedding",
            [](const OmniTalkerSpeechConfig& self) -> py::object {
                if (std::holds_alternative<ov::Tensor>(self.speaker)) {
                    return py::cast(std::get<ov::Tensor>(self.speaker));
                }
                return py::none();
            },
            [](OmniTalkerSpeechConfig& self, py::object value) {
                if (value.is_none()) {
                    // Setting speaker_embedding to None reverts to the string default
                    if (!std::holds_alternative<std::string>(self.speaker)) {
                        self.speaker = std::string{};
                    }
                } else {
                    self.speaker = value.cast<ov::Tensor>();
                }
            },
            "Legacy alias. Reading returns the Tensor if speaker holds one, else None. "
            "Writing sets the Tensor alternative of the speaker variant.")
        .def_readwrite("audio_chunk_frames", &OmniTalkerSpeechConfig::audio_chunk_frames)
        .def_readwrite("max_new_tokens", &OmniTalkerSpeechConfig::max_new_tokens)
        .def_readwrite("rng_seed", &OmniTalkerSpeechConfig::rng_seed)
        .def_readwrite("talker_temperature", &OmniTalkerSpeechConfig::talker_temperature)
        .def_readwrite("talker_top_k", &OmniTalkerSpeechConfig::talker_top_k)
        .def_readwrite("talker_repetition_penalty", &OmniTalkerSpeechConfig::talker_repetition_penalty)
        .def_readwrite("cp_temperature", &OmniTalkerSpeechConfig::cp_temperature)
        .def_readwrite("cp_top_k", &OmniTalkerSpeechConfig::cp_top_k)
        .def_readwrite("cp_repetition_penalty", &OmniTalkerSpeechConfig::cp_repetition_penalty);

    py::class_<ov::genai::TalkerPerfMetrics>(m, "TalkerPerfMetrics",
        R"(Performance metrics for Talker speech generation.

        Parameters:
        num_generated_samples:  number of audio samples generated (waveform length).
        generation_time_ms:     total speech generation time in milliseconds.
        )")
        .def(py::init<>())
        .def_readonly("num_generated_samples", &ov::genai::TalkerPerfMetrics::num_generated_samples)
        .def_readonly("generation_time_ms", &ov::genai::TalkerPerfMetrics::generation_time_ms);

    py::class_<ov::genai::TalkerResults>(m, "TalkerResults",
        R"(Output of the talker speech backend. Holds speech waveforms and perf metrics.

        Parameters:
        waveforms:       speech waveform tensors (one per result, present when return_audio=True).
        perf_metrics:    speech-side perf metrics (TalkerPerfMetrics).
        )")
        .def(py::init<>())
        .def_readonly("waveforms", &ov::genai::TalkerResults::waveforms)
        .def_readonly("perf_metrics", &ov::genai::TalkerResults::perf_metrics);

    py::class_<OmniDecodedResults, VLMDecodedResults>(m, "OmniDecodedResults",
        R"(Omni-specific decoded results including speech outputs.

        Extends VLMDecodedResults with a TalkerResults that holds speech waveforms and perf metrics.

        Parameters:
        texts:           vector of resulting sequences (inherited from DecodedResults).
        scores:          scores for each sequence (inherited from DecodedResults).
        perf_metrics:    text-side perf metrics (inherited from VLMDecodedResults).
        speech_result:   TalkerResults with waveforms and perf_metrics.
        )")
        .def(py::init<>())
        .def_readonly("speech_result", &OmniDecodedResults::speech_result);

    py::class_<OmniPipeline>(m, "OmniPipeline", omni_pipeline_docstring)
        .def(py::init([](const std::filesystem::path& models_path,
                         const std::string& device,
                         const py::kwargs& kwargs) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                 py::gil_scoped_release rel;
                 return std::make_unique<OmniPipeline>(models_path, device, properties);
             }),
             py::arg("models_path"),
             "folder with exported Qwen3-Omni model files (VLM + speech)",
             py::arg("device"),
             "device on which inference will be done",
             R"(
                OmniPipeline path-based constructor.
                models_path (os.PathLike): Path to the folder with exported Qwen3-Omni model files.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
             )")

        .def(py::init([](const std::shared_ptr<VLMPipelineBase>& vlm,
                         const std::shared_ptr<TalkerBase>& talker) {
                 py::gil_scoped_release rel;
                 return std::make_unique<OmniPipeline>(vlm, talker);
             }),
             py::arg("vlm"),
             py::arg("talker"),
             R"(
                OmniPipeline dependency-injection constructor.
                Compose a pre-built VLM (thinker) and Talker (speech) so the two stages can use
                independent devices/properties, or so a custom TalkerBase subclass can be injected.
                vlm (VLMPipeline): Backing VLM pipeline. Must be a Qwen3-Omni-capable model loaded
                    with the continuous-batching backend (attention_backend=PA).
                talker (TalkerBase): Backing speech generator (default impl is Talker).
             )")

        .def(
            "generate",
            [](OmniPipeline& pipe,
               const std::string& prompt,
               const std::vector<ov::Tensor>& images,
               const std::vector<ov::Tensor>& videos,
               const std::vector<VideoMetadata>& videos_metadata,
               const std::vector<ov::Tensor>& audios,
               const std::optional<GenerationConfig>& text_config,
               const std::optional<OmniTalkerSpeechConfig>& talker_speech_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const pyutils::PyBindOmniSpeechStreamerVariant& speech_streamer)
                -> py::typing::Union<OmniDecodedResults> {
                GenerationConfig resolved_text_config = text_config.value_or(GenerationConfig{});
                OmniTalkerSpeechConfig resolved_talker_config = talker_speech_config.value_or(OmniTalkerSpeechConfig{});
                return call_omni_generate(pipe,
                                          prompt,
                                          images,
                                          videos,
                                          videos_metadata,
                                          audios,
                                          resolved_text_config,
                                          resolved_talker_config,
                                          streamer,
                                          speech_streamer);
            },
            py::arg("prompt"),
            py::arg("images") = std::vector<ov::Tensor>{},
            py::arg("videos") = std::vector<ov::Tensor>{},
            py::arg("videos_metadata") = std::vector<VideoMetadata>{},
            py::arg("audios") = std::vector<ov::Tensor>{},
            py::arg("text_config") = py::none(),
            py::arg("talker_speech_config") = py::none(),
            py::arg("streamer") = std::monostate(),
            py::arg("speech_streamer") = std::monostate(),
            omni_generate_prompt_docstring)

        .def(
            "generate",
            [](OmniPipeline& pipe,
               const ChatHistory& history,
               const std::vector<ov::Tensor>& images,
               const std::vector<ov::Tensor>& videos,
               const std::vector<VideoMetadata>& videos_metadata,
               const std::vector<ov::Tensor>& audios,
               const std::optional<GenerationConfig>& text_config,
               const std::optional<OmniTalkerSpeechConfig>& talker_speech_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const pyutils::PyBindOmniSpeechStreamerVariant& speech_streamer)
                -> py::typing::Union<OmniDecodedResults> {
                GenerationConfig resolved_text_config = text_config.value_or(GenerationConfig{});
                OmniTalkerSpeechConfig resolved_talker_config = talker_speech_config.value_or(OmniTalkerSpeechConfig{});
                return call_omni_generate_history(pipe,
                                                  history,
                                                  images,
                                                  videos,
                                                  videos_metadata,
                                                  audios,
                                                  resolved_text_config,
                                                  resolved_talker_config,
                                                  streamer,
                                                  speech_streamer);
            },
            py::arg("history"),
            py::arg("images") = std::vector<ov::Tensor>{},
            py::arg("videos") = std::vector<ov::Tensor>{},
            py::arg("videos_metadata") = std::vector<VideoMetadata>{},
            py::arg("audios") = std::vector<ov::Tensor>{},
            py::arg("text_config") = py::none(),
            py::arg("talker_speech_config") = py::none(),
            py::arg("streamer") = std::monostate(),
            py::arg("speech_streamer") = std::monostate(),
            omni_generate_history_docstring)

        .def("get_vlm",
             &OmniPipeline::get_vlm,
             R"(
                Return the underlying VLM (thinker) as a VLMPipelineBase. Useful for inspecting
                model metadata or reusing the same VLM across pipelines via the DI constructor.
             )")

        .def("get_talker",
             &OmniPipeline::get_talker,
             R"(
                Return the underlying TalkerBase. Speaker enumeration and embedding retrieval
                live here: pipe.get_talker().list_speakers(),
                pipe.get_talker().get_speaker_embedding(name).
             )");
}
