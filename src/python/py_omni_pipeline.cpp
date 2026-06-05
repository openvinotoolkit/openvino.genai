// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <optional>

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
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/pipeline_base.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::ChatHistory;
using ov::genai::GenerationConfig;
using ov::genai::OmniDecodedResults;
using ov::genai::OmniPipeline;
using ov::genai::OmniTalkerSpeechConfig;
using ov::genai::Qwen3OmniTalker;
using ov::genai::TalkerBase;
using ov::genai::VLMDecodedResults;
using ov::genai::VLMPipeline;

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

    :param speaker: Speaker name. Empty selects the model's default speaker. Available
        names are listed under `talker_config.speaker_id` in the model's `config.json`.
        Ignored when `speaker_embedding` is non-empty.
    :type speaker: str

    :param speaker_embedding: Optional explicit talker speaker embedding
        ([1, 1, talker_hidden_size], f32). Overrides `speaker` when non-empty.
    :type speaker_embedding: openvino.Tensor

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

    :return: VLMDecodedResults with `speech_outputs` populated when
        `talker_speech_config.return_audio` is True.
    :rtype: VLMDecodedResults
)";

auto omni_generate_history_docstring = R"(
    Generate text and (optionally) speech from a chat history. Same parameter semantics as the
    prompt overload.

    :param history: Chat history
    :type history: ChatHistory
)";

py::object call_omni_generate(OmniPipeline& pipe,
                              const std::string& prompt,
                              const std::vector<ov::Tensor>& images,
                              const std::vector<ov::Tensor>& videos,
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
        is Qwen3OmniTalker. Subclasses must override generate(), list_speakers(),
        get_speaker_embedding(), and is_available().)")
        .def("list_speakers", &TalkerBase::list_speakers)
        .def("get_speaker_embedding", &TalkerBase::get_speaker_embedding, py::arg("name"))
        .def("is_available", &TalkerBase::is_available);

    py::class_<Qwen3OmniTalker, TalkerBase, std::shared_ptr<Qwen3OmniTalker>>(m, "Qwen3OmniTalker",
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
                 return std::make_shared<Qwen3OmniTalker>(model_dir, device, properties);
             }),
             py::arg("model_dir"),
             "folder with Qwen3-Omni speech submodels + config.json",
             py::arg("device"),
             "device on which inference will be done",
             R"(
                Qwen3OmniTalker constructor.
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
        .def_readwrite("speaker", &OmniTalkerSpeechConfig::speaker)
        .def_readwrite("speaker_embedding", &OmniTalkerSpeechConfig::speaker_embedding)
        .def_readwrite("audio_chunk_frames", &OmniTalkerSpeechConfig::audio_chunk_frames)
        .def_readwrite("max_new_tokens", &OmniTalkerSpeechConfig::max_new_tokens)
        .def_readwrite("rng_seed", &OmniTalkerSpeechConfig::rng_seed)
        .def_readwrite("talker_temperature", &OmniTalkerSpeechConfig::talker_temperature)
        .def_readwrite("talker_top_k", &OmniTalkerSpeechConfig::talker_top_k)
        .def_readwrite("talker_repetition_penalty", &OmniTalkerSpeechConfig::talker_repetition_penalty)
        .def_readwrite("cp_temperature", &OmniTalkerSpeechConfig::cp_temperature)
        .def_readwrite("cp_top_k", &OmniTalkerSpeechConfig::cp_top_k)
        .def_readwrite("cp_repetition_penalty", &OmniTalkerSpeechConfig::cp_repetition_penalty)
        .def("update_generation_config",
             [](OmniTalkerSpeechConfig& config, const py::kwargs& kwargs) {
                 config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
             })
        .def("validate", &OmniTalkerSpeechConfig::validate);

    py::class_<OmniDecodedResults, VLMDecodedResults>(m, "OmniDecodedResults",
        R"(Omni-specific decoded results including speech outputs.

        Extends VLMDecodedResults with speech output waveforms for Qwen3-Omni models.

        Parameters:
        texts:            vector of resulting sequences (inherited from DecodedResults).
        scores:           scores for each sequence (inherited from DecodedResults).
        perf_metrics:     performance metrics (inherited from VLMDecodedResults).
        speech_outputs:   speech waveform tensors (one per result, present when return_audio=True).
        )")
        .def(py::init<>())
        .def_readonly("speech_outputs", &OmniDecodedResults::speech_outputs);

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

        .def(py::init([](VLMPipeline& vlm, std::shared_ptr<TalkerBase> talker) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 auto base = vlm.get_base();
                 py::gil_scoped_release rel;
                 return std::make_unique<OmniPipeline>(base, talker);
             }),
             py::arg("vlm"),
             "an already-loaded VLMPipeline whose backing pipeline is reused (no reload of weights)",
             py::arg("talker"),
             "speech-output backend (default Qwen3OmniTalker, or any TalkerBase subclass)",
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(),
             R"(
                OmniPipeline DI constructor.
                vlm (VLMPipeline): Already-loaded VLMPipeline. Its base is shared with the OmniPipeline.
                talker (TalkerBase): Speech-output backend. Use Qwen3OmniTalker(model_dir, device, **props)
                    for the default Qwen3-Omni stack, or pass your own TalkerBase subclass.
             )")

        .def(
            "generate",
            [](OmniPipeline& pipe,
               const std::string& prompt,
               const std::vector<ov::Tensor>& images,
               const std::vector<ov::Tensor>& videos,
               const std::vector<ov::Tensor>& audios,
               const std::optional<GenerationConfig>& text_config,
               const std::optional<OmniTalkerSpeechConfig>& talker_speech_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const pyutils::PyBindOmniSpeechStreamerVariant& speech_streamer)
                -> py::typing::Union<VLMDecodedResults> {
                // text_config defaults to None — fall back to the VLM's loaded
                // generation_config.json so a bare generate() call works without
                // forcing the user to construct a fully-populated GenerationConfig.
                GenerationConfig resolved_text_config =
                    text_config.has_value() ? text_config.value() : pipe.get_text_generation_config();
                OmniTalkerSpeechConfig resolved_talker_config =
                    talker_speech_config.value_or(OmniTalkerSpeechConfig{});
                return call_omni_generate(pipe,
                                          prompt,
                                          images,
                                          videos,
                                          audios,
                                          resolved_text_config,
                                          resolved_talker_config,
                                          streamer,
                                          speech_streamer);
            },
            py::arg("prompt"),
            py::arg("images") = std::vector<ov::Tensor>{},
            py::arg("videos") = std::vector<ov::Tensor>{},
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
               const std::vector<ov::Tensor>& audios,
               const std::optional<GenerationConfig>& text_config,
               const std::optional<OmniTalkerSpeechConfig>& talker_speech_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const pyutils::PyBindOmniSpeechStreamerVariant& speech_streamer)
                -> py::typing::Union<VLMDecodedResults> {
                GenerationConfig resolved_text_config =
                    text_config.has_value() ? text_config.value() : pipe.get_text_generation_config();
                OmniTalkerSpeechConfig resolved_talker_config =
                    talker_speech_config.value_or(OmniTalkerSpeechConfig{});
                return call_omni_generate_history(pipe,
                                                  history,
                                                  images,
                                                  videos,
                                                  audios,
                                                  resolved_text_config,
                                                  resolved_talker_config,
                                                  streamer,
                                                  speech_streamer);
            },
            py::arg("history"),
            py::arg("images") = std::vector<ov::Tensor>{},
            py::arg("videos") = std::vector<ov::Tensor>{},
            py::arg("audios") = std::vector<ov::Tensor>{},
            py::arg("text_config") = py::none(),
            py::arg("talker_speech_config") = py::none(),
            py::arg("streamer") = std::monostate(),
            py::arg("speech_streamer") = std::monostate(),
            omni_generate_history_docstring)

        .def("get_speaker_embedding",
             &OmniPipeline::get_speaker_embedding,
             py::arg("name"),
             R"(
                Return the precomputed talker speaker embedding for the named speaker.
                Tensor shape is [1, 1, talker_hidden_size], f32. Use to blend voices: weight-sum
                two named-speaker embeddings and pass the result via
                talker_speech_config.speaker_embedding.
                Raises if the model has no named speakers or `name` doesn't match.
             )")
        .def("list_speakers",
             &OmniPipeline::list_speakers,
             R"(
                List names of speakers available in the loaded model's talker_config.speaker_id.
                Returns an empty list when the model exposes no named speakers.
             )");
}
