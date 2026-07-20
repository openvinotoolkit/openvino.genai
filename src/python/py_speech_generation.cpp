// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::GenerationConfig;
using ov::genai::PerfMetrics;
using ov::genai::SpeechGenerationConfig;
using ov::genai::SpeechGenerationPerfMetrics;
using ov::genai::Text2SpeechDecodedResults;
using ov::genai::Text2SpeechPipeline;

namespace pyutils = ov::genai::pybind::utils;

namespace {

auto speech_generation_config_docstring = R"(
    SpeechGenerationConfig

    Speech-generation specific parameters:
    :param minlenratio: minimum ratio of output length to input text length; prevents output that's too short.
    :type minlenratio: float

    :param maxlenratio: maximum ratio of output length to input text length; prevents excessively long outputs.
    :type maxlenratio: float

    :param threshold: probability threshold for stopping decoding; when output probability exceeds above this, generation will stop.
    :type threshold: float

    Kokoro-specific parameters:
    :param speed: speech speed multiplier.
    :type speed: float

    :param language: language code for Kokoro G2P (for example, "en-us" or "en-gb").
    :type language: str

    :param max_phoneme_length: maximum phoneme chunk length for Kokoro preprocessing.
    :type max_phoneme_length: int

    :param phonemize_fallback_model_dir: Optional OpenVINO fallback phonemizer model directory.
                                         This applies only to fallback during phonemize / G2P
                                         (graphemes to phonemes), before acoustic model inference.
                                         If set, this OpenVINO G2P fallback is used.
                                         If unset (None), espeak-ng G2P fallback is used.
                                         For kwargs-based APIs (`SpeechGenerationConfig(**kwargs)`,
                                         `update_generation_config(**kwargs)`, and pipeline kwargs),
                                         omit this key instead of passing None because kwargs-to-AnyMap
                                         conversion rejects None values.
    :type phonemize_fallback_model_dir: str | None

    Qwen3-TTS-specific parameters:
    :param speaker: predefined speaker name for Qwen3 CustomVoice variants.
    :type speaker: str

    :param instruct: optional instruction text that controls speaking style.
    :type instruct: str

    :param non_streaming_mode: Qwen3 prompt assembly mode.
                               ``True`` means non-streaming prompt assembly.
                               ``False`` means streaming-style prompt assembly.
    :type non_streaming_mode: bool

    :param subtalker_dosample: whether to sample residual code groups with Qwen3 subtalker.
    :type subtalker_dosample: bool

    :param subtalker_top_k: top-k parameter for Qwen3 subtalker sampling.
    :type subtalker_top_k: int

    :param subtalker_top_p: top-p parameter for Qwen3 subtalker sampling.
    :type subtalker_top_p: float

    :param subtalker_temperature: temperature parameter for Qwen3 subtalker sampling.
    :type subtalker_temperature: float

    Qwen3 Base voice-clone over ``generate``:
    :param voice_clone_ref_text: reference transcript for ICL mode.
    :type voice_clone_ref_text: str

    :param voice_clone_ref_audio: reference audio waveform tensor used to internally derive Qwen3 Base clone artifacts.
                           Expected shape: [T], [1, T], or [1, 1, T].
                           Expected dtype: float32.
                           Expected sample rate: 24000 Hz.
                           OV GenAI does not decode audio files or resample this tensor.
    :type voice_clone_ref_audio: openvino.Tensor

    :param voice_clone_ref_codec_ids: reference codec ids tensor for ICL mode, shape [T, G] or [1, T, G].
    :type voice_clone_ref_codec_ids: openvino.Tensor

)";

auto speech_generation_perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param num_generated_samples: Returns a number of generated samples in output
    :type num_generated_samples: int
)";

auto text_to_speech_decoded_results = R"(
    Structure that stores the result from the generate method, including a list of waveform tensors
    along with output sample rate and performance metrics

    :param speeches: a list of generated waveform tensors
    :type speeches: list

    :param output_sample_rate: sample rate of generated waveform tensors
    :type output_sample_rate: int

    :param perf_metrics: performance metrics
    :type perf_metrics: SpeechGenerationPerfMetrics

    :param speaker_embedding: Qwen3-TTS Base voice-clone speaker embedding used for generation.
                              Persist and pass it back as the ``speaker_embedding`` argument to reuse a
                              cloned voice without re-encoding reference audio. Empty for other backends.
    :type speaker_embedding: openvino.Tensor

    :param voice_clone_ref_codec_ids: Qwen3-TTS Base reference codec ids used for ICL-mode cloning.
                                      Persist and pass it back via the ``voice_clone_ref_codec_ids``
                                      property to reuse the reference prompt. Empty otherwise.
    :type voice_clone_ref_codec_ids: openvino.Tensor
)";

auto text_to_speech_generate_docstring = R"(
    Generates speeches based on input texts

    :param text_or_texts: input text(s) for which to generate speech
    :type text_or_texts: str or list[str]

    :param speaker_embedding optional speaker embedding tensor representing the unique characteristics of a speaker's
                             voice. If not provided for SpeechT5 TSS model, the 7306-th vector from the validation set of the
                             `Matthijs/cmu-arctic-xvectors` dataset is used by default. Kokoro backend requires callers
                             to prepare this tensor externally and pass it explicitly. Qwen3-TTS Base backend also
                             expects caller-provided speaker embedding (x-vector style cloning), while Qwen3-TTS
                             CustomVoice uses predefined speaker ids and does not require this tensor.
    :type speaker_embedding: openvino.Tensor or None

    :param properties: speech generation parameters specified as properties
    :type properties: dict

    :returns: raw audios of the input texts spoken in the specified speaker's voice;
              sample rate is provided via Text2SpeechDecodedResults.output_sample_rate
    :rtype: Text2SpeechDecodedResults
)";

SpeechGenerationConfig update_speech_generation_config_from_kwargs(const SpeechGenerationConfig& config,
                                                                   const py::kwargs& kwargs) {
    if (kwargs.empty())
        return config;

    SpeechGenerationConfig res_config = config;
    if (!kwargs.empty())
        res_config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));

    return res_config;
}

}  // namespace

void init_speech_generation_pipeline(py::module_& m) {
    // Binding for SpeechGenerationConfig
    py::class_<SpeechGenerationConfig, GenerationConfig>(m,
                                                         "SpeechGenerationConfig",
                                                         speech_generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](const py::kwargs& kwargs) {
            return update_speech_generation_config_from_kwargs(SpeechGenerationConfig(), kwargs);
        }))
        .def_readwrite("speed", &SpeechGenerationConfig::speed)
        .def_readwrite("minlenratio", &SpeechGenerationConfig::minlenratio)
        .def_readwrite("maxlenratio", &SpeechGenerationConfig::maxlenratio)
        .def_readwrite("threshold", &SpeechGenerationConfig::threshold)
        .def_readwrite("language", &SpeechGenerationConfig::language)
        .def_readwrite("max_phoneme_length", &SpeechGenerationConfig::max_phoneme_length)
        .def_readwrite("phonemize_fallback_model_dir", &SpeechGenerationConfig::phonemize_fallback_model_dir)
        .def_readwrite("speaker", &SpeechGenerationConfig::speaker)
        .def_readwrite("instruct", &SpeechGenerationConfig::instruct)
        .def_readwrite("non_streaming_mode", &SpeechGenerationConfig::non_streaming_mode)
        .def_readwrite("subtalker_dosample", &SpeechGenerationConfig::subtalker_dosample)
        .def_readwrite("subtalker_top_k", &SpeechGenerationConfig::subtalker_top_k)
        .def_readwrite("subtalker_top_p", &SpeechGenerationConfig::subtalker_top_p)
        .def_readwrite("subtalker_temperature", &SpeechGenerationConfig::subtalker_temperature)
        .def_readwrite("voice_clone_ref_text", &SpeechGenerationConfig::voice_clone_ref_text)
        .def_readwrite("voice_clone_ref_audio", &SpeechGenerationConfig::voice_clone_ref_audio)
        .def_readwrite("voice_clone_ref_codec_ids", &SpeechGenerationConfig::voice_clone_ref_codec_ids)
        .def("update_generation_config", [](ov::genai::SpeechGenerationConfig& config, const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });

    py::class_<SpeechGenerationPerfMetrics, PerfMetrics>(m,
                                                         "SpeechGenerationPerfMetrics",
                                                         speech_generation_perf_metrics_docstring)
        .def(py::init<>())
        .def_readonly("num_generated_samples", &SpeechGenerationPerfMetrics::num_generated_samples)
        .def_readonly("m_evaluated", &SpeechGenerationPerfMetrics::m_evaluated)
        .def_readonly("generate_duration", &SpeechGenerationPerfMetrics::generate_duration)
        .def_readonly("throughput", &SpeechGenerationPerfMetrics::throughput);

    py::class_<Text2SpeechDecodedResults>(m, "Text2SpeechDecodedResults", text_to_speech_decoded_results)
        .def(py::init<>())
        .def_readonly("speeches", &Text2SpeechDecodedResults::speeches)
        .def_readonly("output_sample_rate", &Text2SpeechDecodedResults::output_sample_rate)
        .def_readonly("perf_metrics", &Text2SpeechDecodedResults::perf_metrics)
        .def_readonly("speaker_embedding", &Text2SpeechDecodedResults::speaker_embedding)
        .def_readonly("voice_clone_ref_codec_ids", &Text2SpeechDecodedResults::voice_clone_ref_codec_ids);

    py::class_<Text2SpeechPipeline>(m, "Text2SpeechPipeline", "Text-to-speech pipeline")
        .def(
            py::init([](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                return std::make_unique<Text2SpeechPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
            }),
            py::arg("models_path"),
            "folder with tokenizer, encoder, decoder, postnet and vocoder .xml files",
            py::arg("device"),
            "device on which inference will be done",
            "openvino.properties map",
            R"(
            Text2SpeechPipeline class constructor.
            models_path (os.PathLike): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU).
        )")

        .def(
            "generate",
            [](Text2SpeechPipeline& pipe,
               const std::string& text,
               py::object speaker_embedding,
               const py::kwargs& kwargs) -> py::typing::Union<ov::genai::Text2SpeechDecodedResults> {
                ov::genai::Text2SpeechDecodedResults res;
                const ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    if (speaker_embedding.is_none()) {
                        res = pipe.generate(text, ov::Tensor(), properties);
                    } else {
                        const ov::Tensor& tensor = speaker_embedding.cast<ov::Tensor>();
                        res = pipe.generate(text, tensor, properties);
                    }
                }
                return py::cast(res);
            },
            py::arg("text"),
            "input text for which to generate speech.",
            py::arg("speaker_embedding") = py::none(),
            "vector representing the unique characteristics of a speaker's voice.",
            (text_to_speech_generate_docstring + std::string(" \n ") + speech_generation_config_docstring).c_str())

        .def(
            "generate",
            [](Text2SpeechPipeline& pipe,
               const std::vector<std::string>& texts,
               py::object speaker_embedding,
               const py::kwargs& kwargs) -> py::typing::Union<ov::genai::Text2SpeechDecodedResults> {
                ov::genai::Text2SpeechDecodedResults res;
                const ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                {
                    py::gil_scoped_release rel;
                    if (speaker_embedding.is_none()) {
                        res = pipe.generate(texts, ov::Tensor(), properties);
                    } else {
                        const ov::Tensor& tensor = speaker_embedding.cast<ov::Tensor>();
                        res = pipe.generate(texts, tensor, properties);
                    }
                }
                return py::cast(res);
            },
            py::arg("texts"),
            "a list of input texts for which to generate speeches.",
            py::arg("speaker_embedding") = py::none(),
            "vector representing the unique characteristics of a speaker's voice.",
            (text_to_speech_generate_docstring + std::string(" \n ") + speech_generation_config_docstring).c_str())

        .def("get_generation_config", &Text2SpeechPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &Text2SpeechPipeline::set_generation_config, py::arg("config"))
        .def("get_speaker_embedding_shape", &Text2SpeechPipeline::get_speaker_embedding_shape,
             "Get the expected speaker embedding shape for the loaded model. "
               "SpeechT5: Shape{1, 512}. Kokoro: Shape{510, 1, 256}. "
               "Qwen3 Base: Shape{1, 1, D}.");
}
