// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "bindings_utils.hpp"
#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::ASRDecodedResultChunk;
using ov::genai::ASRDecodedResults;
using ov::genai::ASRGenerationConfig;
using ov::genai::ASRPerfMetrics;
using ov::genai::ASRPipeline;
using ov::genai::ASRRawPerfMetrics;
using ov::genai::AudioInputs;
using ov::genai::GenerationConfig;
using ov::genai::PerfMetrics;
using ov::genai::StreamerBase;
using ov::genai::StreamerVariant;
using ov::genai::Tokenizer;

namespace pyutils = ov::genai::pybind::utils;
namespace common_utils = ov::genai::common_bindings::utils;

namespace {

auto asr_generate_docstring = R"(
    High level generate that receives raw speech as a vector of floats and returns decoded output.

    :param audio_inputs: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
    :type audio_inputs: list[float]

    :param generation_config: generation_config
    :type generation_config: ASRGenerationConfig

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped.
                     Streamer supported for short-form audio (< 30 seconds) with `return_timestamps=False` only
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to ASRGenerationConfig fields.
    :type : dict

    :return: return results in decoded form
    :rtype: ASRDecodedResults
)";

auto asr_decoded_results_docstring = R"(
    Structure to store resulting text outputs and scores.

    Parameters:
    texts:              vector of resulting sequences.
    scores:             scores for each sequence.
    languages:          detected languages for the input audio(s), e.g. ["en"].
    perf_metrics:       performance metrics with tpot, ttft, etc. of type ov::genai::ASRPerfMetrics.
    chunks:             optional chunks of resulting sequences with timestamps
    words:              optional chunks of resulting words with timestamps
)";

auto asr_decoded_result_chunk_docstring = R"(
    Structure to store decoded text with corresponding timestamps

    :param start_ts chunk start time in seconds
    :param end_ts   chunk end time in seconds
    :param text     chunk text
    :param token_ids token ids corresponding to the chunk text
)";

auto asr_generation_config_docstring = R"(
    ASRGenerationConfig

    Common parameters:

    :param language: Language token to use for generation.
                     In the form of <|en|> for Whisper models. Can be set for multilingual models only.
                     In the form of English for Qwen3-ASR models.
    :type language: Optional[str]

    :param return_timestamps: Whether to return segment-level timestamps.
    :type return_timestamps: bool

    Whisper parameters:

    :param decoder_start_token_id: Corresponds to the "<|startoftranscript|>" token.
    :type decoder_start_token_id: int

    :param pad_token_id: Padding token id.
    :type pad_token_id: int

    :param translate_token_id: Translate token id.
    :type translate_token_id: int

    :param transcribe_token_id: Transcribe token id.
    :type transcribe_token_id: int

    :param prev_sot_token_id: Corresponds to the "<|startofprev|>" token.
    :type prev_sot_token_id: int

    :param no_timestamps_token_id: No timestamps token id.
    :type no_timestamps_token_id: int

    :param begin_suppress_tokens: A list containing tokens that will be suppressed at the beginning of the sampling process.
    :type begin_suppress_tokens: list[int]

    :param suppress_tokens: A list containing the non-speech tokens that will be suppressed during generation.
    :type suppress_tokens: list[int]

    :param max_initial_timestamp_index: Maximum initial timestamp index.
    :type max_initial_timestamp_index: int

    :param is_multilingual: Whether the model is multilingual.
    :type is_multilingual: bool

    :param task: Task to use for generation, either "translate" or "transcribe".
                 Can be set for multilingual models only.
    :type task: Optional[str]

    :param lang_to_id: Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
    :type lang_to_id: dict[str, int]

    :param word_timestamps: If `true` the pipeline will return word-level timestamps.
                            When enabled word_timestamps=True property should be passed to ASRPipeline constructor:
                            ASRPipeline("model_path", "CPU", word_timestamps=True)
    :type word_timestamps: bool

    :param alignment_heads: Encoder attention alignment heads used for word-level timestamps prediction.
                            Each pair represents (layer_index, head_index).
    :type alignment_heads: list[tuple[int, int]]

    :param initial_prompt: Initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing
                           window. Can be used to steer the model to use particular spellings or styles.

                           Example::

                             result = pipeline.generate(raw_speech)
                             #  He has gone and gone for good answered Paul Icrom who...

                             result = pipeline.generate(raw_speech, initial_prompt="Polychrome")
                             #  He has gone and gone for good answered Polychrome who...
    :type initial_prompt: Optional[str]

    :param hotwords: Hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to all processing windows.
                     Can be used to steer the model to use particular spellings or styles.

                     Example::

                       result = pipeline.generate(raw_speech)
                       #  He has gone and gone for good answered Paul Icrom who...

                       result = pipeline.generate(raw_speech, hotwords="Polychrome")
                       #  He has gone and gone for good answered Polychrome who...
    :type hotwords: Optional[str]

    Qwen3-ASR parameters:

    :param context: System prompt context prepended to Qwen3-ASR transcription requests.
    :type context: Optional[str]

    For generic generation parameters (max_length, max_new_tokens, num_beams, temperature, etc.)
    see GenerationConfig documentation.
)";

auto asr_raw_perf_metrics_docstring = R"(
    Structure with ASR specific raw performance metrics for each generation before any statistics are calculated.

    :param features_extraction_durations: Duration for each features extraction call.
    :type features_extraction_durations: list[MicroSeconds]

    :param word_level_timestamps_processing_durations: Duration for each word-level timestamps processing call.
    :type word_level_timestamps_processing_durations: list[MicroSeconds]

    :param encode_inference_durations: Duration for each encoder inference call.
    :type encode_inference_durations: list[MicroSeconds]

    :param decode_inference_durations: Duration for each decoder inference call during token generation.
    :type decode_inference_durations: list[MicroSeconds]
)";

auto asr_perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param get_features_extraction_duration: Returns mean and standard deviation of features extraction duration in milliseconds
    :type get_features_extraction_duration: MeanStdPair

    :param get_word_level_timestamps_processing_duration: Returns mean and standard deviation of word-level timestamps processing duration in milliseconds
    :type get_word_level_timestamps_processing_duration: MeanStdPair

    :param get_encode_inference_duration: Returns mean and standard deviation of encoder inference duration in milliseconds
    :type get_encode_inference_duration: MeanStdPair

    :param get_decode_inference_duration: Returns mean and standard deviation of decoder inference duration in milliseconds
    :type get_decode_inference_duration: MeanStdPair

    :param asr_raw_metrics: ASR specific raw metrics
    :type ASRRawPerfMetrics:
)";

std::optional<ASRGenerationConfig> update_asr_config_from_kwargs(const std::optional<ASRGenerationConfig>& config,
                                                                 const py::kwargs& kwargs) {
    if (!config.has_value() && kwargs.empty())
        return std::nullopt;

    ASRGenerationConfig res_config;
    if (config.has_value())
        res_config = *config;

    if (!kwargs.empty())
        res_config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));

    return res_config;
}

py::object call_asr_common_generate(ASRPipeline& pipe,
                                    const AudioInputs& audio_inputs,
                                    const std::optional<ASRGenerationConfig>& config,
                                    const pyutils::PyBindStreamerVariant& py_streamer,
                                    const py::kwargs& kwargs) {
    // ASR config should be initialized from generation_config.json in case of only kwargs provided
    // otherwise it would be initialized with default values which is unexpected for kwargs use case
    // if full config was provided then rely on it as a base config
    std::optional<ASRGenerationConfig> base_config = config.has_value() ? config : pipe.get_generation_config();

    auto updated_config = update_asr_config_from_kwargs(base_config, kwargs);

    ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
    ASRDecodedResults res;
    {
        py::gil_scoped_release rel;
        res = pipe.generate(audio_inputs, updated_config, streamer);
    }
    return py::cast(res);
}

}  // namespace

void init_asr_pipeline(py::module_& m) {
    // Binding for ASRGenerationConfig
    py::class_<ASRGenerationConfig, GenerationConfig>(m, "ASRGenerationConfig", asr_generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](const py::kwargs& kwargs) {
            return *update_asr_config_from_kwargs(ASRGenerationConfig(), kwargs);
        }))
        .def_readwrite("begin_suppress_tokens", &ASRGenerationConfig::begin_suppress_tokens)
        .def_readwrite("suppress_tokens", &ASRGenerationConfig::suppress_tokens)
        .def_readwrite("decoder_start_token_id", &ASRGenerationConfig::decoder_start_token_id)
        .def_readwrite("pad_token_id", &ASRGenerationConfig::pad_token_id)
        .def_readwrite("translate_token_id", &ASRGenerationConfig::translate_token_id)
        .def_readwrite("transcribe_token_id", &ASRGenerationConfig::transcribe_token_id)
        .def_readwrite("max_initial_timestamp_index", &ASRGenerationConfig::max_initial_timestamp_index)
        .def_readwrite("no_timestamps_token_id", &ASRGenerationConfig::no_timestamps_token_id)
        .def_readwrite("prev_sot_token_id", &ASRGenerationConfig::prev_sot_token_id)
        .def_readwrite("is_multilingual", &ASRGenerationConfig::is_multilingual)
        .def_readwrite("language", &ASRGenerationConfig::language)
        .def_readwrite("lang_to_id", &ASRGenerationConfig::lang_to_id)
        .def_readwrite("task", &ASRGenerationConfig::task)
        .def_readwrite("return_timestamps", &ASRGenerationConfig::return_timestamps)
        .def_readwrite("word_timestamps", &ASRGenerationConfig::word_timestamps)
        .def_readwrite("alignment_heads", &ASRGenerationConfig::alignment_heads)
        .def_readwrite("initial_prompt", &ASRGenerationConfig::initial_prompt)
        .def_readwrite("hotwords", &ASRGenerationConfig::hotwords)
        .def_readwrite("context", &ASRGenerationConfig::context)
        .def("update_generation_config", [](ASRGenerationConfig& config, const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });

    py::class_<ASRRawPerfMetrics>(m, "ASRRawPerfMetrics", asr_raw_perf_metrics_docstring)
        .def(py::init<>())
        .def_property_readonly("features_extraction_durations",
                               [](const ASRRawPerfMetrics& rw) {
                                   return common_utils::get_ms(rw, &ASRRawPerfMetrics::features_extraction_durations);
                               })
        .def_property_readonly(
            "word_level_timestamps_processing_durations",
            [](const ASRRawPerfMetrics& rw) {
                return common_utils::get_ms(rw, &ASRRawPerfMetrics::word_level_timestamps_processing_durations);
            })
        .def_property_readonly("encode_inference_durations",
                               [](const ASRRawPerfMetrics& rw) {
                                   return common_utils::get_ms(rw, &ASRRawPerfMetrics::encode_inference_durations);
                               })
        .def_property_readonly("decode_inference_durations", [](const ASRRawPerfMetrics& rw) {
            return common_utils::get_ms(rw, &ASRRawPerfMetrics::decode_inference_durations);
        });

    py::class_<ASRPerfMetrics, PerfMetrics>(m, "ASRPerfMetrics", asr_perf_metrics_docstring)
        .def(py::init<>())
        .def("get_features_extraction_duration", &ASRPerfMetrics::get_features_extraction_duration)
        .def("get_word_level_timestamps_processing_duration",
             &ASRPerfMetrics::get_word_level_timestamps_processing_duration)
        .def("get_encode_inference_duration", &ASRPerfMetrics::get_encode_inference_duration)
        .def("get_decode_inference_duration", &ASRPerfMetrics::get_decode_inference_duration)
        .def_readonly("asr_raw_metrics", &ASRPerfMetrics::asr_raw_metrics);

    py::class_<ASRDecodedResultChunk>(m, "ASRDecodedResultChunk", asr_decoded_result_chunk_docstring)
        .def(py::init<>())
        .def_readonly("start_ts", &ASRDecodedResultChunk::start_ts)
        .def_readonly("end_ts", &ASRDecodedResultChunk::end_ts)
        .def_property_readonly("text",
                               [](const ASRDecodedResultChunk& chunk) {
                                   return pyutils::handle_utf8(chunk.text);
                               })
        .def_readonly("token_ids", &ASRDecodedResultChunk::token_ids);

    py::class_<ASRDecodedResults>(m, "ASRDecodedResults", asr_decoded_results_docstring)
        .def_property_readonly("texts",
                               [](const ASRDecodedResults& dr) -> py::typing::List<py::str> {
                                   return pyutils::handle_utf8((std::vector<std::string>)dr);
                               })
        .def_readonly("scores", &ASRDecodedResults::scores)
        .def_readonly("languages", &ASRDecodedResults::languages)
        .def_readonly("chunks", &ASRDecodedResults::chunks)
        .def_readonly("words", &ASRDecodedResults::words)
        .def_readonly("perf_metrics", &ASRDecodedResults::perf_metrics)
        .def("__repr__", [](const ASRDecodedResults& dr) -> py::str {
            auto valid_utf8_strings = pyutils::handle_utf8((std::vector<std::string>)dr);
            if (valid_utf8_strings.size() == 1)
                return valid_utf8_strings[0];

            py::str res;
            for (size_t i = 0; i < valid_utf8_strings.size() - 1; i++) {
                res += py::str(std::to_string(dr.scores[i])) + py::str(": ") + valid_utf8_strings[i] + py::str("\n");
            }
            res += py::str(std::to_string(dr.scores.back())) + py::str(": ") +
                   valid_utf8_strings[valid_utf8_strings.size() - 1];
            return res;
        });

    py::class_<ASRPipeline>(m, "ASRPipeline", "Automatic speech recognition pipeline")
        .def(
            py::init([](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                return std::make_unique<ASRPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
            }),
            py::arg("models_path"),
            "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
            py::arg("device"),
            "device on which inference will be done",
            R"(
            ASRPipeline class constructor.
            models_path (os.PathLike): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU).
        )")

        .def(
            "generate",
            [](ASRPipeline& pipe,
               const AudioInputs& audio_inputs,
               const std::optional<ASRGenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const py::kwargs& kwargs) -> py::typing::Union<ASRDecodedResults> {
                return call_asr_common_generate(pipe, audio_inputs, generation_config, streamer, kwargs);
            },
            py::arg("audio_inputs"),
            "List of floats representing raw speech audio. "
            "Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.",
            py::arg("generation_config") = std::nullopt,
            "generation_config",
            py::arg("streamer") = std::monostate(),
            "streamer",
            (asr_generate_docstring + std::string(" \n ") + asr_generation_config_docstring).c_str())

        .def("get_tokenizer", &ASRPipeline::get_tokenizer)
        .def("get_generation_config", &ASRPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &ASRPipeline::set_generation_config, py::arg("config"));
}
