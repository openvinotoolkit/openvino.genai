// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "py_utils.hpp"
#include "bindings_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::ChunkStreamerBase;
using ov::genai::DecodedResults;
using ov::genai::GenerationConfig;
using ov::genai::OptionalWhisperGenerationConfig;
using ov::genai::PerfMetrics;
using ov::genai::RawSpeechInput;
using ov::genai::StreamerBase;
using ov::genai::StreamerVariant;
using ov::genai::StreamingStatus;
using ov::genai::Tokenizer;
using ov::genai::WhisperDecodedResultChunk;
using ov::genai::WhisperDecodedResults;
using ov::genai::WhisperGenerationConfig;
using ov::genai::WhisperPerfMetrics;
using ov::genai::WhisperPipeline;
using ov::genai::WhisperRawPerfMetrics;

namespace pyutils = ov::genai::pybind::utils;
namespace common_utils = ov::genai::common_bindings::utils;

namespace {

auto whisper_generate_docstring = R"(
    High level generate that receives raw speech as a vector of floats and returns decoded output.

    :param raw_speech_input: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
    :type raw_speech_input: list[float]

    :param generation_config: generation_config
    :type generation_config: WhisperGenerationConfig or a dict

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped.
                     Streamer supported for short-form audio (< 30 seconds) with `return_timestamps=False` only
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to WhisperGenerationConfig fields.
    :type : dict

    :return: return results in decoded form
    :rtype: WhisperDecodedResults
)";

auto whisper_decoded_results_docstring = R"(
    Structure to store resulting text outputs and scores.

    Parameters:
    texts:      vector of resulting sequences.
    scores:     scores for each sequence.
    metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
    shunks:     optional chunks of resulting sequences with timestamps
)";

auto whisper_decoded_result_chunk = R"(
    Structure to store decoded text with corresponding timestamps

    :param start_ts chunk start time in seconds
    :param end_ts   chunk end time in seconds
    :param text     chunk text
)";

auto whisper_generation_config_docstring = R"(
    WhisperGenerationConfig
    
    Whisper specific parameters:
    :param decoder_start_token_id: Corresponds to the ”<|startoftranscript|>” token.
    :type decoder_start_token_id: int

    :param pad_token_id: Padding token id.
    :type pad_token_id: int

    :param translate_token_id: Translate token id.
    :type translate_token_id: int

    :param transcribe_token_id: Transcribe token id.
    :type transcribe_token_id: int

    :param no_timestamps_token_id: No timestamps token id.
    :type no_timestamps_token_id: int

    :param prev_sot_token_id: Corresponds to the ”<|startofprev|>” token.
    :type prev_sot_token_id: int

    :param is_multilingual:
    :type is_multilingual: bool

    :param begin_suppress_tokens: A list containing tokens that will be suppressed at the beginning of the sampling process.
    :type begin_suppress_tokens: list[int]

    :param suppress_tokens: A list containing the non-speech tokens that will be suppressed during generation.
    :type suppress_tokens: list[int]

    :param language: Language token to use for generation in the form of <|en|>.
                     You can find all the possible language tokens in the generation_config.json lang_to_id dictionary.
    :type language: Optional[str]

    :param lang_to_id: Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
    :type lang_to_id: dict[str, int]

    :param task: Task to use for generation, either “translate” or “transcribe”
    :type task: int

    :param return_timestamps: If `true` the pipeline will return timestamps along the text for *segments* of words in the text.
                       For instance, if you get
                       WhisperDecodedResultChunk
                           start_ts = 0.5
                           end_ts = 1.5
                           text = " Hi there!"
                       then it means the model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                       Note that a segment of text refers to a sequence of one or more words, rather than individual words.
    :type return_timestamps: bool

    :param initial_prompt: Initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing
    window. Can be used to steer the model to use particular spellings or styles.

    Example:
      auto result = pipeline.generate(raw_speech);
      //  He has gone and gone for good answered Paul Icrom who...

      auto result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
      //  He has gone and gone for good answered Polychrome who...
    :type initial_prompt: Optional[str]

    :param hotwords:  Hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows.
    Can be used to steer the model to use particular spellings or styles.

    Example:
      auto result = pipeline.generate(raw_speech);
      //  He has gone and gone for good answered Paul Icrom who...

      auto result = pipeline.generate(raw_speech, ov::genai::hotwords("Polychrome"));
      //  He has gone and gone for good answered Polychrome who...
    :type hotwords: Optional[str]

    Generic parameters:
    max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                   max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
    max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
    min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
    ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
    eos_token_id:  token_id of <eos> (end of sentence)
    stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
    include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
    stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
    echo:           if set to true, the model will echo the prompt in the output.
    logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                    Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).

    repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
    presence_penalty: reduces absolute log prob if the token was generated at least once.
    frequency_penalty: reduces absolute log prob as many times as the token was generated.

    Beam search specific parameters:
    num_beams:         number of beams for beam search. 1 disables beam search.
    num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
    diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
    length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
        likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
        length_penalty < 0.0 encourages shorter sequences.
    num_return_sequences: the number of sequences to return for grouped beam search decoding.
    no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
    stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
        "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
        "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
        "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).

    Random sampling parameters:
    temperature:        the value used to modulate token probabilities for random sampling.
    top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
    do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
    num_return_sequences: the number of sequences to generate from a single prompt.
)";

auto streamer_base_docstring = R"(
    Base class for chunk streamers. In order to use inherit from from this class.
)";

auto raw_perf_metrics_docstring = R"(
    Structure with whisper specific raw performance metrics for each generation before any statistics are calculated.

    :param features_extraction_durations: Duration for each features extraction call.
    :type features_extraction_durations: list[MicroSeconds]
)";

auto perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param get_features_extraction_duration: Returns mean and standard deviation of features extraction duration in milliseconds
    :type get_features_extraction_duration: MeanStdPair

    :param whisper_raw_metrics: Whisper specific raw metrics
    :type WhisperRawPerfMetrics:
)";

OptionalWhisperGenerationConfig update_whisper_config_from_kwargs(const OptionalWhisperGenerationConfig& config,
                                                                  const py::kwargs& kwargs) {
    if (!config.has_value() && kwargs.empty())
        return std::nullopt;

    WhisperGenerationConfig res_config;
    if (config.has_value())
        res_config = *config;

    if (!kwargs.empty())
        res_config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));

    return res_config;
}

OPENVINO_SUPPRESS_DEPRECATED_START

class ConstructableChunkStreamer : public ChunkStreamerBase {
    bool put(int64_t token) override {
        PYBIND11_OVERRIDE(bool,               // Return type
                          ChunkStreamerBase,  // Parent class
                          put,                // Name of function in C++ (must match Python name)
                          token               // Argument(s)
        );
    }
    bool put_chunk(std::vector<int64_t> tokens) override {
        PYBIND11_OVERRIDE(bool,               // Return type
                          ChunkStreamerBase,  // Parent class
                          put_chunk,          // Name of function in C++ (must match Python name)
                          tokens              // Argument(s)
        );
    }
    StreamingStatus write(const std::vector<int64_t>& token) override {
        PYBIND11_OVERRIDE(StreamingStatus,    // Return type
                          ChunkStreamerBase,  // Parent class
                          write,              // Name of function in C++ (must match Python name)
                          token               // Argument(s)
        );
    }
    StreamingStatus write(int64_t token) override {
        PYBIND11_OVERRIDE(StreamingStatus,    // Return type
                          ChunkStreamerBase,  // Parent class
                          write,              // Name of function in C++ (must match Python name)
                          token               // Argument(s)
        );
    }
    void end() override {
        PYBIND11_OVERRIDE_PURE(void, ChunkStreamerBase, end);
    }
};

OPENVINO_SUPPRESS_DEPRECATED_END

py::object call_whisper_common_generate(WhisperPipeline& pipe,
                                        const RawSpeechInput& raw_speech_input,
                                        const OptionalWhisperGenerationConfig& config,
                                        const pyutils::PyBindStreamerVariant& py_streamer,
                                        const py::kwargs& kwargs) {
    // whisper config should initialized from generation_config.json in case of only kwargs provided
    // otherwise it would be initialized with default values which is unexpected for kwargs use case
    // if full config was provided then rely on it as a base config
    OptionalWhisperGenerationConfig base_config = config.has_value() ? config : pipe.get_generation_config();

    auto updated_config = update_whisper_config_from_kwargs(base_config, kwargs);

    ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
    ov::genai::WhisperDecodedResults res;
    {
        py::gil_scoped_release rel;
        res = pipe.generate(raw_speech_input, updated_config, streamer);
    }
    return py::cast(res);
}

}  // namespace

void init_whisper_pipeline(py::module_& m) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    py::class_<ChunkStreamerBase, ConstructableChunkStreamer, std::shared_ptr<ChunkStreamerBase>, StreamerBase>(
        m,
        "ChunkStreamerBase",
        streamer_base_docstring)  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("put",
             &ChunkStreamerBase::put,
             "Put is called every time new token is generated. Returns a bool flag to indicate whether generation "
             "should be stopped, if return true generation stops",
             py::arg("token"))
        .def("put_chunk",
             &ChunkStreamerBase::put_chunk,
             "put_chunk is called every time new token chunk is generated. Returns a bool flag to indicate whether "
             "generation should be stopped, if return true generation stops",
             py::arg("tokens"))
        .def("end",
             &ChunkStreamerBase::end,
             "End is called at the end of generation. It can be used to flush cache if your own streamer has one");
    OPENVINO_SUPPRESS_DEPRECATED_END

    // Binding for WhisperGenerationConfig
    py::class_<WhisperGenerationConfig, GenerationConfig>(m,
                                                          "WhisperGenerationConfig",
                                                          whisper_generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](const py::kwargs& kwargs) {
            return *update_whisper_config_from_kwargs(WhisperGenerationConfig(), kwargs);
        }))
        .def_readwrite("begin_suppress_tokens", &WhisperGenerationConfig::begin_suppress_tokens)
        .def_readwrite("suppress_tokens", &WhisperGenerationConfig::suppress_tokens)
        .def_readwrite("decoder_start_token_id", &WhisperGenerationConfig::decoder_start_token_id)
        .def_readwrite("pad_token_id", &WhisperGenerationConfig::pad_token_id)
        .def_readwrite("translate_token_id", &WhisperGenerationConfig::translate_token_id)
        .def_readwrite("transcribe_token_id", &WhisperGenerationConfig::transcribe_token_id)
        .def_readwrite("max_initial_timestamp_index", &WhisperGenerationConfig::max_initial_timestamp_index)
        .def_readwrite("no_timestamps_token_id", &WhisperGenerationConfig::no_timestamps_token_id)
        .def_readwrite("prev_sot_token_id", &WhisperGenerationConfig::prev_sot_token_id)
        .def_readwrite("is_multilingual", &WhisperGenerationConfig::is_multilingual)
        .def_readwrite("language", &WhisperGenerationConfig::language)
        .def_readwrite("lang_to_id", &WhisperGenerationConfig::lang_to_id)
        .def_readwrite("task", &WhisperGenerationConfig::task)
        .def_readwrite("return_timestamps", &WhisperGenerationConfig::return_timestamps)
        .def_readwrite("initial_prompt", &WhisperGenerationConfig::initial_prompt)
        .def_readwrite("hotwords", &WhisperGenerationConfig::hotwords)
        .def("update_generation_config", [](ov::genai::WhisperGenerationConfig& config, const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });

    py::class_<WhisperRawPerfMetrics>(m, "WhisperRawPerfMetrics", raw_perf_metrics_docstring)
        .def(py::init<>())
        .def_property_readonly("features_extraction_durations", [](const WhisperRawPerfMetrics& rw) {
            return common_utils::get_ms(rw, &WhisperRawPerfMetrics::features_extraction_durations);
        });

    py::class_<WhisperPerfMetrics, PerfMetrics>(m, "WhisperPerfMetrics", perf_metrics_docstring)
        .def(py::init<>())
        .def("get_features_extraction_duration", &WhisperPerfMetrics::get_features_extraction_duration)
        .def_readonly("whisper_raw_metrics", &WhisperPerfMetrics::whisper_raw_metrics);

    py::class_<WhisperDecodedResultChunk>(m, "WhisperDecodedResultChunk", whisper_decoded_result_chunk)
        .def(py::init<>())
        .def_readonly("start_ts", &WhisperDecodedResultChunk::start_ts)
        .def_readonly("end_ts", &WhisperDecodedResultChunk::end_ts)
        .def_property_readonly("text", [](WhisperDecodedResultChunk& chunk) {
            return pyutils::handle_utf8(chunk.text);
        });

    py::class_<WhisperDecodedResults>(m, "WhisperDecodedResults", whisper_decoded_results_docstring)
        .def_property_readonly("texts",
                               [](const WhisperDecodedResults& dr) -> py::typing::List<py::str> {
                                   return pyutils::handle_utf8((std::vector<std::string>)dr);
                               })
        .def_readonly("scores", &WhisperDecodedResults::scores)
        .def_readonly("chunks", &WhisperDecodedResults::chunks)
        .def_readonly("perf_metrics", &WhisperDecodedResults::perf_metrics)
        .def("__str__", [](const WhisperDecodedResults& dr) -> py::str {
            auto valid_utf8_strings = pyutils::handle_utf8((std::vector<std::string>)dr);
            py::str res;
            if (valid_utf8_strings.size() == 1)
                return valid_utf8_strings[0];

            for (size_t i = 0; i < valid_utf8_strings.size() - 1; i++) {
                res += py::str(std::to_string(dr.scores[i])) + py::str(": ") + valid_utf8_strings[i] + py::str("\n");
            }
            res += py::str(std::to_string(dr.scores.back())) + py::str(": ") +
                   valid_utf8_strings[valid_utf8_strings.size() - 1];
            return res;
        });

    py::class_<WhisperPipeline>(m, "WhisperPipeline", "Automatic speech recognition pipeline")
        .def(
            py::init([](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                return std::make_unique<WhisperPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
            }),
            py::arg("models_path"),
            "folder with openvino_model.xml and openvino_tokenizer[detokenizer].xml files",
            py::arg("device"),
            "device on which inference will be done",
            "openvino.properties map",
            R"(
            WhisperPipeline class constructor.
            models_path (os.PathLike): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU).
        )")

        .def(
            "generate",
            [](WhisperPipeline& pipe,
               const RawSpeechInput& raw_speech_input,
               const OptionalWhisperGenerationConfig& generation_config,
               const pyutils::PyBindStreamerVariant& streamer,
               const py::kwargs& kwargs) -> py::typing::Union<ov::genai::WhisperDecodedResults> {
                return call_whisper_common_generate(pipe, raw_speech_input, generation_config, streamer, kwargs);
            },
            py::arg("raw_speech_input"),
            "List of floats representing raw speech audio. "
            "Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.",
            py::arg("generation_config") = std::nullopt,
            "generation_config",
            py::arg("streamer") = std::monostate(),
            "streamer",
            (whisper_generate_docstring + std::string(" \n ") + whisper_generation_config_docstring).c_str())

        .def("get_tokenizer", &WhisperPipeline::get_tokenizer)
        .def("get_generation_config", &WhisperPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &WhisperPipeline::set_generation_config, py::arg("config"));
}
