// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "openvino/genai/omni/channel.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/streamer_base.hpp"
#include "openvino/genai/omni/text_source_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/parsers.hpp"
#include "py_utils.hpp"
#include "openvino/genai/json_container.hpp"

namespace py = pybind11;

using ov::genai::CallbackTypeVariant;
using ov::genai::OmniChannel;
using ov::genai::OmniStreamerBase;
using ov::genai::OmniTextSourceBase;
using ov::genai::StreamingStatus;
using ov::genai::TextStreamer;
using ov::genai::TextParserStreamer;
using ov::genai::IncrementalParser;
using ov::genai::JsonContainer;
using ov::genai::Tokenizer;

namespace pyutils = ov::genai::pybind::utils;

namespace {

auto streamer_base_docstring =  R"(
    Base class for streamers. In order to use inherit from from this class and implement write and end methods.
)";

auto text_streamer_docstring =  R"(
TextStreamer is used to decode tokens into text and call a user-defined callback function.

tokenizer: Tokenizer object to decode tokens into text.
callback: User-defined callback function to process the decoded text, callback should return either boolean flag or StreamingStatus.
detokenization_params: AnyMap with detokenization parameters, e.g. ov::genai::skip_special_tokens(...)
)";

auto text_parser_streamer_docstring = R"(
Base class for text streamers which works with parsed messages. In order to use inherit from this class and implement write method which takes a dict as input parameter.

tokenizer: Tokenizer object to decode tokens into text.
parsers: vector of IncrementalParser to process the text stream incrementally.
)";

class ConstructableStreamer: public StreamerBase {
    StreamingStatus write(int64_t token) override {
        PYBIND11_OVERRIDE_PURE(
            StreamingStatus,  // Return type
            StreamerBase,  // Parent class
            write,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    StreamingStatus write(const std::vector<int64_t>& token) override {
        PYBIND11_OVERRIDE(
            StreamingStatus,  // Return type
            StreamerBase,  // Parent class
            write,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    void end() override {
        PYBIND11_OVERRIDE_PURE(void, StreamerBase, end);
    }
};

class ConstructableTextParserStreamer: public TextParserStreamer {
public:
    using TextParserStreamer::TextParserStreamer;  // inherit base constructors

    StreamingStatus write(JsonContainer& message) override {
        // Since C++ calls function with JsonContainer while python override expects py::dict, 
        // this function is a wrapper to call Python implementation of 'write' with py::dict
        py::gil_scoped_acquire acquire;

        // A subclass that does not implement 'write': fail with an error instead of calling a null override.
        py::function override = py::get_override(this, "write");
        if (!override) {
            throw py::type_error("TextParserStreamer subclasses must implement write(message: dict) -> StreamingStatus");
        }

        py::dict message_py = pyutils::json_container_to_py_object(message);

        // Call python implementation which accepts py::dict instead of JsonContainer
        // And convert back the resulting message back to JsonContainer
        auto res = override(message_py);
        message = pyutils::py_object_to_json_container(message_py);
        
        return res.cast<StreamingStatus>();
    }
};

class ConstructableOmniSpeechStreamer : public ov::genai::OmniSpeechStreamerBase {
    ov::genai::StreamingStatus write(const ov::Tensor& audio_chunk) override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(ov::genai::StreamingStatus, ov::genai::OmniSpeechStreamerBase, write, audio_chunk);
    }
    void end() override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(void, ov::genai::OmniSpeechStreamerBase, end);
    }
};

auto omni_speech_streamer_base_docstring = R"(
    Base class for audio streamers. Inherit and implement write() and end()
    to receive audio chunks during speech generation.

    write(audio_chunk: ov.Tensor) -> StreamingStatus:
        Called with each audio chunk [1, 1, N_samples] float32 PCM at 24kHz.
        Return StreamingStatus.RUNNING to continue or STOP/CANCEL to halt.

    end():
        Called when speech generation completes (always, even on early stop).
)";

// Step payload of the thinker -> talker bridge, as a dict on the Python side. The documented
// keys are converted with their declared types so a Python writer can pass an empty list where the
// generic converter would produce an empty ov::Any; unknown keys go through the generic converter.
ov::AnyMap omni_step_from_py(const py::dict& step) {
    ov::AnyMap data;
    for (const auto& item : step) {
        const std::string key = py::cast<std::string>(item.first);
        const py::object value = py::reinterpret_borrow<py::object>(item.second);
        if (key == ov::genai::omni_stream::tokens.name()) {
            data[key] = value.cast<std::vector<int64_t>>();
        } else if (key == ov::genai::omni_stream::hidden_states.name()) {
            data[key] = value.cast<std::vector<ov::Tensor>>();
        } else if (key == ov::genai::omni_stream::prompt_length.name()) {
            data[key] = value.cast<size_t>();
        } else {
            py::dict single;
            single[item.first] = item.second;
            const ov::AnyMap converted = pyutils::py_object_to_any_map(single);
            data.insert(converted.begin(), converted.end());
        }
    }
    return data;
}

// Trampoline so a Python subclass can be the sink of VLMPipeline.generate(..., omni_streamer=...).
class ConstructableOmniStreamer : public OmniStreamerBase {
    StreamingStatus write(const ov::AnyMap& data) override {
        py::gil_scoped_acquire acquire;
        py::function override = py::get_override(this, "write");
        if (!override) {
            throw py::type_error("OmniStreamerBase subclasses must implement write(step: dict) -> StreamingStatus");
        }
        return override(pyutils::any_map_to_py_object(data)).cast<StreamingStatus>();
    }
    void end() override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(void, OmniStreamerBase, end);
    }
    void abandon() override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(void, OmniStreamerBase, abandon);
    }
};

// Trampoline so a Python subclass can be the source of TalkerBase.generate(text_source, ...).
class ConstructableOmniTextSource : public OmniTextSourceBase {
    std::optional<ov::AnyMap> read() override {
        py::gil_scoped_acquire acquire;
        py::function override = py::get_override(this, "read");
        if (!override) {
            throw py::type_error("OmniTextSourceBase subclasses must implement read() -> dict | None");
        }
        py::object step = override();
        if (step.is_none()) {
            return std::nullopt;
        }
        return omni_step_from_py(step.cast<py::dict>());
    }
    bool truncated() const override {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(bool, OmniTextSourceBase, truncated);
    }
};

auto omni_streamer_base_docstring = R"(
    Write end of the thinker -> talker bridge. VLMPipeline.generate(..., omni_streamer=...) calls
    write() once per decode step with that step's tokens and thinker hidden states, so a talker
    reading the other end can start speaking before text generation finishes.

    Inherit and implement write(), end() and abandon(). OmniChannel is the built-in implementation
    that also implements OmniTextSourceBase; OmniPipeline builds one itself when
    GenerationConfig.stream_text2speech is set, so a custom one is for driving VLMPipeline and
    Talker yourself.

    write(step: dict) -> StreamingStatus:
        Called once per VLM decode step. Ignore unknown keys. Known keys:
          "tokens": list[int] -- token ids produced by this step; continuous batching may split a
              long prefill across writes when the prompt exceeds batch capacity, and each decode
              step carries one token.
          "hidden_states": list[ov.Tensor] -- f32 [1, 1, hidden_size] thinker hidden states,
              co-indexed with the token stream by absolute position across the whole stream, not
              within one payload; the token stream ends one entry longer. A hidden state is written
              in the same step as, or before, the token it predicted, so once a step carries the
              first generated token, every prompt hidden state the producer will provide has
              arrived. Continuous batching keeps hidden states for a bounded number of leading
              positions only, so this stream can end short of the prompt. Present only when the
              backend collects hidden states (continuous batching with
              GenerationConfig.return_omni_outputs).
          "prompt_length": int -- total number of prompt tokens, repeated in each step. Because
              continuous batching may split prefill when the prompt exceeds batch capacity, an early
              step may not contain the assistant header; wait until this many tokens have arrived
              before locating the assistant segment.
        Return StreamingStatus.RUNNING to continue, STOP or TOOL_CALL_STOP to stop, CANCEL to
        cancel the VLM.

    end():
        Called exactly once when the VLM stage ends, including on an early stop or cancel.
        Must release any reader blocked in OmniTextSourceBase.read().

    abandon():
        Called instead of end() when the VLM stage raised: what was written is a prefix of a
        response that will never be completed. Same obligation to release readers.

    write() and end()/abandon() are called sequentially from the thread running the VLM decode
    loop. This is a preview API and is subject to change.
)";

auto omni_text_source_base_docstring = R"(
    Read end of the thinker -> talker bridge. Talker.generate(text_source, ...) calls read() in a
    loop to pull the VLM's decode steps as they are produced instead of waiting for a finished
    VLMDecodedResults.

    Inherit and implement read() and truncated(). What a producer passed to OmniStreamerBase.write()
    is what read() returns, with the same keys and in the same order.

    read() -> dict | None:
        Take the next VLM step, blocking until one arrives or the stream ends. Return None once the
        write end has called end() and every queued step has been read; None is final. Must not
        block forever on a VLM that stopped early or raised: end() or abandon() is always called
        on the write end and must release the reader.

    truncated() -> bool:
        Whether the producer called abandon() instead of end(). Only meaningful once read() has
        returned None.

    This is a preview API and is subject to change.
)";

auto omni_channel_docstring = R"(
    Built-in thinker -> talker bridge: the VLM writes decode steps in, the talker reads them out,
    so speech generation runs while text generation is still going. Implements both
    OmniStreamerBase and OmniTextSourceBase.

    OmniPipeline creates one per generate() call when GenerationConfig.stream_text2speech is set.
    Construct one directly only when driving VLMPipeline.generate(..., omni_streamer=channel) and
    Talker.generate(channel, ...) yourself, typically on two threads.

    Steps are queued in arrival order and are never dropped or coalesced. The queue is unbounded,
    so a talker slower than the thinker costs memory rather than stalling text generation.

    The reader closes its own end with finish_reading() when it stops listening normally and with
    fail_reading() when it fails. Both drop the queue and are idempotent, with fail_reading() taking
    precedence. After finish_reading(), write() keeps returning RUNNING so text generation can
    finish; after fail_reading(), it returns STOP.

    The writer closes with end() or abandon(), and the difference reaches the reader as
    truncated(). abandon() also drops the queue.

    Safe to use from two threads, one writing and one reading. read() releases the GIL while it
    waits. This is a preview API and is subject to change.
)";

} // namespace

void init_streamers(py::module_& m) {
    auto streamer = py::class_<StreamerBase, ConstructableStreamer, std::shared_ptr<StreamerBase>>(m, "StreamerBase", streamer_base_docstring)  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("write",
            [](StreamerBase& self, std::variant<int64_t, std::vector<int64_t>> token) {
               if (auto _token = std::get_if<int64_t>(&token)) {
                   return self.write(*_token);
               } else {
                   auto tokens = std::get<std::vector<int64_t>>(token);
                   return self.write(tokens);
               }
            },
            "Write is called every time new token or vector of tokens is decoded. Returns a StreamingStatus flag to indicate whether generation should be stopped or cancelled",
            py::arg("token"))
        .def("end", &StreamerBase::end, "End is called at the end of generation. It can be used to flush cache if your own streamer has one");

    py::class_<TextStreamer, std::shared_ptr<TextStreamer>, StreamerBase>(m, "TextStreamer", text_streamer_docstring)
        .def(py::init([](const Tokenizer& tokenizer, std::function<CallbackTypeVariant(std::string)> callback, const std::map<std::string, py::object>& detokenization_params) {
            return std::make_shared<TextStreamer>(tokenizer, callback, pyutils::properties_to_any_map(detokenization_params));
        }),
        py::arg("tokenizer"),
        py::arg("callback"),
        py::arg("detokenization_params") = ov::AnyMap({}))

        .def("write",
            [](TextStreamer& self, std::variant<int64_t, std::vector<int64_t>> token) {
                if (auto _token = std::get_if<int64_t>(&token)) {
                    return self.write(*_token);
                } else {
                    auto tokens = std::get<std::vector<int64_t>>(token);
                    return self.write(tokens);
                }
            },
            py::arg("token"))
            .def("end", &TextStreamer::end);
        
    py::class_<TextParserStreamer, ConstructableTextParserStreamer, std::shared_ptr<TextParserStreamer>, TextStreamer>(m, "TextParserStreamer", text_parser_streamer_docstring)
        .def(py::init(
                 // TextParserStreamer itself: write() is pure virtual, so it has to be subclassed.
                 [](const Tokenizer&, std::vector<std::shared_ptr<IncrementalParser>>) -> std::shared_ptr<TextParserStreamer> {
                     throw py::type_error("TextParserStreamer is abstract: subclass it and implement write(message: dict)");
                 },
                 // A Python subclass.
                 [](const Tokenizer& tokenizer, std::vector<std::shared_ptr<IncrementalParser>> parsers) {
                     return std::make_shared<ConstructableTextParserStreamer>(tokenizer, parsers);
                 }),
            py::arg("tokenizer"),
            py::arg("parsers") = std::vector<std::shared_ptr<IncrementalParser>>(),
            py::keep_alive<1, 3>())
        
        // If we inherit and implement 'write' in Python and try to call write with text chunks or integer tokens 
        // then Python implementation will be called since python does not have overloads.
        // But for texts we need to check that when we call write with strings/integer tokens they are accumulated and stored correctly in py::dict.
        // Therefore we provide a private method '_write' which is used to call 'write' with correct parameters from C++ side.
        .def("_write",
            [](TextParserStreamer& self, std::variant<std::vector<int64_t>, std::string> chunk) -> StreamingStatus {
                if (auto _token = std::get_if<std::vector<int64_t>>(&chunk)) {
                    return self.write(*_token);
                } else if (auto _str =  std::get_if<std::string>(&chunk)) {
                    auto res = self.write(*_str);
                    return std::get<StreamingStatus>(res);
                }
                return StreamingStatus::RUNNING;
            },
            py::arg("chunk"), "This is a private method is used to call write with integer tokens or text chunks. Is used for text purposes only.")
        
        .def("get_parsed_message",
            [](TextParserStreamer& self) -> py::dict{
                return pyutils::json_container_to_py_object(self.get_parsed_message());

            }, "Returns the accumulated message.")
        
        .def("reset", &TextParserStreamer::reset, "Resets the internal state of the parser streamer.");

    py::class_<ov::genai::OmniSpeechStreamerBase, ConstructableOmniSpeechStreamer,
               std::shared_ptr<ov::genai::OmniSpeechStreamerBase>>(m, "OmniSpeechStreamerBase", omni_speech_streamer_base_docstring)
        .def(py::init<>())
        .def("write", &ov::genai::OmniSpeechStreamerBase::write,
             "Called with each audio chunk tensor [1, 1, N_samples]. Return StreamingStatus.",
             py::arg("audio_chunk"))
        .def("end", &ov::genai::OmniSpeechStreamerBase::end,
             "Called when speech generation completes.");

    py::class_<OmniStreamerBase, ConstructableOmniStreamer, std::shared_ptr<OmniStreamerBase>>(
        m, "OmniStreamerBase", omni_streamer_base_docstring)
        .def(py::init<>())
        .def("write",
             [](OmniStreamerBase& self, const py::dict& step) {
                 return self.write(omni_step_from_py(step));
             },
             py::arg("step"),
             "Called once per VLM decode step with that step's payload. Return StreamingStatus.")
        .def("end", &OmniStreamerBase::end,
             "Called exactly once when the VLM stage ends, including on an early stop or cancel.")
        .def("abandon", &OmniStreamerBase::abandon,
             "Called instead of end() when the VLM stage raised.");

    py::class_<OmniTextSourceBase, ConstructableOmniTextSource, std::shared_ptr<OmniTextSourceBase>>(
        m, "OmniTextSourceBase", omni_text_source_base_docstring)
        .def(py::init<>())
        .def("read",
             [](OmniTextSourceBase& self) -> py::typing::Optional<py::dict> {
                 std::optional<ov::AnyMap> step;
                 {
                     py::gil_scoped_release rel;
                     step = self.read();
                 }
                 if (!step) {
                     return py::none();
                 }
                 return pyutils::any_map_to_py_object(*step);
             },
             "Take the next VLM step, blocking until one arrives or the stream ends. None once the stream ended.")
        .def("truncated", &OmniTextSourceBase::truncated,
             "Whether the producer called abandon() instead of end(). Meaningful once read() returned None.");

    py::class_<OmniChannel, OmniStreamerBase, OmniTextSourceBase, std::shared_ptr<OmniChannel>>(
        m, "OmniChannel", omni_channel_docstring)
        .def(py::init<>())
           .def("finish_reading", &OmniChannel::finish_reading,
               "Close the read end after the talker finished normally. write() keeps returning RUNNING.")
           .def("fail_reading", &OmniChannel::fail_reading,
               "Close the read end after the talker failed. write() returns STOP from now on.");
}
