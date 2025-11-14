// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "py_utils.hpp"

namespace {

constexpr char class_docstring[] = R"(
    ChatHistory stores conversation messages and optional metadata for chat templates.

    Manages:
    - Message history (array of message objects)
    - Optional tools definitions array (for function calling)
    - Optional extra context object (for custom template variables)

    Messages are stored as JSON-like structures but accessed as Python dicts.
    Use get_messages() to retrieve the list of all messages, modify them,
    and set_messages() to update the history.

    Example:
        ```python
        history = ChatHistory()
        history.append({"role": "user", "content": "Hello"})
        
        # Modify messages
        messages = history.get_messages()
        messages[0]["content"] = "Updated"
        history.set_messages(messages)
        ```
)";

}  // namespace

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::ChatHistory;
using ov::genai::JsonContainer;

void init_chat_history(py::module_& m) {
    py::class_<ChatHistory>(m, "ChatHistory", class_docstring)
        .def(py::init<>(), "Create an empty chat history.")

        .def(py::init([](const py::list& messages) {
            JsonContainer history = pyutils::py_object_to_json_container(messages);
            return ChatHistory(history);
        }), py::arg("messages"), R"(Create chat history from a list of message dicts.)")

        .def("get_messages", [](const ChatHistory& self) -> py::list {
            return pyutils::json_container_to_py_object(self.get_messages());
        }, R"(Get all messages as a list of dicts (deep copy).)")

        .def("set_messages", [](ChatHistory& self, const py::list& messages) {
            self.get_messages() = pyutils::py_object_to_json_container(messages);
        }, py::arg("messages"), R"(Replace all messages with a new list.)")

        .def("append", [](ChatHistory& self, const py::dict& message) {
            JsonContainer message_jc = pyutils::py_object_to_json_container(message);
            self.push_back(message_jc);
        }, py::arg("message"), R"(Add a message to the end of chat history.)")

        .def("pop", [](ChatHistory& self) -> py::dict {
            if (self.empty()) {
                throw py::index_error("Cannot pop from an empty chat history");
            }
            JsonContainer last = self.last().copy();
            self.pop_back();
            return pyutils::json_container_to_py_object(last);
        }, R"(Remove and return the last message.)")

        .def("clear", &ChatHistory::clear)

        .def("__len__", &ChatHistory::size)
        
        .def("__bool__", [](const ChatHistory& self) {
            return !self.empty();
        })

        .def("set_tools", [](ChatHistory& self, const py::list& tools) {
            self.set_tools(pyutils::py_object_to_json_container(tools));
        }, py::arg("tools"), R"(Set the tools definitions array.)")

        .def("get_tools", [](const ChatHistory& self) -> py::list {
            return pyutils::json_container_to_py_object(self.get_tools());
        }, R"(Get the tools definitions array.)")

        .def("set_extra_context", [](ChatHistory& self, const py::dict& extra_context) {
            self.set_extra_context(pyutils::py_object_to_json_container(extra_context));
        }, py::arg("extra_context"), R"(Set the extra context object.)")

        .def("get_extra_context", [](const ChatHistory& self) -> py::dict {
            return pyutils::json_container_to_py_object(self.get_extra_context());
        }, R"(Get the extra context object.)");
}
