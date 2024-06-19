#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <openvino/openvino.hpp>
#include <sstream>
#include <variant>

#include "embeddings.hpp"
#include "httplib.h"
#include "json.hpp"
#include "util.hpp"

using json = nlohmann::json;
using HandleInput = std::variant<int, std::shared_ptr<Embeddings>, std::shared_ptr<ov::genai::LLMPipeline>>;

class HandleMaster {
public:
    HandleMaster() = default;
    ~HandleMaster() = default;

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle(std::string handle_name,
                                                                                HandleInput handle_type,
                                                                                util::Args args);

private:
    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_llm_init(
        std::shared_ptr<ov::genai::LLMPipeline>& llm_pointer_ref,
        util::Args args);
    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_llm(
        std::shared_ptr<ov::genai::LLMPipeline>& llm_pointer_ref,
        util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_embeddings_init(
        std::shared_ptr<Embeddings>& embedding_pointer_ref,
        util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_embeddings(
        std::shared_ptr<Embeddings>& embedding_ref);
};
