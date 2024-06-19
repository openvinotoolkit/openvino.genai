// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "embeddings.hpp"
#include "handle_master.hpp"
#include "httplib.h"
#include "json.hpp"
#include "util.hpp"

using json = nlohmann::json;

// (TODO)
// 1. Add init llm pipeline function                                  - handle_init_llm
// 2. Add init bert pipeline function                                 - handle_init_embedings
// 3. Add bert embedding feature extraction function                  - handle_embeddings
// 4. Add server health check function                                - handle_health
// 5. Add llm pipeline unloading function                             - handle_unload_llm
// 6. Add bert pipeline unloading function                            - handle_unload_embeddings
// 7. Add store embeding in vector database function                  - handle_db_store_embeddings
// 8. Add document retrival from database based on embedding function - handle_db_retrival
// 9. Add request cancel function to stop generation                  - handle_request_cancel
// 10. Add reset function to reset rag sever status                   - handle_reset

int main(int argc, char** argv) try {
    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    std::cout << "Init http server" << std::endl;

    util::Args args = util::parse_args(argc, argv);

    HandleMaster handle_master;

    std::shared_ptr<ov::genai::LLMPipeline> llm_pointer;
    auto handle_llm_init = handle_master.get_handle("llm_init", llm_pointer, args);
    auto handle_llm = handle_master.get_handle("llm", llm_pointer, args);

    std::shared_ptr<Embeddings> embedding_pointer;
    auto handle_embeddings_init = handle_master.get_handle("embeddings_init", embedding_pointer, args);
    auto handle_embeddings = handle_master.get_handle("embedding", embedding_pointer, args);

    svr->Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    // svr->Post ("/health",       handle_health);
    svr->Post("/embeddings_init", handle_embeddings_init);
    svr->Post("/embeddings", handle_embeddings);
    svr->Post("/llm_init", handle_llm_init);
    svr->Post("/completions", handle_llm);

    svr->listen("0.0.0.0", 8080);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
