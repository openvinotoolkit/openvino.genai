#include "handle_master.hpp"

using json = nlohmann::json;

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle(std::string handle_name,
                                                                                          HandleInput handle_type,
                                                                                          util::Args args) {
    if (auto embedding_pointer = std::get_if<std::shared_ptr<Embeddings>>(&handle_type)) {
        if (handle_name == "embeddings_init") {
            return get_handle_embeddings_init(*embedding_pointer, args);
        } else {
            return get_handle_embeddings(*embedding_pointer);
        }
    } else if (auto llm_pointer = std::get_if<std::shared_ptr<ov::genai::LLMPipeline>>(&handle_type)) {
        if (handle_name == "llm_init") {
            return get_handle_llm_init(*llm_pointer, args);
        } else {
            return get_handle_llm(*llm_pointer, args);
        }
    } else {
        std::cout << "handle_type is not supported, return void handle function\n";
        return {};
    }
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_llm_init(
    std::shared_ptr<ov::genai::LLMPipeline>& llm_pointer_ref,
    util::Args args) {
    const auto handle_llm_init = [&llm_pointer_ref, args](const httplib::Request& req, httplib::Response& res) {
        llm_pointer_ref = std::make_shared<ov::genai::LLMPipeline>(args.llm_model_path, args.llm_device);
    };
    return handle_llm_init;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_llm(
    std::shared_ptr<ov::genai::LLMPipeline>& llm_pointer_ref,
    util::Args args) {
    const auto handle_llm = [&llm_pointer_ref, args](const httplib::Request& req_llm, httplib::Response& res_llm) {
        res_llm.set_header("Access-Control-Allow-Origin", req_llm.get_header_value("Origin"));
        std::cout << "req_llm.body: " << req_llm.body << "\n";
        std::string prompt = req_llm.body;
        auto config = llm_pointer_ref->get_generation_config();
        config.max_new_tokens = args.max_new_tokens;
        llm_pointer_ref->start_chat();
        std::string response = llm_pointer_ref->generate(prompt, config);
        llm_pointer_ref->finish_chat();
        // std::string processed_prompt = apply_chat_template(prompt);
        // std::cout << "processed_prompt: " << processed_prompt << "\n";
        res_llm.set_content(response, "text/plain");
    };
    return handle_llm;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_embeddings_init(
    std::shared_ptr<Embeddings>& embedding_pointer_ref,
    util::Args args) {
    const auto handle_embeddings_init = [&embedding_pointer_ref, args](const httplib::Request& req_embedding,
                                                                       httplib::Response& res_embedding) {
        embedding_pointer_ref = std::make_shared<Embeddings>(args.embedding_model_path, args.embedding_device);
    };

    return handle_embeddings_init;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_embeddings(
    std::shared_ptr<Embeddings>& embedding_ref) {
    const auto handle_embeddings = [&embedding_ref](const httplib::Request& req_embedding,
                                                    httplib::Response& res_embedding) {
        res_embedding.set_header("Access-Control-Allow-Origin", req_embedding.get_header_value("Origin"));
        json json_file = json::parse(req_embedding.body);
        std::cout << "get json_file successed\n";
        std::vector<std::string> inputs;
        for (auto& elem : json_file["data"])
            inputs.push_back(elem);
        std::cout << "get inputs successed\n";
        std::vector<std::vector<std::vector<float>>> res = embedding_ref->encode_queries(inputs);
        // res.set_content(response, "application/json");
    };

    return handle_embeddings;
}
