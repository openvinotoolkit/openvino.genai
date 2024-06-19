#include "httplib.h"
#include "json.hpp"
#include "iostream"
#include <unistd.h>
using json = nlohmann::json;

int main(){
    // HTTP
    std::cout << "Init client \n";
    httplib::Client cli("http://0.0.0.0:8080");
    cli.set_default_headers({{"Client", "openvino.genai.rag_sample"}});
    
    // Json Input sample
    //init embeddings request
    std::cout << "init embeddings\n";
    auto init_embeddings = cli.Post("/embeddings_init", "", "");
    std::cout << "init_embeddings->status: " << init_embeddings->status << "\n";
    std::cout << "init_embeddings->body: " << init_embeddings->body << "\n";
    sleep(1);

    std::cout << "load json\n";
    std::ifstream f("/home/chentianmeng/workspace/openvino.genai_server/openvino.genai/samples/cpp/rag_sample/document_data.json");
    json data = json::parse(f);
    auto embeddings = cli.Post("/embeddings", data.dump(), "application/json");

    std::cout << "embeddings->status: " << embeddings->status << "\n";
    std::cout << "embeddings->body: " << embeddings->body << "\n";
    std::cout << "embeddings finished\n";
    //promot input sample
    std::string user_prompt = "What is OpenVINO?";

    auto llm_init = cli.Post("/llm_init",  "", "");
    std::cout << "llm_init->status: " << llm_init->status << "\n";
    std::cout << "llm_init->body: " << llm_init->body << "\n";
    sleep(1);

    auto completions = cli.Post("/completions", user_prompt, "text/plain");
    std::cout << "completions->status: " << completions->status << "\n";
    std::cout << "completions->body: " << completions->body << "\n";
    return 0;
}
