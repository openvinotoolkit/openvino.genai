#include "httplib.h"
#include "json.hpp"
#include "iostream"
using json = nlohmann::ordered_json;

int main(){
    // HTTP
    std::cout << "Init client \n";
    httplib::Client cli("http://0.0.0.0:8080");
    cli.set_default_headers({{"Client", "openvino.genai.rag_sample"}});
    
    // Json Input sample
    //std::cout << "load json\n";
    //std::ifstream f("document_data.json");
    //json data = json::parse(f);
    //std::cout << "data: " << data.dump() << "\n";
    std::string user_prompt = "What is OpenVINO?";
    //auto res = cli.Post("/completions", data.dump(), "text/plain");
    auto res = cli.Post("/completions", user_prompt, "text/plain");
    std::cout << "res->status: " << res->status << "\n";
    std::cout << "res->body: " << res->body << "\n";
    return 0;
}
