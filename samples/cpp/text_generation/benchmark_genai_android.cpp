// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include "misc.h"
#include "openvino/genai/llm_pipeline.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string csv_header =
    "iteration,model,framework,device,pretrain_time(s),input_size,infer_count,generation_time(s),"
    "output_size,latency(ms),1st_latency(ms),2nd_avg_latency(ms),precision,max_rss_mem(MB),max_uss_"
    "mem(MB),max_shared_mem(MB),prompt_idx,1st_infer_latency(ms),2nd_infer_avg_latency(ms),num_"
    "beams,batch_size,tokenization_time,detokenization_time,result_md5,start,end";

extern bool exit_mem_read;
extern MEM_CONSUME_INFO genai_mem_info;

typedef struct csv_report_data {
    int iteration;
    std::string model;
    std::string framework;
    std::string device;
    float pretrain_time;
    int input_size;
    int infer_count;
    float generation_time;
    float output_size;
    float latency;
    float _1st_latency;
    float _2nd_avg_latency;
    float precision;
    int max_rss_mem;
    int max_uss_mem;
    int max_shared_mem;
    int prompt_idx;
    float _1st_infer_latency;
    float _2nd_infer_avg_latency;
    int num_beams;
    int batch_size;
    float tokenization_time;
    float detokenization_time;
    std::string result_md5;
} CSV_REPORT_DATA;


int main(int argc, char* argv[]) try {

    cxxopts::Options options("benchmark_vanilla_genai", "Help command");
    std::string jsonlpath;
    std::string report_path = "/data/local/tmp/";
    std::ofstream report_file;
    std::string prompt_string;
    int prompt_size;

    options.add_options()("m,model", "Path to model and tokenizers base directory",
                          cxxopts::value<std::string>())(
        "p,prompt", "Prompt Sentence",
        cxxopts::value<std::string>()->default_value(
            "How does sunrise time gets affected during winter"))(
        "f,promptfile", "Path to Prompt file",
        cxxopts::value<std::string>()->default_value("NONE"))(
        "n,num_warmup", "Number of warmup iterations",
        cxxopts::value<size_t>()->default_value(std::to_string(1)))(
        "i,num_iter", "Number of iterations",
        cxxopts::value<size_t>()->default_value(std::to_string(3)))(
        "t,max_new_tokens", "Maximal number of new tokens",
        cxxopts::value<size_t>()->default_value(std::to_string(100)))(
        "d,device", "device", cxxopts::value<std::string>()->default_value("NPU"))("h,help",
                                                                                   "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
        std::cout<<"num of iteration : "<<result["num_iter"].as<size_t>()<<std::endl;
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    if (result["promptfile"].as<std::string>() != "NONE") {
        // Open and read the prompt file and get the prompt string.
        // The prompt size if present in the file will be ok, else it is strlen(). This needs to be
        // modified.
        jsonlpath = result["promptfile"].as<std::string>();
        if (getpromptfromfile(jsonlpath, prompt_string, &prompt_size)==EXIT_FAILURE) {
            return EXIT_FAILURE;
        }
    } else if (result["prompt"].as<std::string>().length()) {
        prompt_string = result["prompt"].as<std::string>();
        prompt_size = 8;  // The prompt size of the default prompt
    }
#ifdef DEBUG
    std::cout << "Prompt string is: " << prompt_string << "\n\n" << std::endl;
#endif
    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();

    // Prepare the report file to store the KPI results
    report_path +=
        "report_benchmark_genai_android_" + device + "-" + gettimestamp(gettimenow()) + ".csv";
    report_file.open(report_path);
    if (!report_file) {
        std::cout << "Unable to open CSV report file. Error code: " << errno << " "
                  << std::strerror(errno) << std::endl;
        return EXIT_FAILURE;
    }
    report_file << csv_header << "\n";
    report_file.flush();

    CSV_REPORT_DATA csv_report = {0};
    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();
    config.ignore_eos = true;
    config.do_sample = false;

    csv_report.model = fs::path(models_path).filename().string();
    csv_report.framework = "OV(Text Generation GenAI)";  // Default FW
    csv_report.device = device;
    csv_report.input_size = prompt_size;
    csv_report.output_size = config.max_new_tokens;
    csv_report.num_beams = config.num_beams;

    // Start memory consumption thread and allow it to run.
    exit_mem_read = false;
    DEBUG_PRINT("Starting memory consumption thread");
    pthread_t mem_thread = init_memory_consumption();

    std::cout << "Starting LLMpipeline on " << device << " device, with prompt size " << prompt_size
              << ". Token size is: " << config.max_new_tokens << std::endl;
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::PerfMetrics metrics_arr[num_iter];
    for (size_t i = 0; i < num_warmup; i++) pipe.generate(prompt_string, config);

    ov::genai::DecodedResults res = pipe.generate(prompt_string, config);

    // Convert 'res' to string.
    std::string result_string("empty string");
    std::ostringstream res_string;
    std::string latency[num_iter];
    std::string res_len[num_iter];
    std::tm start_tm[num_iter];
    std::tm end_tm[num_iter];
    size_t gen_token_size;

    // TBD: Collect the perf_metrics indiviually for each of the iteration and log them
    for (size_t i = 0; i < num_iter; i++) {
        // Record start time
        std::memcpy((void*)&start_tm[i], (void*)gettimenow(), sizeof(std::tm));

        res = pipe.generate(prompt_string, config);
        metrics_arr[i] = res.perf_metrics;

        // Record end time.
        std::memcpy((void*)&end_tm[i], (void*)gettimenow(), sizeof(std::tm));

        gen_token_size = res.perf_metrics.get_num_generated_tokens();
        res_string << res;
        result_string = res_string.str();
        std::cout<<"result_string : "<<result_string<<std::endl;
        std::cout<<"gen_token_size : "<<gen_token_size<<std::endl;
        if (gen_token_size) {
            latency[i] =
            std::to_string(float(gen_token_size / metrics_arr[i].get_generate_duration().mean));
            res_len[i] = std::to_string(result_string.length());
        } else {
            latency[i] = "NA";
            res_len[i] = "NA";
            DEBUG_PRINT("Result String not availiable");
        }
    }

    // Set flag for memory consumption thread to exit
    exit_mem_read = true;

#ifdef DEBUG
    std::cout << "Result_string: " << result_string << std::endl;
    std::cout << "Str len is: " << result_string.length() << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± "
              << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± "
              << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± "
              << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean << " ± " << metrics.get_ttft().std << " ms"
              << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean << " ± " << metrics.get_tpot().std
              << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean << " ± "
              << metrics.get_throughput().std << " tokens/s" << std::endl;
    std::cout << "Temperature: " << config.temperature << std::endl;
#endif

    // Print report to csv file
    std::string report_string;

    for (int i=0; i<num_iter; i++) {
        report_string =
            std::to_string(csv_report.iteration) + "," + csv_report.model + "," + csv_report.framework +
            "," + csv_report.device + "," + std::to_string(csv_report.pretrain_time) + "," +
            std::to_string(csv_report.input_size) + "," +
            /*infer count*/ std::to_string(csv_report.infer_count) + "," +
            /*generation_time*/ std::to_string(metrics_arr[i].get_generate_duration().mean) + "," +
            /*output_size*/ res_len[i] + "," +
            /*latency*/ latency[i] + "," +
            /*1st_latency*/ std::to_string(metrics_arr[i].get_ttft().mean) + "," +
            /*2nd_avg_latency*/ "NA" + "," +
            /*precision until*/ "," +
            /*max_rss_mem(MB),max_uss_mem(MB),max_shared_mem(MB)*/
            std::to_string(float(genai_mem_info.rss_mem / 1024)) + "," +
            std::to_string(float(genai_mem_info.uss_mem / 1024)) + "," +
            std::to_string(float(genai_mem_info.sh_mem / 1024)) + "," +
            /*prompt_idx to 2nd infer latency*/ std::to_string(0) + ",NA,NA," +
            /*num beam, batchsize*/ std::to_string(csv_report.num_beams) + "," +
            std::to_string(csv_report.batch_size) + "," +
            /*token & detoken time*/ std::to_string(metrics_arr[i].get_tokenization_duration().mean) + "," +
            std::to_string(metrics_arr[i].get_detokenization_duration().mean) + "," +
            /*md5, start,end time*/ "NA," + gettimestamp(&start_tm[i]) + "," +
            gettimestamp(&end_tm[i]) + "\n";
            csv_report.iteration++;
        report_file << report_string;
        report_file.flush();
    }

    float time=0, avg_time=0;
    float tftt=0, avg_tftt=0;
    float tokenization_time=0, avg_tokenization_time=0;
    float detokenization_time=0, avg_detokenization_time=0;
    float latency_time=0, avg_latency_time=0;
    for (int i=0; i<num_iter; i++) {
        time += metrics_arr[i].get_generate_duration().mean;
        tftt += metrics_arr[i].get_ttft().mean;
        tokenization_time += metrics_arr[i].get_tokenization_duration().mean;
        detokenization_time += metrics_arr[i].get_detokenization_duration().mean;
        latency_time += std::stof(latency[i]);
    }
    avg_time = time/num_iter;
    avg_tftt = tftt/num_iter;
    avg_tokenization_time = tokenization_time/num_iter;
    avg_detokenization_time = detokenization_time/num_iter;
    avg_latency_time = latency_time/num_iter;
    //Writing a summary
    std::string summary = "Summary (Avg)";
    report_string =
        summary + "," + csv_report.model + "," + csv_report.framework +
        "," + csv_report.device + "," + std::to_string(csv_report.pretrain_time) + "," +
        std::to_string(csv_report.input_size) + "," +
        /*infer count*/ std::to_string(csv_report.infer_count) + "," +
        /*generation_time*/ std::to_string(avg_time) + "," +
        /*output_size*/ res_len[num_iter-1] + "," +
        /*latency*/ std::to_string(avg_latency_time) /*latency[num_iter-1]*/ + "," +
        /*1st_latency*/ std::to_string(avg_tftt) + "," +
        /*2nd_avg_latency*/ "NA" + "," +
        /*precision until*/ "," +
        /*max_rss_mem(MB),max_uss_mem(MB),max_shared_mem(MB)*/
        std::to_string(float(genai_mem_info.rss_mem / 1024)) + "," +
        std::to_string(float(genai_mem_info.uss_mem / 1024)) + "," +
        std::to_string(float(genai_mem_info.sh_mem / 1024)) + "," +
        /*prompt_idx to 2nd infer latency*/ std::to_string(0) + ",NA,NA," +
        /*num beam, batchsize*/ std::to_string(csv_report.num_beams) + "," +
        std::to_string(csv_report.batch_size) + "," +
        /*token & detoken time*/ std::to_string(avg_tokenization_time) + "," +
        std::to_string(avg_detokenization_time) + "\n";

    report_file << report_string;
    report_file.flush();

    report_file.close();
    std::cout << "Test completed, result stored in " << report_path << std::endl;
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
