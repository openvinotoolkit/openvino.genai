#include "qwen.h"
#include <iomanip>
#include <iostream>
#include <openvino/openvino.hpp>
#include <chrono>

/*
enum InferenceMode {
  INFERENCE_MODE_CHAT,
  INFERENCE_MODE_GENERATE,
};

static inline auto to_inference_mode(const std::string &s) -> InferenceMode {
  static std::unordered_map<std::string, InferenceMode> m{{"chat", INFERENCE_MODE_CHAT},
                                                          {"generate", INFERENCE_MODE_GENERATE}};
  return m.at(s);
}
*/

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::string sentences[] =
{
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
};

double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

struct Args {
  std::string model_path = "openvino_model.xml";
  std::string tiktoken_path = "qwen.tiktoken";
  //InferenceMode mode = INFERENCE_MODE_CHAT;
  std::string prompt = "你好";
  int max_length = 2048;
  int max_context_length = 512;
  std::string device = "CPU";
  //bool interactive = false;
  //int top_k = 0;
  //float top_p = 0.5;
  //float temp = 0.95;
  //float repeat_penalty = 1.0;
  int num_threads = 0;
  bool verbose = false;
};

static auto usage(const std::string &prog) -> void {
  std::cout << "Usage: " << prog << " [options]\n"
            << "\n"
            << "options:\n"
            << "  -h, --help              show this help message and exit\n"
            << "  -m, --model PATH        model path (default: openvino_model.xml)\n"
            << "  -t, --tiktoken_path PATH    tokenizer path (default: qwen.tiktoken)\n"
            //<< "  --mode                  inference mode chose from {chat, generate} (default: chat)\n"
            << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
            << "  -i, --interactive       run in interactive mode\n"
            << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
            << "  -c, --max_context_length N\n"
            << "                          max context length (default: 512)\n"
            //<< "  --top_k N               top-k sampling (default: 0)\n"
            //<< "  --top_p N               top-p sampling (default: 0.7)\n"
            //<< "  --temp N                temperature (default: 0.95)\n"
            //<< "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
            //<< "  -t, --threads N         number of threads for inference\n"
            << "  -d, --device DEVICE     specify which device used for inference\n"
            << "  -v, --verbose           display verbose output including config/system/performance info\n";
}

static auto parse_args(const std::vector<std::string> &argv) -> Args {
  Args args;

  for (size_t i = 1; i < argv.size(); i++) {
    const std::string &arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else if (arg == "-m" || arg == "--model") {
      args.model_path = argv[++i];
    } else if (arg == "-t" || arg == "--tiktoken_path") {
      args.tiktoken_path = argv[++i];
    //} else if (arg == "--mode") {
    //  args.mode = to_inference_mode(argv[++i]);
    } else if (arg == "-p" || arg == "--prompt") {
      args.prompt = argv[++i];
    //} else if (arg == "-i" || arg == "--interactive") {
    //  args.interactive = true;
    } else if (arg == "-l" || arg == "--max_length") {
      args.max_length = std::stoi(argv[++i]);
    } else if (arg == "-c" || arg == "--max_context_length") {
      args.max_context_length = std::stoi(argv[++i]);
    } else if (arg == "-d" || arg == "--device") {
      args.device = argv[++i];
    //} else if (arg == "--top_k") {
    //  args.top_k = std::stoi(argv[++i]);
    //} else if (arg == "--top_p") {
    //  args.top_p = std::stof(argv[++i]);
    //} else if (arg == "--temp") {
    //  args.temp = std::stof(argv[++i]);
    //} else if (arg == "--repeat_penalty") {
    //  args.repeat_penalty = std::stof(argv[++i]);
    //} else if (arg == "-t" || arg == "--threads") {
    //  args.num_threads = std::stoi(argv[++i]);
    } else if (arg == "-v" || arg == "--verbose") {
      args.verbose = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  return args;
}

static auto parse_args(int argc, char **argv) -> Args {
  std::vector<std::string> argv_vec;
  argv_vec.reserve(argc);

  for (int i = 0; i < argc; i++) {
    argv_vec.emplace_back(argv[i]);
  }

  return parse_args(argv_vec);
}

static auto get_utf8_line(std::string &line) -> bool {
  return !!std::getline(std::cin, line);
}
/*
static auto chat(Args &args) -> void {
  ggml_time_init();
  int64_t start_load_us = ggml_time_us();
  qwen::Pipeline pipeline(args.model_path, args.tiktoken_path);
  int64_t end_load_us = ggml_time_us();

  std::string model_name = "qwen";

  auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, pipeline.tokenizer.get());
  auto perf_streamer = std::make_shared<qwen::PerfStreamer>();
  auto streamer = std::make_shared<qwen::StreamerGroup>(
    std::vector<std::shared_ptr<qwen::BaseStreamer>>{text_streamer, perf_streamer});

  qwen::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                    args.top_p, args.temp, args.repeat_penalty, args.num_threads);

  if (args.verbose) {
    std::cout << "system info: | "
              << "AVX = " << ggml_cpu_has_avx() << " | "
              << "AVX2 = " << ggml_cpu_has_avx2() << " | "
              << "AVX512 = " << ggml_cpu_has_avx512() << " | "
              << "AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << " | "
              << "AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << " | "
              << "FMA = " << ggml_cpu_has_fma() << " | "
              << "NEON = " << ggml_cpu_has_neon() << " | "
              << "ARM_FMA = " << ggml_cpu_has_arm_fma() << " | "
              << "F16C = " << ggml_cpu_has_f16c() << " | "
              << "FP16_VA = " << ggml_cpu_has_fp16_va() << " | "
              << "WASM_SIMD = " << ggml_cpu_has_wasm_simd() << " | "
              << "BLAS = " << ggml_cpu_has_blas() << " | "
              << "SSE3 = " << ggml_cpu_has_sse3() << " | "
              << "VSX = " << ggml_cpu_has_vsx() << " |\n";

    std::cout << "inference config: | "
              << "max_length = " << args.max_length << " | "
              << "max_context_length = " << args.max_context_length << " | "
              << "top_k = " << args.top_k << " | "
              << "top_p = " << args.top_p << " | "
              << "temperature = " << args.temp << " | "
              << "num_threads = " << args.num_threads << " |\n";

    std::cout << "loaded qwen model from " << args.model_path
              << " within: " << (end_load_us - start_load_us) / 1000.f << " ms\n";

    std::cout << std::endl;
  }

  if (args.mode != INFERENCE_MODE_CHAT && args.interactive) {
    std::cerr << "interactive demo is only supported for chat mode, falling back to non-interactive one\n";
    args.interactive = false;
  }

  if (args.interactive) {
    std::cout << R"( _____                                                      )" << '\n'
              << R"(|  _  |                                                     )" << '\n'
              << R"(| | | | __      __   ___   _ __         ___   _ __    _ __  )" << '\n'
              << R"(| | | | \ \ /\ / /  / _ \ | '_ \       / __| | '_ \  | '_ \ )" << '\n'
              << R"(\ \/' /  \ V  V /  |  __/ | | | |  _  | (__  | |_) | | |_) |)" << '\n'
              << R"( \_/\_\   \_/\_/    \___| |_| |_| (_)  \___| | .__/  | .__/ )" << '\n'
              << R"(                                             | |     | |    )" << '\n'
              << R"(                                             |_|     |_|    )" << '\n'
              << '\n';

    std::cout
        << "Welcome to Qwen.cpp! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit.\n"
        << "\n";

    std::vector<std::string> history;
    while (1) {
      std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                << " > " << std::flush;
      std::string prompt;
      if (!get_utf8_line(prompt) || prompt == "stop") {
        break;
      }
      if (prompt.empty()) {
        continue;
      }
      if (prompt == "clear") {
        history.clear();
        continue;
      }
      history.emplace_back(std::move(prompt));
      std::cout << model_name << " > ";
      std::string output = pipeline.chat(history, gen_config, streamer.get());
      history.emplace_back(std::move(output));
      if (args.verbose) {
          std::cout << "\n" << perf_streamer->to_string() << "\n\n";
      }
      perf_streamer->reset();
    }
    std::cout << "Bye\n";
  } else {
    if (args.mode == INFERENCE_MODE_CHAT) {
      pipeline.chat({args.prompt}, gen_config, streamer.get());
    } else {
      pipeline.generate(args.prompt, gen_config, streamer.get());
    }
    if (args.verbose) {
      std::cout << "\n" << perf_streamer->to_string() << "\n\n";
    }
  }
}
*/
int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    //chat(args);
    qwen::QwenConfig config;
    // load tokenizer
    double total_time;
    auto startTime = Time::now();
    std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(args.tiktoken_path, config);
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load Qwen tokenizer took " << duration_ms << " ms" << std::endl;

    auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, tokenizer.get());
    //std::cout << "load tokenzier successfully!\n";
    std::string prompt = args.prompt;
    int max_length = args.max_length;
    //std::vector<int> input_ids = tokenizer->encode({prompt}, max_length);
    
    startTime = Time::now();
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(args.model_path);
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Read Qwen Model took " << duration_ms << " ms" << std::endl;

    //std::cout << "Read model successfully\n";
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }}
    };
    
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 1; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    //std::cout << "Model reshape successfully\n";

    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    
    ov::preprocess::PrePostProcessor p3(model);
    p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenyzer's output
    p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
    //p3.input("position_ids").tensor().set_element_type(ov::element::i32);
    model = p3.build();
    //inputs = model->inputs();
    
    //std::cout << "PPP build successfully\n";
    //std::cout << "input_ids.size(): " << input_ids.size() << "\n";
    startTime = Time::now();
    ov::InferRequest ireq = core.compile_model(model, args.device, ov::cache_dir("llm-cache")).create_infer_request();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile model and create infer request took " << duration_ms << " ms" << std::endl;
    //std::cout << "Compile model on " << args.device << " successfully!\n";
    std::cout << "Input text: " << prompt << "\n";
    startTime = Time::now();
    std::vector<int> input_ids = tokenizer->encode_history({prompt}, max_length);
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Input prompt encode using tokenizer took " << duration_ms << " ms" << std::endl;
    /*
    std::cout << "Qwen tokenzier encode result: input ids [ ";
    for (auto & token : input_ids) {
      std::cout << token << " "; 
    }
    std::cout << "]\n";
    */
    std::string output_text = tokenizer->decode(input_ids);
    std::cout << "Build input with prompt template: \n" << output_text << "\n";

    if (text_streamer) {
      text_streamer->put({input_ids});
    }
    for (size_t idx = 1; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    startTime = Time::now();
    ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, input_ids.size() });
    ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, input_ids.size() });
    std::copy_n(input_ids.data(), input_ids.size(), ireq.get_tensor("input_ids").data<int32_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.size(), 1);
    std::cout << "Input token length: " << input_ids.size() << ", set first input tensor took " << duration_ms << " ms" << std::endl;

    startTime = Time::now();
    ireq.infer();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "First inference took " << duration_ms << " ms" << std::endl;

    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.size() - 1) * vocab_size;
    int32_t out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
    if (text_streamer) {
      text_streamer->put({out_token});
    }

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    //ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    //constexpr int64_t SPECIAL_EOS_TOKEN = ;  // There's no way to extract the value from the detokenizer for now
    //const int32_t SPECIAL_EOS_TOKEN = config.eos_token_id;
    //std::cout << "SPECIAL_EOS_TOKEN: " << SPECIAL_EOS_TOKEN << "\n";
    
    total_time = 0;
    int count = 0;
    double second_time = 0;
    while (out_token !=config.eos_token_id && out_token!=config.im_end_id) {
        startTime = Time::now();
        ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
        ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1});
        std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
        //ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;

        for (const ov::Output<ov::Node>& input : model->inputs()) {
            for (const std::string& name : input.get_names()) {
                if (name.rfind("past_key_values", 0) == 0) {
                    //std::cout << "name: " << name << "\n";
                    ireq.set_tensor(input, ireq.get_tensor("present" + name.substr(15)));
                    break;
                }
            }
        }
        
        ireq.start_async();
        //print_token(detokenizer, out_token);
        ireq.wait();
        duration_ms = get_duration_ms_until_now(startTime);
        count += 1;
        logits = ireq.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
        if (text_streamer) {
          text_streamer->put({out_token});
        }
        if (count != 1) {
          total_time += duration_ms;
        }
        else {
          second_time = duration_ms;
        }

        if (count + 1 > args.max_context_length) {
          break;
        }
        /*
        std::string second_infer_text = tokenizer->decode({out_token});
        std::cout << second_infer_text;
        */
    }
    if (text_streamer) {
      text_streamer->end();
    }
    std::cout << '\n';
    std::cout << "Second inference latency: " << second_time << " ms" << std::endl;
    if (count > 2) {
      std::cout << "Other inference tooks " << total_time << " ms, generated num tokens: " << count - 1 << ", Average other token latency: " << total_time / (count - 1) << " ms" << std::endl;
      std::cout << "Average inference speed: " << (count - 1) / total_time * 1000.0 << " token/s\n";
    }
    /*
    Args args = parse_args(argc, argv);
    //chat(args);
    qwen::QwenConfig config;
    // load tokenizer
    std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(args.tiktoken_path, config);
    std::cout << "load tokenzier successfully!\n";
    std::string prompt = args.prompt;
    int max_length = 2048;
    std::vector<int> input_ids = tokenizer->encode(prompt, max_length);
    std::cout << "Qwen tokenzier encode result: input ids [";
    for (auto & token : input_ids) {
      std::cout << token << " "; 
    }
    std::cout << "]\n";
    std::string output_text = tokenizer->decode(input_ids);
    std::cout << "Qwen tokenzier decode results: " << output_text << "\n";
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("/home/openvino/workspaces/AIGC/repo/ov_llm_bench/llm_models/Qwen-7B-Chat-GPTQ_INT4_FP16-2K/openvino_model.xml");
    std::cout << "Read model finished\n";
    */
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "\n";
  return 0;
}
