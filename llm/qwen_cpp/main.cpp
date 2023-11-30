#include "qwen.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"


typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::vector<std::string> english_sentences =
{
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
};

const std::vector<std::string> chinese_sentences =
{
    "介绍下清华大学",
    "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
    "糕点商店里原本有三种蛋糕：草莓奶油蛋糕，巧克力椰蓉蛋糕，和红丝绒布朗尼蛋糕。如名字所描述的那样，每种蛋糕都有两种成分：草莓奶油蛋糕包含草莓和奶油两个成分，巧克力椰蓉蛋糕包含巧克力和椰蓉两种成分，红丝绒布朗尼蛋糕包含红丝绒和布朗尼两种成分。在蛋糕制作完成后，往往每一种成分的材料都会有所剩余。为了减少浪费，商店常常会把多出来的成分两两搭配，做成新的小商品卖出去。比如草莓和巧克力可以做成草莓味巧克力酱，布朗尼和椰蓉可以做成布朗尼椰蓉饼干。以此类推可知，如果所有的成分都可以两两组合，那么最终商店能做出哪些小商品出来？",
    "桌子有左中右3个抽屉；张三，李四，王五，赵六都看到桌子上有一袋巧克力。张三让李四和王五出门后，在赵六面前把这袋巧克力放进了右抽屉；王五回来后，张三让赵六出门去找李四，并在王五面前从左抽屉拿出一盒饼干放进中抽屉里；等李四和赵六返回，张三又让王五和赵六出去买酱油，等二人走后，他告诉李四刚才已将一盒饼干放进中抽屉；张三等了很久，发现王五和赵六还没回来，就派李四去寻找，可最后只有王五和李四回来了。王五告诉张三，一开始他们没有找到卖酱油的店，所以只好分头去买，后来赵六走丢了；回来的路上，王五碰上了李四，两人便先赶了回来。于是，张三让两人出门去找赵六；为防再次走丢，张三叮嘱李四和王五要时刻同行，就算酱油买不到，也要找回赵六。结果，李四和王五在外面找到了赵六，发现他已经买了酱油。三人觉得张三从来不出门跑腿，十分气愤，讨论并达成共识，回去见到张三后，不要告诉他买到了酱油的事情，并让王五把酱油藏到自己的背包里。等三人一同回来后，他们按照计划谎称没有买到酱油，并希望张三以后买东西也要一同出门，不能偷懒，张三答应了。当大家最后站在桌子前，四人分别写下自己知道的物品清单和物品所在位置。问，这四人写下的物品和位置信息是否一致，为什么？",
    "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。请赏析以上内容的精妙之处。",
};

double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

struct Args {
  std::string model_path = "openvino_model.xml";
  std::string tiktoken_path = "qwen.tiktoken";
  std::string prompt = "你好";
  int max_length = 2048;
  int max_context_length = 512;
  std::string device = "CPU";
  bool verbose = false;
  std::string language = "chinese";
};

static auto usage(const std::string &prog) -> void {
  std::cout << "Usage: " << prog << " [options]\n"
            << "\n"
            << "options:\n"
            << "  -h, --help              show this help message and exit\n"
            << "  -m, --model PATH        model path (default: openvino_model.xml)\n"
            << "  -t, --tiktoken_path PATH    tokenizer path (default: qwen.tiktoken)\n"
            << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
            << "  -i, --interactive       run in interactive mode\n"
            << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
            << "  -c, --max_context_length N\n"
            << "                          max context length (default: 512)\n"
            << "  -d, --device DEVICE     specify which device used for inference\n"
            << "  -l, --language LANGUAGE specify test sentencen language, either english or chinese\n"
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
    } else if (arg == "-p" || arg == "--prompt") {
      args.prompt = argv[++i];
    } else if (arg == "-ml" || arg == "--max_length") {
      args.max_length = std::stoi(argv[++i]);
    } else if (arg == "-mcl" || arg == "--max_context_length") {
      args.max_context_length = std::stoi(argv[++i]);
    } else if (arg == "-d" || arg == "--device") {
      args.device = argv[++i];
    } else if (arg == "-l" || arg == "--language") {
      args.language = argv[++i];
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

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    qwen::QwenConfig config;
    double total_time;

    // Init Tokenizer
    auto startTime = Time::now();
    std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(args.tiktoken_path, config);
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load Qwen tokenizer took " << duration_ms << " ms" << std::endl;

    // Init Text Streamer
    auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, tokenizer.get());
    startTime = Time::now();

    // Init OpenVINO Runtime
    ov::Core core;
    ov::AnyMap device_config = {};
    if (args.device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    if (args.device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    // Read OpenVINO Model
    std::shared_ptr<ov::Model> model = core.read_model(args.model_path);
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Read Qwen Model took " << duration_ms << " ms" << std::endl;
    constexpr size_t BATCH_SIZE = 1;

    // Reshape model
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

    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    
    // Modify model input type to algin with tokenizer outputs with PrePostProcessor
    ov::preprocess::PrePostProcessor p3(model);
    p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenizer's output
    p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
    model = p3.build();
    inputs = model->inputs();

    // Compile model
    startTime = Time::now();
    ov::InferRequest ireq = core.compile_model(model, args.device, device_config).create_infer_request();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile model and create infer request took " << duration_ms << " ms" << std::endl;
    if (args.device.find("GPU") != std::string::npos) {
	    model = nullptr; // Release system memory after model compiled on GPU
    }
    int32_t out_token;
    int sentence_num = 0;
    std::vector<std::string> sentences;
    if (args.language.find("ch") != std::string::npos){
      sentences = chinese_sentences;
    }
    else if (args.language.find("en") != std::string::npos){
      sentences = english_sentences;
    }
    for (std::string input_text : sentences) {
      // Build input prompt with prompt template
      std::cout << "******************************************* Text Sentence #" << sentence_num << " Start *******************************************\n";
      std::cout << "Input text: " << input_text << "\n";
      startTime = Time::now();
      std::vector<int> input_ids = tokenizer->encode_history({input_text}, args.max_length);
      duration_ms = get_duration_ms_until_now(startTime);
      std::cout << "Input prompt encode using tokenizer took " << duration_ms << " ms" << std::endl;
      std::string output_text = tokenizer->decode(input_ids);
      std::cout << "Build input prompt with prompt template: \n" << output_text << "\n";

      // Prepare input tensor for first infer
      startTime = Time::now();
      for (size_t idx = 1; idx < inputs.size() - 1; ++idx) {
          ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
      }
      ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, input_ids.size() });
      ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, input_ids.size() });
      std::copy_n(input_ids.data(), input_ids.size(), ireq.get_tensor("input_ids").data<int32_t>());
      std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.size(), 1);
      std::cout << "Input token length: " << input_ids.size() << "\n";

      // First inference
      startTime = Time::now();
      ireq.infer();
      duration_ms = get_duration_ms_until_now(startTime);
      std::cout << "First inference took " << duration_ms << " ms" << std::endl;

      // Get first inference results
      size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
      float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.size() - 1) * vocab_size;
      out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
      if (text_streamer) {
        text_streamer->put({out_token});
      }

      ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
      total_time = 0;
      int count = 1;
      double second_time = 0;
      while (out_token !=config.eos_token_id && out_token!=config.im_end_id && count < args.max_context_length) {
          // Prepare input tensor for 2nd+ inference
          ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
          ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1});
          std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
          for (size_t idx = 1; idx < inputs.size() - 1; ++idx) {
            ireq.set_input_tensor(idx, ireq.get_output_tensor(idx));
          }
          // 2nd+ inference
          startTime = Time::now();
          ireq.start_async();
          ireq.wait();
          duration_ms = get_duration_ms_until_now(startTime);
          count += 1;
          // Get 2nd+ inference results
          logits = ireq.get_tensor("logits").data<float>();
          out_token = std::max_element(logits, logits + vocab_size) - logits;
          if (text_streamer) {
            text_streamer->put({out_token});
          }
          if (count != 2) {
            total_time += duration_ms;
          }
          else {
            second_time = duration_ms;
          }
      }
      std::cout << '\n';
      std::cout << "Second inference latency: " << second_time << " ms" << std::endl;
      if (count > 2) {
        std::cout << "Other inference tooks in total: " << total_time << " ms, Average other token latency: " << total_time / (count - 2) << " ms" << std::endl;
        std::cout << "Input num tokens: " << input_ids.size() << ", output num tokens: " << count - 1 << ", Average inference speed: " << (count - 2) / total_time * 1000.0 << " token/s\n";
      }
      std::cout << "******************************************* Text Sentence #" << sentence_num << " Finished ****************************************\n\n";
      sentence_num+=1;
    }
    if (text_streamer) {
        text_streamer->end();
    }
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}
