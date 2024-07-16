#include <opencv2/opencv.hpp>

#include "imwrite.hpp"
#include "openpose_detector.hpp"
#include "utils.hpp"

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " -i <input_image> -o <output_image> -m <model>" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_image;
    std::string output_image;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc) {
            input_image = argv[++i];
        } else if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            output_image = argv[++i];
        } else if (std::string(argv[i]) == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input_image.empty() || output_image.empty() || model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    auto ori_img = read_image_to_tensor(input_image);

    std::cout << "Input image tensor shape: " << ori_img.get_shape() << std::endl;

    OpenposeDetector detector;
    detector.load(model_path + "/openpose.xml");

    // forward, get subset and candidate
    std::vector<std::vector<float>> subset;
    std::vector<std::vector<float>> candidate;
    auto result = detector.forward(ori_img, subset, candidate);

    std::cout << "[DONE] result: " << output_image << std::endl;

    imwrite(output_image, result, true);
}