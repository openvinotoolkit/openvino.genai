#include <opencv2/opencv.hpp>

#include "imwrite.hpp"
#include "openpose_detector.hpp"

ov::Tensor read_image_to_tensor(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read the image file: " + image_path);
    }

    int height = image.rows;
    int width = image.cols;

    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }

    int resolution = 512;
    float k = static_cast<float>(resolution) / std::min(height, width);
    float kH = height * k;
    float kW = width * k;

    int H = static_cast<int>(std::round(kH / 64.0)) * 64;
    int W = static_cast<int>(std::round(kW / 64.0)) * 64;

    int interpolation_method = (k > 1) ? cv::INTER_LANCZOS4 : cv::INTER_AREA;
    cv::resize(image, image, cv::Size(W, H), 0, 0, interpolation_method);

    height = image.rows;
    width = image.cols;

    int channels = image.channels();
    ov::Shape tensor_shape = {1,
                              static_cast<unsigned long>(height),
                              static_cast<unsigned long>(width),
                              static_cast<unsigned long>(channels)};
    ov::Tensor tensor(ov::element::u8, tensor_shape);

    std::memcpy(tensor.data(), image.data, height * width * channels * sizeof(uint8_t));

    return tensor;
}

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