#include <opencv2/opencv.hpp>

#include "openpose_detector.hpp"
#include "utils.hpp"

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

int main(int argc, char* argv[]) {
    // use the first argument as the input image
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }
    // // read image from ndarray
    // std::vector<std::uint8_t> input_array = read_bgr_from_txt(argv[1]);
    // ov::Tensor ori_img(ov::element::u8, {1, 768, 512, 3});
    // std::uint8_t* input_data = ori_img.data<std::uint8_t>();
    // std::copy(input_array.begin(), input_array.end(), input_data);
    auto ori_img = read_image_to_tensor(argv[1]);

    std::cout << "Input image tensor shape: " << ori_img.get_shape() << std::endl;

    OpenposeDetector detector;
    detector.load("model");

    // forward, get subset and candidate
    std::vector<std::vector<float>> subset;
    std::vector<std::vector<float>> candidate;
    detector.forward(ori_img, subset, candidate);

    // print results
    for (auto& cand : candidate) {
        std::cout << "Candidate: " << cand[0] << " " << cand[1] << " " << cand[2] << " " << cand[3] << std::endl;
    }

    for (auto& sub : subset) {
        std::cout << "Subset: ";
        for (auto& s : sub) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }

    std::string outputname = argv[1];
    std::string candidate_output = outputname + ".candidate.txt";
    std::string subset_output = outputname + ".subset.txt";

    std::ofstream out(candidate_output);
    for (auto& cand : candidate) {
        out << cand[0] << " " << cand[1] << " " << cand[2] << " " << cand[3] << std::endl;
    }
    out.close();

    // save subset into a text file
    out.open(subset_output);
    for (auto& sub : subset) {
        for (auto& s : sub) {
            out << s << " ";
        }
        out << std::endl;
    }
}