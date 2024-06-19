#include <gtest/gtest.h>

#include "imwrite.hpp"
#include "openpose_detector.hpp"
#include "utils.hpp"

TEST(OpenposeDetectorTest, UtilsFunction) {
    OpenposeDetector detector;
    detector.load("model");

    auto input_image = "scripts/im.txt";

    // Set up initial parameters
    int stride = 8;
    int pad_val = 128;

    unsigned long H = 768;
    unsigned long W = 512;
    unsigned long C = 3;

    // functional tests
    // Load Image
    std::cout << "Load " << input_image << std::endl;
    std::vector<std::uint8_t> im_array = read_bgr_from_txt(input_image);

    ov::Shape img_shape = {1, H, W, C};  // NHWC
    ov::Tensor img_tensor(ov::element::u8, img_shape);

    // validate the read function
    std::uint8_t* tensor_data = img_tensor.data<std::uint8_t>();
    std::copy(im_array.begin(), im_array.end(), tensor_data);
    std::cerr << "Tensor shape: " << img_tensor.get_shape() << std::endl;
    imwrite(std::string("im.bmp"), img_tensor, false);

    // validate the resize function
    ov::Tensor small_img_tensor = smart_resize_k(img_tensor, 0.5, 0.5);
    imwrite(std::string("im.half.bmp"), small_img_tensor, false);

    ov::Tensor big_img_tensor = smart_resize_k(img_tensor, 2, 2);
    imwrite(std::string("im.double.bmp"), big_img_tensor, false);

    ov::Tensor need_pad_img_tensor = smart_resize(img_tensor, 761, 505);
    auto [img_padded, pad] = pad_right_down_corner(need_pad_img_tensor, stride, pad_val);
    imwrite(std::string("im.paded.bmp"), img_padded, false);

    auto img_cropped = crop_right_down_corner(img_padded, pad);
    imwrite(std::string("im.cropped.bmp"), img_cropped, false);
}

TEST(OpenposeDetectorTest, ForwardFunction) {
    OpenposeDetector detector;
    detector.load("model");

    unsigned long H = 768;
    unsigned long W = 512;
    unsigned long C = 3;

    // read image from ndarray
    auto input_image = "scripts/im.txt";
    std::vector<std::uint8_t> input_array = read_bgr_from_txt(input_image);
    ov::Tensor ori_img(ov::element::u8, {1, H, W, C});
    std::uint8_t* input_data = ori_img.data<std::uint8_t>();
    std::copy(input_array.begin(), input_array.end(), input_data);

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

    // save candidate into a text file
    std::ofstream out("candidate.txt");
    for (auto& cand : candidate) {
        out << cand[0] << " " << cand[1] << " " << cand[2] << " " << cand[3] << std::endl;
    }
    out.close();

    // save subset into a text file
    out.open("subset.txt");
    for (auto& sub : subset) {
        for (auto& s : sub) {
            out << s << " ";
        }
        out << std::endl;
    }

    // we inspect the results in python
}
