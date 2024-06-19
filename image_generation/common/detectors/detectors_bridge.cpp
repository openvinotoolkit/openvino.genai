#include "openpose_detector.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    // use the first argument as the input image
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }
    // read image from ndarray
    std::vector<std::uint8_t> input_array = read_bgr_from_txt(argv[1]);
    ov::Tensor ori_img(ov::element::u8, {1, 768, 512, 3});
    std::uint8_t* input_data = ori_img.data<std::uint8_t>();
    std::copy(input_array.begin(), input_array.end(), input_data);

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