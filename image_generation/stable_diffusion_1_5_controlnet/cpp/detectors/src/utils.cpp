#include "utils.hpp"

ov::Tensor init_tensor_with_zeros(const ov::Shape& shape, ov::element::Type type) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

std::vector<uint8_t> read_bgr_from_txt(const std::string& file_name) {
    std::ifstream input_data(file_name, std::ifstream::in);

    std::vector<uint8_t> res;
    std::string line;
    while (std::getline(input_data, line)) {
        try {
            int value = std::stoi(line);
            if (value < 0 || value > 255) {
                std::cerr << "invalid uint8: " << value << std::endl;
                continue;
            }
            res.push_back(static_cast<uint8_t>(value));
        } catch (const std::invalid_argument& e) {
            std::cerr << "invalid line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "out of range: " << line << std::endl;
        }
    }

    return res;
}

// Function to resize a single channel, could use libeigen or even openvino later
cv::Mat resize_single_channel(const cv::Mat& channel, int Ht, int Wt, float k) {
    cv::Mat resized_channel;
    int interpolation = (k < 1) ? cv::INTER_AREA : cv::INTER_LANCZOS4;
    cv::resize(channel, resized_channel, cv::Size(Wt, Ht), 0, 0, interpolation);
    return resized_channel;
}

// Smart resize function
template <typename T>
ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = input_tensor.get_shape();
    auto N = input_shape[0];
    auto Ho = input_shape[1];
    auto Wo = input_shape[2];
    auto Co = input_shape[3];

    // Determine the scaling factor
    float k = static_cast<float>(Ht + Wt) / static_cast<float>(Ho + Wo);

    // Prepare the output tensor
    ov::Shape output_shape = {N, static_cast<unsigned long>(Ht), static_cast<unsigned long>(Wt), Co};
    ov::Tensor output_tensor(ov::element::from<T>(), output_shape);
    T* output_data = output_tensor.data<T>();
    const T* input_data = input_tensor.data<T>();

    // Process each channel separately
    for (int c = 0; c < Co; ++c) {
        // Extract single channel
        cv::Mat channel(Ho, Wo, cv::DataType<T>::type);
        for (int h = 0; h < Ho; ++h) {
            for (int w = 0; w < Wo; ++w) {
                channel.at<T>(h, w) = input_data[h * Wo * Co + w * Co + c];
            }
        }
        // Resize the single channel
        cv::Mat resized_channel = resize_single_channel(channel, Ht, Wt, k);

        // Copy resized channel back to the output tensor
        for (int h = 0; h < Ht; ++h) {
            for (int w = 0; w < Wt; ++w) {
                output_data[h * Wt * Co + w * Co + c] = resized_channel.at<T>(h, w);
            }
        }
    }

    return output_tensor;
}

ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    if (input_tensor.get_element_type() == ov::element::u8) {
        return smart_resize<uint8_t>(input_tensor, Ht, Wt);
    } else if (input_tensor.get_element_type() == ov::element::f32) {
        return smart_resize<float>(input_tensor, Ht, Wt);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

ov::Tensor smart_resize_k(const ov::Tensor& input_tensor, float fx, float fy) {
    auto input_shape = input_tensor.get_shape();
    auto Ho = input_shape[1];
    auto Wo = input_shape[2];
    int Ht = Ho * fy;
    int Wt = Wo * fx;
    return smart_resize(input_tensor, Ht, Wt);
}

template <typename T>
ov::Tensor lanczos_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = input_tensor.get_shape();
    auto N = input_shape[0];
    auto Ho = input_shape[1];
    auto Wo = input_shape[2];
    auto Co = input_shape[3];

    // Prepare the output tensor
    ov::Shape output_shape = {N, static_cast<unsigned long>(Ht), static_cast<unsigned long>(Wt), Co};
    ov::Tensor output_tensor(ov::element::from<T>(), output_shape);
    T* output_data = output_tensor.data<T>();
    const T* input_data = input_tensor.data<T>();

    // Process each channel separately
    for (int c = 0; c < Co; ++c) {
        // Extract single channel
        cv::Mat channel(Ho, Wo, cv::DataType<T>::type);
        for (int h = 0; h < Ho; ++h) {
            for (int w = 0; w < Wo; ++w) {
                channel.at<T>(h, w) = input_data[h * Wo * Co + w * Co + c];
            }
        }
        // Resize the single channel
        cv::Mat resized_channel = resize_single_channel(channel, Ht, Wt, 1);

        // Copy resized channel back to the output tensor
        for (int h = 0; h < Ht; ++h) {
            for (int w = 0; w < Wt; ++w) {
                output_data[h * Wt * Co + w * Co + c] = resized_channel.at<T>(h, w);
            }
        }
    }

    return output_tensor;
}

ov::Tensor lanczos_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    if (input_tensor.get_element_type() == ov::element::u8) {
        return lanczos_resize<uint8_t>(input_tensor, Ht, Wt);
    } else if (input_tensor.get_element_type() == ov::element::f32) {
        return lanczos_resize<float>(input_tensor, Ht, Wt);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

// Function to pad the tensor
std::pair<ov::Tensor, std::vector<int>> pad_right_down_corner(const ov::Tensor& img, int stride, uint8_t pad_val) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = img.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];

    // Calculate padding sizes
    std::vector<int> pad(4);
    pad[0] = 0;                                              // up
    pad[1] = 0;                                              // left
    pad[2] = (H % stride == 0) ? 0 : stride - (H % stride);  // down
    pad[3] = (W % stride == 0) ? 0 : stride - (W % stride);  // right

    // Calculate new dimensions
    int H_new = H + pad[0] + pad[2];
    int W_new = W + pad[1] + pad[3];

    // Create a new tensor with the new dimensions
    ov::Shape output_shape = {N, static_cast<unsigned long>(H_new), static_cast<unsigned long>(W_new), C};
    ov::Tensor img_padded(ov::element::u8, output_shape);

    // Initialize img_padded with padValue
    uint8_t* padded_data = img_padded.data<uint8_t>();
    std::fill(padded_data, padded_data + N * H_new * W_new * C, pad_val);

    // Copy the original image into the new padded image
    const uint8_t* img_data = img.data<uint8_t>();

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = ((n * H + h) * W + w) * C + c;
                    int dst_idx = ((n * H_new + (h + pad[0])) * W_new + (w + pad[1])) * C + c;
                    padded_data[dst_idx] = img_data[src_idx];
                }
            }
        }
    }

    return {img_padded, pad};
}

// Function to crop the tensor
template <typename T>
ov::Tensor crop_right_down_corner(const ov::Tensor& input, std::vector<int> pad) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = input.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];

    int down = pad[2];
    int right = pad[3];

    // Calculate new dimensions
    int H_new = H - down;
    int W_new = W - right;

    // Create a new tensor with the new dimensions
    ov::Shape output_shape = {N, static_cast<unsigned long>(H_new), static_cast<unsigned long>(W_new), C};
    ov::Tensor output(ov::element::from<T>(), output_shape);

    T* cropped_data = output.data<T>();
    const T* input_data = input.data<T>();

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_new; ++h) {
            for (int w = 0; w < W_new; ++w) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = ((n * H + h) * W + w) * C + c;
                    int dst_idx = ((n * H_new + h) * W_new + w) * C + c;
                    cropped_data[dst_idx] = input_data[src_idx];
                }
            }
        }
    }

    return output;
}

// Overloaded function to handle specific conversions
ov::Tensor crop_right_down_corner(const ov::Tensor& input, const std::vector<int>& pad) {
    if (input.get_element_type() == ov::element::u8) {
        return crop_right_down_corner<uint8_t>(input, pad);
    } else if (input.get_element_type() == ov::element::f32) {
        return crop_right_down_corner<float>(input, pad);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

// Function to convert uint8_t rgb tensor to float32 normalized tensor
ov::Tensor normalize_rgb_tensor(const ov::Tensor& input) {
    ov::Tensor output(ov::element::f32, input.get_shape());

    float* output_data = output.data<float>();
    const uint8_t* input_data = input.data<uint8_t>();

    auto input_shape = input.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int idx = ((n * H + h) * W + w) * C + c;
                    output_data[idx] = static_cast<float>(input_data[idx]) / 256 - 0.5;
                }
            }
        }
    }

    return output;
}

template <typename T>
ov::Tensor cv_gaussian_blur(const ov::Tensor& input_tensor, int sigma) {
    // Get input tensor dimensions (NHWC)
    // Assume N and C are always 1 (we apply it to each channel of the heatmap)
    auto input_shape = input_tensor.get_shape();
    auto H = input_shape[1];
    auto W = input_shape[2];

    // Convert input tensor data pointer to cv::Mat
    const T* input_data = input_tensor.data<T>();
    cv::Mat img(H, W, cv::DataType<T>::type, const_cast<T*>(input_data));

    // Calculate kernel size
    int truncate = 4;
    int radius = static_cast<int>(truncate * sigma + 0.5);
    int ksize = 2 * radius + 1;

    // Apply Gaussian blur
    cv::Mat blurred_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(ksize, ksize), sigma, sigma);

    // Copy blurred image data back to output tensor
    ov::Tensor output_tensor(input_tensor.get_element_type(), input_shape);
    T* output_data = output_tensor.data<T>();
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            output_data[i * W + j] = blurred_img.at<T>(i, j);
        }
    }

    return output_tensor;
}

ov::Tensor cv_gaussian_blur(const ov::Tensor& input, int sigma) {
    if (input.get_element_type() == ov::element::u8) {
        return cv_gaussian_blur<uint8_t>(input, sigma);
    } else if (input.get_element_type() == ov::element::f32) {
        return cv_gaussian_blur<float>(input, sigma);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

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