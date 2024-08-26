#ifndef MINICPMV_H
#define MINICPMV_H

#include "openvino/genai/clip.hpp"
#include "openvino/genai/visibility.hpp"

struct clip_ctx;

struct HeightWidth {
    size_t height, width;
};

struct EncodedImage {
    ov::Tensor resized_source;
    HeightWidth resized_source_size;
    ov::Tensor slices;
    std::vector<HeightWidth> slices_sizes;
};

/** build an image embed from image file bytes */
OPENVINO_GENAI_EXPORTS std::vector<std::vector<clip_image_u8>> slice_image(const clip_image_u8& img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14, const bool never_split=false);
OPENVINO_GENAI_EXPORTS EncodedImage llava_image_embed_make_with_bytes_slice(clip_ctx& ctx_clip, const ov::Tensor& img, ov::InferRequest& encoder, int max_slice_nums, int scale_resolution, size_t patch_size, bool never_split);

OPENVINO_GENAI_EXPORTS void llava_image_embed_free_slice(std::vector<std::vector<struct llava_image_embed*>> embed);
#endif
