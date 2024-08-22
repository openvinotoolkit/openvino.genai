#ifndef MINICPMV_H
#define MINICPMV_H

#include "openvino/genai/clip.hpp"
#include "openvino/genai/visibility.hpp"

struct clip_ctx;

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

struct HeightWidth {
    size_t height, width;
};

/** build an image embed from image file bytes */
OPENVINO_GENAI_EXPORTS std::vector<std::vector<clip_image_u8 *>> slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14, const bool never_split=false);
OPENVINO_GENAI_EXPORTS std::pair<std::vector<std::vector<ov::Tensor>>, std::vector<std::vector<HeightWidth>>> llava_image_embed_make_with_bytes_slice(struct clip_ctx * ctx_clip, const ov::Tensor& img, int max_slice_nums, int scale_resolution, int patch_size, bool never_split);

OPENVINO_GENAI_EXPORTS void llava_image_embed_free_slice(std::vector<std::vector<struct llava_image_embed*>> embed);
#endif
