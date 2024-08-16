#ifndef MINICPMV_H
#define MINICPMV_H

#include "openvino/genai/clip.hpp"
#include "openvino/genai/visibility.hpp"

struct clip_ctx;

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};


/** build an image embed from image file bytes */
OPENVINO_GENAI_EXPORTS std::vector<std::vector<clip_image_u8 *>> slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14, const bool never_split=false);
OPENVINO_GENAI_EXPORTS std::vector<std::vector<struct llava_image_embed *>> llava_image_embed_make_with_bytes_slice(struct clip_ctx * ctx_clip, const unsigned char * image_bytes, int image_bytes_length);

OPENVINO_GENAI_EXPORTS void llava_image_embed_free_slice(std::vector<std::vector<struct llava_image_embed*>> embed);

OPENVINO_GENAI_EXPORTS bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);
#endif
