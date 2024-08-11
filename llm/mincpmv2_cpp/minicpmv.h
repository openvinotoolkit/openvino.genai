#ifndef MINICPMV_H
#define MINICPMV_H


#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MINICPMV_API __declspec(dllexport)
#        else
#            define MINICPMV_API __declspec(dllimport)
#        endif
#    else
#        define MINICPMV_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MINICPMV_API
#endif

struct clip_ctx;

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};


/** build an image embed from image file bytes */
MINICPMV_API std::vector<std::vector<clip_image_u8 *>> slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14, const bool never_split=false);
MINICPMV_API std::vector<std::vector<struct llava_image_embed *>> llava_image_embed_make_with_bytes_slice(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);

MINICPMV_API void llava_image_embed_free_slice(std::vector<std::vector<struct llava_image_embed*>> embed);

//MINICPMV_API bool llava_image_embed_make_with_clip_img_ollama(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);
MINICPMV_API bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);

//read image from file
MINICPMV_API bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long* sizeOut);


#endif
