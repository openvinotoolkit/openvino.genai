#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_tensor ov_tensor_t;

ov_tensor_t* load_image(const char* image_path);

const ov_tensor_t** load_images(const char* image_path, size_t* tensor_count);

void free_tensor(ov_tensor_t* tensor);

void free_tensor_array(ov_tensor_t** tensors, size_t count);

int file_exists(const char* path);

#ifdef __cplusplus
}
#endif

#endif // LOAD_IMAGE_H
