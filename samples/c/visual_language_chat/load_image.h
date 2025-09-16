#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ov_tensor ov_tensor_t;

/**
 * @param image_path 
 * @return 
 */
ov_tensor_t* load_image(const char* image_path);

/**
 * @param image_path 
 * @param tensor_count 
 * @return 
 */
ov_tensor_t** load_images(const char* image_path, size_t* tensor_count);

/**
 * @param tensor 
 */
void free_tensor(ov_tensor_t* tensor);

/**
 * @param tensors 
 * @param count 
 */
void free_tensor_array(ov_tensor_t** tensors, size_t count);

/**
 * @param path 
 * @return 
 */
int file_exists(const char* path);

#ifdef __cplusplus
}
#endif

#endif // LOAD_IMAGE_H
