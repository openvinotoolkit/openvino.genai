#include "load_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #define strcasecmp _stricmp
#else
  #include <strings.h>
#endif

#ifdef _WIN32
    #include <io.h>
    #define stat _stat
#else
    #include <sys/stat.h>
#endif

#include "openvino/c/openvino.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static const char* supported_extensions[] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".psd", ".gif", ".hdr", ".pic", ".pnm"
};
static const size_t num_extensions = sizeof(supported_extensions) / sizeof(supported_extensions[0]);

static int is_supported_image(const char* filename) {
    if (!filename) return 0;
    
    size_t len = strlen(filename);
    for (size_t i = 0; i < num_extensions; i++) {
        size_t ext_len = strlen(supported_extensions[i]);
        if (len >= ext_len) {
            const char* ext = filename + len - ext_len;
            if (strcasecmp(ext, supported_extensions[i]) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

typedef struct {
    unsigned char* image_data;
    int channels;
    int height;
    int width;
} image_allocator_t;

static void* image_allocate(size_t bytes, size_t alignment, void* user_data) {
    image_allocator_t* allocator = (image_allocator_t*)user_data;
    if (allocator && allocator->image_data && 
        allocator->channels * allocator->height * allocator->width == (int)bytes) {
        return allocator->image_data;
    }
    return NULL;
}

static void image_deallocate(void* ptr, size_t bytes, size_t alignment, void* user_data) {
    image_allocator_t* allocator = (image_allocator_t*)user_data;
    if (allocator && allocator->image_data && 
        allocator->channels * allocator->height * allocator->width == (int)bytes) {
        stbi_image_free(allocator->image_data);
        allocator->image_data = NULL;
    }
}

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

ov_tensor_t* load_image(const char* image_path) {
    if (!image_path) {
        fprintf(stderr, "Error: image_path is NULL\n");
        return NULL;
    }
    
    if (!file_exists(image_path)) {
        fprintf(stderr, "Error: Image file '%s' does not exist\n", image_path);
        return NULL;
    }
    
    int width, height, channels;
    const int desired_channels = 3;
    
    unsigned char* data = stbi_load(image_path, &width, &height, &channels, desired_channels);
    if (!data) {
        fprintf(stderr, "Error: Failed to load image '%s': %s\n", image_path, stbi_failure_reason());
        return NULL;
    }
    
    image_allocator_t* allocator = (image_allocator_t*)malloc(sizeof(image_allocator_t));
    if (!allocator) {
        fprintf(stderr, "Error: Failed to allocate memory for allocator\n");
        stbi_image_free(data);
        return NULL;
    }
    
    allocator->image_data = data;
    allocator->channels = desired_channels;
    allocator->height = height;
    allocator->width = width;
    
    ov_tensor_t* tensor = NULL;
    ov_element_type_e input_type = U8;
    int64_t dims[4] = {1, height, width, desired_channels};

    ov_shape_t input_shape = {.rank = 0, .dims = NULL};
    ov_shape_create(4, dims, &input_shape);

    ov_tensor_create_from_host_ptr(
        input_type,
        input_shape,  // shape: [1, H, W, C]
        data,
        &tensor
    );
    
    free(allocator);
    
    return tensor;
}

const ov_tensor_t** load_images(const char* image_path, size_t* tensor_count) {
    if (!image_path || !tensor_count) {
        fprintf(stderr, "Error: image_path or tensor_count is NULL\n");
        return NULL;
    }
    
    if (!file_exists(image_path)) {
        fprintf(stderr, "Error: Image file '%s' does not exist\n", image_path);
        return NULL;
    }
    
    ov_tensor_t* tensor = load_image(image_path);
    if (!tensor) {
        return NULL;
    }
    
    const ov_tensor_t** tensors = (const ov_tensor_t**)malloc(sizeof(ov_tensor_t*));
    if (!tensors) {
        fprintf(stderr, "Error: Failed to allocate memory for single tensor\n");
        free_tensor(tensor);
        return NULL;
    }
    
    tensors[0] = tensor;
    *tensor_count = 1;
    
    return tensors;
}

void free_tensor(ov_tensor_t* tensor) {
    if (tensor) {
        ov_tensor_free(tensor);
    }
}

void free_tensor_array(ov_tensor_t** tensors, size_t count) {
    if (tensors) {
        for (size_t i = 0; i < count; i++) {
            if (tensors[i]) {
                ov_tensor_free(tensors[i]);
            }
        }
        free(tensors);
    }
}

int file_exists(const char* path) {
    if (!path) return 0;
    
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

