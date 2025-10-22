# Qwen2.5-VL Image Processing Sample

This C++ sample demonstrates how to use Qwen2.5-VL (Visual Language Model) to process and analyze images using OpenVINO GenAI. The sample is designed as a foundation for integrating with OneVPL video processing output.

## Overview

The `qwen2vl_image_processing` sample shows how to:
- Load a Qwen2.5-VL model using OpenVINO GenAI
- Process a single image file
- Generate detailed descriptions or answer questions about the image
- Stream the model's response in real-time
- Run on different devices (CPU, GPU, NPU)

## Model Support

This sample is designed for Qwen2.5-VL models, which support:
- High-resolution image understanding
- Detailed visual analysis
- Multi-turn conversations (foundation for future enhancements)

## Prerequisites

### Install Dependencies

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
```

### Download and Export the Model

You can use Qwen2.5-VL models from Hugging Face. For example:

```sh
# Download and export Qwen2-VL-2B-Instruct (smaller model, faster inference)
optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct --trust-remote-code Qwen2-VL-2B-Instruct

# Or download Qwen2-VL-7B-Instruct (larger model, better quality)
optimum-cli export openvino --model Qwen/Qwen2-VL-7B-Instruct --trust-remote-code Qwen2-VL-7B-Instruct
```

**Note:** Qwen2.5-VL models may have specific export requirements. Check the model documentation for the latest export instructions.

## Building the Sample

Follow the [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) guide to build the samples.

Alternatively, from the build directory:

```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target qwen2vl_image_processing
```

## Usage

```sh
qwen2vl_image_processing <MODEL_DIR> <IMAGE_FILE> [PROMPT] [DEVICE]
```

### Arguments

- `MODEL_DIR` (required): Path to the exported Qwen2.5-VL model directory
- `IMAGE_FILE` (required): Path to the image file to process (supports common formats: jpg, png, etc.)
- `PROMPT` (optional): Question or instruction for the model (default: "Describe the image in detail.")
- `DEVICE` (optional): Device to run inference on - CPU, GPU, or NPU (default: CPU)

### Examples

#### Basic Usage (CPU)
```sh
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./sample_image.jpg
```

#### With Custom Prompt
```sh
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./photo.jpg "What objects are visible in this image?"
```

#### Using GPU
```sh
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./image.jpg "Describe the scene" GPU
```

#### Multiple Analysis Tasks
```sh
# Identify objects
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./image.jpg "List all objects in the image"

# Describe colors and composition
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./image.jpg "Describe the colors and composition"

# Count specific items
qwen2vl_image_processing ./Qwen2-VL-2B-Instruct ./image.jpg "How many people are in this image?"
```

## Sample Output

```
Loading image from: ./sample_image.jpg
Number of images loaded: 1
Initializing Qwen2.5-VL pipeline on CPU...

========================================
Prompt: Describe the image in detail.
========================================
Response: The image shows a beautiful sunset over a calm ocean. The sky is painted 
with vibrant shades of orange, pink, and purple, creating a stunning gradient effect. 
In the foreground, there are silhouettes of palm trees swaying gently in the breeze...

========================================
Image processing completed successfully.
```

## Device Recommendations

- **CPU**: Suitable for development and testing. Recommended for smaller models (2B parameters).
- **GPU**: Recommended for production use and larger models (7B+ parameters). Provides significant speedup and uses model caching for faster subsequent runs.
- **NPU**: Optimized for efficiency on Intel NPUs (Note: only the language model runs on NPU).

## Integration with OneVPL

This sample provides the foundation for processing video frames from OneVPL:

### Future Integration Steps:
1. **OneVPL Video Decoding**: Use OneVPL to decode video and extract frames
2. **Frame Processing**: Pass decoded frames to this sample's pipeline
3. **Batch Processing**: Process multiple frames efficiently
4. **Real-time Analysis**: Stream video analysis results

### Example Integration Concept:
```cpp
// Pseudocode for future OneVPL integration
while (oneVPL.hasFrame()) {
    ov::Tensor frame = oneVPL.getNextFrame();
    // Convert frame format if needed
    vlm_pipeline.generate(prompt, ov::genai::images({frame}), ...);
}
```

## Performance Tips

1. **GPU Caching**: When using GPU, the sample automatically caches compiled models in `qwen2vl_cache/` directory for faster subsequent runs.

2. **Model Size**: Choose the model size based on your requirements:
   - 2B model: Faster, suitable for real-time applications
   - 7B model: Higher quality, suitable for detailed analysis

3. **Batch Processing**: For multiple images, consider modifying the sample to process them in batches.

## Troubleshooting

### Model Loading Issues
- Ensure the model directory contains all required files (model files, tokenizer, configuration)
- Verify the model was exported correctly using `optimum-cli`

### Memory Issues
- Try using a smaller model variant (e.g., 2B instead of 7B)
- Reduce `max_new_tokens` in the generation config
- Use GPU with sufficient VRAM (16GB+ recommended for 7B models)

### Performance Issues
- Enable GPU caching by running on GPU device
- Ensure you're using the latest OpenVINO version
- Check that your hardware supports the selected device

## Related Samples

- [`visual_language_chat.cpp`](./visual_language_chat.cpp) - Interactive chat with visual language models
- [`benchmark_vlm.cpp`](./benchmark_vlm.cpp) - Benchmark visual language models performance
- [`encrypted_model_vlm.cpp`](./encrypted_model_vlm.cpp) - Use encrypted VLM models

## References

- [Qwen2-VL Model on Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [OpenVINO GenAI Documentation](https://docs.openvino.ai/2025/openvino-workflow-generative.html)
- [Supported Visual Language Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms)
- [OneVPL Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onevpl.html)
