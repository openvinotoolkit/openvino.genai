## Run vlm continuous batching benchmark:

```sh
vlm_continuous_batching_benchmark [OPTIONS]
```

### Options

- `-m, --model`(default: `.`): Path to the model and tokenizers base directory.
- `-n, --num_prompts` (default: `1`): Number of prompts, a prompt corresponds to one or more images.
- `--dataset`: Path to dataset .json file, read prompts and images from this file.
- `-mt, --max_new_tokens` (default: `128`): Maximal number of output tokens.
- `-d, --device` (default: `"CPU"`): Device to run the model on.

### Dataset JSON file format

```
[
    {
        "prompt": "what is it in the image?",
        "image": "multi_images_448x448/image_448x448.jpg"    //support one image file
    },
    {
        "prompt": "what is it in the image?",
        "image": "multi_images_448x448/"                     //support image path
    },
    .......
]
```

### Output:

```
vlm_continuous_batching_benchmark -m qwen2-vl-2b --dataset vlm_input.json -n 2
```

```
Benchmark duration: 4 s
Total number of input prompt tokens: 2036
Total number of output tokens: 254
Input throughput: 509 tokens / s
Output throughput: 63 tokens / s
Mean TTFT: 731 ms
Mean TPOT: 23 ms
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
