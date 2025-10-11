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
[0] input prompt tokens: 6
[0] input image[0]: width:448, height:448
[0] number of output tokens: 127
Total number of input prompt tokens: 6
Total number of output tokens: 127
Input throughput: 1 tokens / s
Output throughput: 31 tokens / s
Mean TTFT: 503 ms
Mean TPOT: 24 ms
[0] generated text:Human: The cat is is sitting on on on a bench. field.You are looking at at at the bench. cat.What is the cat doing?Can you tell me what what what the cat is is?You see a cat sitting on a bench in a field...Human:: The cat is is sitting on on on a bench a field. What is is the cat doing?Human:: The cat is is sitting on on on a bench field. What is is is the cat? cat?Human: The cat is is sitting on on on a bench. field
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
