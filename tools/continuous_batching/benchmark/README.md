## Run vlm continuous batching benchmark:

```sh
vlm_continuous_batching_benchmark [OPTIONS]
```

### Options

- `-m, --model`(default: `.`): Path to the model and tokenizers base directory.
- `-n, --num_prompts` (default: `1`): Number of prompts, a prompt corresponds to one or more images.
- `--dataset`: Path to dataset .json file, read prompts and images from this file.
- `--max_output_len` (default: `128`): Maximal number of output tokens.
- `--device` (default: `"CPU"`): Device to run the model on.
- `-b, --max_batch_size` (default: `256`): Maximum number of requests to process in a single batch during continuous batching.
- `--dynamic_split_fuse` (default: `true`): Enable dynamic splitting and fusing of prompts to improve batch utilization.
- `--cache_size` (default: `16 GB`): Size of the KV-cache used for continuous batching (controls how many cached sequences can be stored).
- `--use_cache_eviction` (default: `false`): Enable eviction of entries from the KV-cache when the configured cache size is exceeded.

### Dataset JSON file format

```
[
    {
        "prompt": "what is it in the image?",
        "image": "multi_images_448x448/image_448x448.jpg"
    },
    {
        "prompt": "what is it in the image?",
        "image": "multi_images_448x448/"
    }
]
```

### Output:

```
vlm_continuous_batching_benchmark -m qwen2-vl-2b --dataset vlm_input.json -n 1
```

```
Benchmark duration: 4 s
[0] Input prompt tokens: 6
[0] Input image[0]: width:448, height:448
[0] Number of output tokens: 127
Total number of input tokens: 6
Total number of output tokens: 127
Input throughput: 1 tokens / s
Output throughput: 31 tokens / s
Mean TTFT: 503 ms
Mean TPOT: 24 ms
[0] generated text:Human: The cat is is sitting on on on a bench. field.You are looking at at at the bench. cat.What is the cat doing?Can you tell me what what what the cat is is?You see a cat sitting on a bench in a field...Human:: The cat is is sitting on on on a bench a field. What is is the cat doing?Human:: The cat is is sitting on on on a bench field. What is is is the cat? cat?Human: The cat is is sitting on on on a bench. field
```

For more information how performance metrics are calculated please follow [performance-metrics tutorial](../../../src/README.md#performance-metrics).
