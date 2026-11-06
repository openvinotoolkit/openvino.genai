import argparse
import time

import openvino as ov
import openvino_genai as ov_genai

FILLER_TEXT = "OpenVINO GenAI continuous batching uses paged KV cache blocks. "


def make_prompt(tokenizer, target_tokens):
    base_ids = tokenizer.encode(FILLER_TEXT, add_special_tokens=False).input_ids.data[0].tolist()
    if not base_ids:
        raise RuntimeError("The tokenizer returned no tokens for the filler text")

    candidate_tokens = target_tokens
    for _ in range(20):
        token_ids = (base_ids * (candidate_tokens // len(base_ids) + 1))[:candidate_tokens]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
        actual_tokens = tokenizer.encode(prompt).input_ids.data.shape[1]
        if actual_tokens == target_tokens:
            return prompt
        candidate_tokens += target_tokens - actual_tokens

    raise RuntimeError(f"Could not construct a prompt with exactly {target_tokens} tokens")


def make_generation_config(max_new_tokens):
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.ignore_eos = True
    return config


def timed_step(pipe, name):
    start = time.perf_counter()
    pipe.step()
    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = pipe.get_metrics()
    print(f"{name}: {elapsed_ms:.2f} ms, "
          f"kv_cache={metrics.kv_cache_size_in_bytes / 1024**2:.1f} MiB, "
          f"usage={metrics.cache_usage:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Trigger dynamic KV cache growth from 64 blocks.")
    parser.add_argument("model_dir")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--chunked", action="store_true", help="Use chunked KV cache with FP16 KV cache (GPU only).")
    args = parser.parse_args()

    # Matches KVCacheManager: 16 tokens per block on GPU (non-XAttention), 32 on CPU.
    block_size = 16 if "GPU" in args.device else 32
    max_new_tokens = 2 * block_size

    # The requests need 64 + 16 logical blocks; the allocator may reserve extra physical capacity.
    prompt_a_len = 64 * block_size - max_new_tokens
    prompt_b_len = 16 * block_size - max_new_tokens

    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.cache_size = 0
    scheduler_config.num_kv_blocks = 0
    scheduler_config.max_num_batched_tokens = 64 * block_size
    scheduler_config.max_num_seqs = 2
    scheduler_config.enable_prefix_caching = False
    pipeline_properties = {}
    if args.chunked:
        scheduler_config.use_chunked_kv_cache = True
        pipeline_properties["KV_CACHE_PRECISION"] = ov.Type.f16

    pipe = ov_genai.ContinuousBatchingPipeline(args.model_dir, scheduler_config, args.device, pipeline_properties)
    tokenizer = pipe.get_tokenizer()
    generation_config = make_generation_config(max_new_tokens)
    generation_config.apply_chat_template = False
    prompt_a = make_prompt(tokenizer, prompt_a_len)
    prompt_b = make_prompt(tokenizer, prompt_b_len)

    print(f"block_size={block_size}, A prompt={prompt_a_len}, B prompt={prompt_b_len}, "
          f"max_new_tokens={max_new_tokens}, chunked={args.chunked}")

    handles = [pipe.add_request(0, prompt_a, generation_config)]
    timed_step(pipe, "step 0: create 64-block cache and prefill A")

    handles.append(pipe.add_request(1, prompt_b, generation_config))
    timed_step(pipe, "step 1: add B (80-block logical demand) and prefill")

    while pipe.has_non_finished_requests():
        pipe.step()

    for request_id, handle in enumerate(handles):
        outputs = handle.read_all()
        print(f"request {request_id}: generated {len(outputs[0].generated_ids)} tokens")


if __name__ == "__main__":
    main()
