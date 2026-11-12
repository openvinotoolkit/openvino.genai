"""Checks the chunk store: with `CacheOffloadConfig.store_completed_blocks`, every block is stored right after the
forward pass that completed it, so a new process can reuse it from a persistent store although nothing was evicted.

Run the three phases as separate processes against the same store directory:

  python verify_chunk_store.py -m <model_dir> -d CPU --plugin <plugin> --store-dir <dir> --phase baseline
  python verify_chunk_store.py -m <model_dir> -d CPU --plugin <plugin> --store-dir <dir> --phase write
  python verify_chunk_store.py -m <model_dir> -d CPU --plugin <plugin> --store-dir <dir> --phase read

  baseline: no offload; prints the TTFT of the prompt and the generated text.
  write:    offload through a persistent provider with store_completed_blocks. The device cache is large, so nothing is
            evicted and the store only holds what the chunk store wrote.
  read:     a new process opens the same store and runs the same prompt. Its TTFT should be far below the baseline's
            and the generated text must be identical.

For a hybrid model use a prompt longer than one linear-attention checkpoint interval (see E2E_VALIDATION.md).
"""

import argparse
import json
import sys
import time

import openvino_genai as ov_genai


def make_pipeline(args, offload: bool) -> ov_genai.ContinuousBatchingPipeline:
    scheduler = ov_genai.SchedulerConfig()
    scheduler.enable_prefix_caching = True
    scheduler.max_num_batched_tokens = 4096
    scheduler.max_num_seqs = 1
    scheduler.dynamic_split_fuse = True
    scheduler.num_kv_blocks = args.num_kv_blocks
    if offload:
        offload_config = ov_genai.CacheOffloadConfig()
        offload_config.store_library_path = args.plugin
        offload_config.store_provider_name = args.provider
        offload_config.allow_untrusted_store_plugin = True
        offload_config.kv_cache_dir = args.store_dir
        offload_config.capacity_bytes = 4 * 1024 ** 3
        offload_config.linear_attention_capacity_bytes = 8 * 1024 ** 3
        offload_config.buffer_slots = 512
        offload_config.store_completed_blocks = not args.no_chunk
        scheduler.use_cache_offload = True
        scheduler.cache_offload_config = offload_config
    return ov_genai.ContinuousBatchingPipeline(args.model, scheduler, args.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="Path to an OpenVINO IR model directory")
    parser.add_argument("-d", "--device", default="CPU", help="Inference device (default: CPU)")
    parser.add_argument("--plugin", required=True, help="Path to your compiled store plugin (.dll/.so)")
    parser.add_argument("--provider", default="sample-persistent-file",
                        help="Provider whose store survives the process (default: the sample's persistent provider)")
    parser.add_argument("--store-dir", required=True, help="Directory of the persistent store")
    parser.add_argument("--phase", choices=["baseline", "write", "read"], required=True)
    parser.add_argument("--no-chunk", action="store_true", help="Leave store_completed_blocks off in write/read")
    parser.add_argument("--repeat", type=int, default=300, help="Repetitions of the sample sentence in the prompt")
    parser.add_argument("--num-kv-blocks", type=int, default=600,
                        help="Device cache blocks; large enough that nothing is evicted")
    parser.add_argument("--new-tokens", type=int, default=8)
    args = parser.parse_args()

    prompt = "The quick brown fox jumps over the lazy dog. " * args.repeat
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = args.new_tokens
    generation_config.do_sample = False
    generation_config.ignore_eos = True

    pipeline = make_pipeline(args, offload=args.phase != "baseline")
    started = time.perf_counter()
    result = pipeline.generate([prompt], [generation_config])[0]
    total_ms = (time.perf_counter() - started) * 1000
    print("RESULT " + json.dumps({"phase": args.phase,
                                  "ttft_ms": round(result.perf_metrics.get_ttft().mean, 1),
                                  "total_ms": round(total_ms, 1),
                                  "text": result.m_generation_ids[0]}))
    sys.stdout.flush()
    del pipeline


if __name__ == "__main__":
    main()
