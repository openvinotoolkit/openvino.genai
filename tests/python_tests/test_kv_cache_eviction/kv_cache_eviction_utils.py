from openvino_genai import SchedulerConfig

def get_scheduler_config(num_kv_blocks: int) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    scheduler_config.num_kv_blocks = num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_batched_tokens = 256
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config