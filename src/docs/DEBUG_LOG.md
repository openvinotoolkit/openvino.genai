## 1. Using Debug Log

There are six levels of logs, which can be called explicitly or set via the ``OPENVINO_LOG_LEVEL`` environment variable:

0 - ``ov::log::Level::NO``
1 - ``ov::log::Level::ERR``
2 - ``ov::log::Level::WARNING``
3 - ``ov::log::Level::INFO``
4 - ``ov::log::Level::DEBUG``
5 - ``ov::log::Level::TRACE``

When setting the environment variable OPENVINO_LOG_LEVEL > ov::log::Level::WARNING, the properties of the compiled model can be printed.

For example:

Linux - export OPENVINO_LOG_LEVEL=3
Windows - set OPENVINO_LOG_LEVEL=3

the properties of the compiled model are printed as follows:
```sh
    NETWORK_NAME: Model0
    OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    NUM_STREAMS: 1
    INFERENCE_NUM_THREADS: 48
    PERF_COUNT: NO
    INFERENCE_PRECISION_HINT: bf16
    PERFORMANCE_HINT: LATENCY
    EXECUTION_MODE_HINT: PERFORMANCE
    PERFORMANCE_HINT_NUM_REQUESTS: 0
    ENABLE_CPU_PINNING: YES
    SCHEDULING_CORE_TYPE: ANY_CORE
    MODEL_DISTRIBUTION_POLICY:
    ENABLE_HYPER_THREADING: NO
    EXECUTION_DEVICES: CPU
    CPU_DENORMALS_OPTIMIZATION: NO
    LOG_LEVEL: LOG_NONE
    CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1
    DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    KV_CACHE_PRECISION: f16
    AFFINITY: CORE
    EXECUTION_DEVICES:
    CPU: Intel(R) Xeon(R) Platinum 8468
```
