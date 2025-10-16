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

When Speculative Decoding or Prompt Lookup pipeline is executed, performance metrics will be also printed.

For example:

```
===============================
Total duration, sec: 26.6217
Draft model duration, sec: 1.60329
Main model duration, sec: 25.0184
Draft model duration, %: 6.02248
Main model duration, %: 93.9775
AVG acceptance rate, %: 21.6809
===============================
REQUEST_ID: 0
Main model iterations: 47
Token per sec: 3.75633
AVG acceptance rate, %: 21.6809
Accepted tokens by draft model: 51
Generated tokens: 100
Accepted token rate, %: 51
===============================
Request_id: 0 ||| 40 0 40 20 0 0 40 40 0 20 20 20 0 40 0 0 20 80 0 80 20 0 0 0 40 80 0 40 60 40 80 0 0 0 0 40 20 20 0 40 20 40 0 20 0 0 0
```


When GGUF model passed to LLMPipeline, the details debug info will be also printed.

For example:
```
[GGUF Reader]: Loading and unpacking model from: gguf_models/qwen2.5-0.5b-instruct-q4_0.gguf
[GGUF Reader]: Loading and unpacking model done. Time: 196ms
[GGUF Reader]: Start generating OpenVINO model...
[GGUF Reader]: Save generated OpenVINO model to: gguf_models/openvino_model.xml done. Time: 466 ms
[GGUF Reader]: Model generation done. Time: 757ms
```
