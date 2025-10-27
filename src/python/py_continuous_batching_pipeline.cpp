// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/sparse_attention.hpp"
#include "tokenizer/tokenizers_path.hpp"

#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::AggregationMode;
using ov::genai::SparseAttentionMode;
using ov::genai::CacheEvictionConfig;
using ov::genai::SparseAttentionConfig;
using ov::genai::ContinuousBatchingPipeline;
using ov::genai::GenerationResult;
using ov::genai::EncodedGenerationResult;
using ov::genai::GenerationHandleImpl;
using ov::genai::GenerationOutput;
using ov::genai::GenerationFinishReason;
using ov::genai::GenerationStatus;
using ov::genai::SchedulerConfig;
using ov::genai::PipelineMetrics;
using ov::genai::KVCrushAnchorPointMode;
using ov::genai::KVCrushConfig;
using ov::genai::ChatHistory;

namespace {

auto cache_eviction_config_docstring = R"(
    Configuration struct for the cache eviction algorithm.
    :param start_size: Number of tokens in the *beginning* of KV cache that should be retained in the KV cache for this sequence during generation. Must be non-zero and a multiple of the KV cache block size for this pipeline.
    :type start_size: int

    :param recent_size: Number of tokens in the *end* of KV cache that should be retained in the KV cache for this sequence during generation. Must be non-zero and a multiple of the KV cache block size for this pipeline.
    :type recent_size: int

    :param max_cache_size: Maximum number of tokens that should be kept in the KV cache. The evictable block area will be located between the "start" and "recent" blocks and its size will be calculated as (`max_cache_size` - `start_size` - `recent_size`). Must be non-zero, larger than (`start_size` + `recent_size`), and a multiple of the KV cache block size for this pipeline. Note that since only the completely filled blocks are evicted, the actual maximum per-sequence KV cache size in tokens may be up to (`max_cache_size` + `SchedulerConfig.block_size - 1`).
    :type max_cache_size: int

    :param aggregation_mode: The mode used to compute the importance of tokens for eviction
    :type aggregation_mode: openvino_genai.AggregationMode

    :param apply_rotation: Whether to apply cache rotation (RoPE-based) after each eviction.
      Set this to false if your model has different RoPE scheme from the one used in the
      original llama model and you experience accuracy issues with cache eviction enabled.
    :type apply_rotation: bool

    :param snapkv_window_size The size of the importance score aggregation window (in token positions from the end of the prompt) for
      computing initial importance scores at the beginning of the generation phase for purposes of eviction,
      following the SnapKV article approach (https://arxiv.org/abs/2404.14469).
    :type snapkv_window_size int
)";

auto sparse_attention_config_docstring = R"(
    Configuration struct for the sparse attention functionality.
    :param mode: Sparse attention mode to be applied.
    :type mode: openvino_genai.SparseAttentionMode

    :param num_last_dense_tokens_in_prefill: TRISHAPE and XATTENTION modes - Number of tokens from the end of the prompt
       for which full attention across previous KV cache contents will be computed. In contrast, for the rest of the tokens
       in the prompt only the sparse attention will be computed according to the selected algorithm.
       TRISHAPE: Due to the block-wise nature of continuous batching cache management, the actual number of prompt tokens
       for which the dense attention will be computed may be up to block-size larger than this value (depending on the
       prompt length and block size).
       XATTENTION: Same as above applies, but the dense attention may overspill up to a subsequence chunk (i.e. multiple
       blocks)
    :type num_last_dense_tokens_in_prefill: int

    :param num_retained_start_tokens_in_cache: TRISHAPE mode only - The number of tokens in the beginning of the cache
     (least recent) to be retained when applying sparse attention. Must be a multiple of block size.
    :type num_retained_start_tokens_in_cache: int

    :param num_retained_recent_tokens_in_cache: TRISHAPE mode only - The number of most recent tokens in cache to be retained when
      applying sparse attention. Must be a multiple of block size.
    :param num_retained_recent_tokens_in_cache: int

    :param xattention_threshold: XATTENTION mode only - Cumulative importance score threshold to be compared against when
      determining blocks to exclude from the attention calculations in the block-sparse approach. Only the attention matrix
      blocks with highest importance score sum not exceeding this threshold will be taking part in the computations. The lower
      the threshold, the less computation will the main attention operation will take, and vice versa, with the corresponding
      potential impact on generation accuracy.
    :type xattention_threshold: float

    :param xattention_block_size: XATTENTION mode only - Block granularity, in tokens, with which the block-sparse attention
      calculation will be applied.
    :type xattention_block_size: int

    :param xattention_stride: XATTENTION mode only - The stride of antidiagonal sampling employed to calculate the importance
     scores of each `xattention_block_size`-sized block of the attention matrix before the actual attention calculation takes
     place.  Directly influences the overhead portion of the importance score computations - if full (dense) attention takes
     M time to be calculated, then the importance score calculation would be taking `M / xattention_stride` time as overhead.
    :type xattention_stride: int
)";
auto scheduler_config_docstring = R"(
    SchedulerConfig to construct ContinuousBatchingPipeline

    Parameters:
    max_num_batched_tokens:     a maximum number of tokens to batch (in contrast to max_batch_size which combines
        independent sequences, we consider total amount of tokens in a batch).
    num_kv_blocks:              total number of KV blocks available to scheduler logic.
    cache_size:                 total size of KV cache in GB.
    block_size:                 block size for KV cache.
    dynamic_split_fuse:         whether to split prompt / generate to different scheduling phases.

    vLLM-like settings:
    max_num_seqs:               max number of scheduled sequences (you can think of it as "max batch size").
    enable_prefix_caching:      Enable caching of KV-blocks.
        When turned on all previously calculated KV-caches are kept in memory for future usages.
        KV-caches can be overridden if KV-cache limit is reached, but blocks are not released.
        This results in more RAM usage, maximum RAM usage is determined by cache_size or num_kv_blocks parameters.
        When turned off only KV-cache required for batch calculation is kept in memory and
        when a sequence has finished generation its cache is released.
    use_cache_eviction:         Whether to use cache eviction during generation.
    cache_eviction_config       Cache eviction configuration struct.
    use_sparse_attention        Whether to use sparse attention during prefill.
    sparse_attention_config     Sparse attention configuration struct.
)";

auto generation_result_docstring = R"(
    GenerationResult stores resulting batched tokens and scores.

    Parameters:
    request_id:         obsolete when handle API is approved as handle will connect results with prompts.
    generation_ids:     in a generic case we have multiple generation results per initial prompt
        depending on sampling parameters (e.g. beam search or parallel sampling).
    scores:             scores.
    status:             status of generation. The following values are possible:
        RUNNING = 0 - Default status for ongoing generation.
        FINISHED = 1 - Status set when generation has been finished.
        IGNORED = 2 - Status set when generation run into out-of-memory condition and could not be continued.
        CANCEL = 3 - Status set when generation handle is cancelled. The last prompt and all generated tokens will be dropped from history, KV cache will include history but last step.
        STOP = 4 - Status set when generation handle is stopped. History will be kept, KV cache will include the last prompt and generated tokens.
        DROPPED_BY_HANDLE = STOP - Status set when generation handle is dropped. Deprecated. Please, use STOP instead.
    perf_metrics: Performance metrics for each generation result.
    extended_perf_metrics: performance pipeline specifics metrics,
                           applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
)";

auto pipeline_metrics_docstring = R"(
    Contains general pipeline metrics, either aggregated throughout the lifetime of the generation pipeline
    or measured at the previous generation step.

    :param requests: Number of requests to be processed by the pipeline.
    :type requests: int

    :param scheduled_requests:  Number of requests that were scheduled for processing at the previous step of the pipeline.
    :type scheduled_requests: int

    :param cache_usage: Percentage of KV cache usage in the last generation step.
    :type cache_usage: float

    :param max_cache_usage: Max KV cache usage during the lifetime of the pipeline in %
    :type max_cache_usage: float


    :param avg_cache_usage: Running average of the KV cache usage (in %) during the lifetime of the pipeline, with max window size of 1000 steps
    :type avg_cache_usage: float
)";

std::ostream& operator << (std::ostream& stream, const GenerationResult& generation_result) {
    stream << generation_result.m_request_id << std::endl;
    const bool has_scores = !generation_result.m_scores.empty();
    for (size_t i = 0; i < generation_result.m_generation_ids.size(); ++i) {
        stream << "{ ";
        if (has_scores)
            stream << generation_result.m_scores[i] << ", ";
        stream << generation_result.m_generation_ids[i] << " }" << std::endl;
    }
    return stream << std::endl;
}

py::object __call_cb_generate(ContinuousBatchingPipeline& pipe,
                              const std::variant<std::vector<ov::Tensor>, std::vector<std::string>, std::vector<ChatHistory>>& inputs,
                              const std::vector<ov::genai::GenerationConfig>& sampling_params,
                              const pyutils::PyBindStreamerVariant& py_streamer) {
    ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
    py::object results;

    std::visit(pyutils::overloaded {
    [&](std::vector<ov::Tensor> input_ids) {
        std::vector<ov::genai::EncodedGenerationResult> encoded_results;
        {
            py::gil_scoped_release rel;
            encoded_results = pipe.generate(input_ids, sampling_params, streamer);
        }  
        results = py::cast(encoded_results);
    },
    [&](std::vector<std::string> prompts) {
        std::vector<ov::genai::GenerationResult> generated_results;
        {
            py::gil_scoped_release rel;
            generated_results = pipe.generate(prompts, sampling_params, streamer);
        }  
        results = py::cast(generated_results);
    },
    [&](std::vector<ChatHistory> histories) {
        std::vector<ov::genai::GenerationResult> generated_results;
        {
            py::gil_scoped_release rel;
            generated_results = pipe.generate(histories, sampling_params, streamer);
        }  
        results = py::cast(generated_results);
    }},
    inputs);

    return results;
}

} // namespace

void init_continuous_batching_pipeline(py::module_& m) {
    py::enum_<ov::genai::GenerationStatus>(m, "GenerationStatus")
        .value("RUNNING", ov::genai::GenerationStatus::RUNNING)
        .value("FINISHED", ov::genai::GenerationStatus::FINISHED)
        .value("IGNORED", ov::genai::GenerationStatus::IGNORED)
        .value("CANCEL", ov::genai::GenerationStatus::CANCEL)
        .value("STOP", ov::genai::GenerationStatus::STOP);

    py::class_<GenerationResult>(m, "GenerationResult", generation_result_docstring)
        .def(py::init<>())
        .def_readonly("m_request_id", &GenerationResult::m_request_id)
        .def_property("m_generation_ids",
            [](GenerationResult &r) -> py::typing::List<py::str> {
                return pyutils::handle_utf8(r.m_generation_ids);
            },
            [](GenerationResult &r, std::vector<std::string> &generation_ids) {
                r.m_generation_ids = generation_ids;
            })
        .def_readwrite("m_scores", &GenerationResult::m_scores)
        .def_readwrite("m_status", &GenerationResult::m_status)
        .def_readonly("perf_metrics", &GenerationResult::perf_metrics)
        .def_readonly("extended_perf_metrics", &GenerationResult::extended_perf_metrics)
        .def("__repr__",
            [](const GenerationResult &r) -> py::str {
                std::stringstream stream;
                stream << "<py_continuous_batching.GenerationResult " << r << ">";
                return pyutils::handle_utf8(stream.str());
            }
        )
        .def("get_generation_ids",
        [](GenerationResult &r) -> py::typing::List<py::str> {
            return pyutils::handle_utf8(r.m_generation_ids);
        });

    py::class_<EncodedGenerationResult>(m, "EncodedGenerationResult", generation_result_docstring)
        .def(py::init<>())
        .def_readonly("m_request_id", &EncodedGenerationResult::m_request_id)
        .def_readwrite("m_generation_ids", &EncodedGenerationResult::m_generation_ids)
        .def_readwrite("m_scores", &EncodedGenerationResult::m_scores)
        .def_readonly("perf_metrics", &EncodedGenerationResult::perf_metrics)
        .def_readonly("extended_perf_metrics", &EncodedGenerationResult::extended_perf_metrics);

    py::enum_<ov::genai::GenerationFinishReason>(m, "GenerationFinishReason")
        .value("NONE", ov::genai::GenerationFinishReason::NONE)
        .value("STOP", ov::genai::GenerationFinishReason::STOP)
        .value("LENGTH", ov::genai::GenerationFinishReason::LENGTH);

    py::class_<GenerationOutput, std::shared_ptr<GenerationOutput>>(m, "GenerationOutput")
        .def_readwrite("generated_ids", &GenerationOutput::generated_ids)
        .def_readwrite("generated_log_probs", &GenerationOutput::generated_log_probs)
        .def_readwrite("score", &GenerationOutput::score)
        .def_readwrite("finish_reason", &GenerationOutput::finish_reason);

    auto generation_handle = py::class_<GenerationHandleImpl, std::shared_ptr<GenerationHandleImpl>>(m, "GenerationHandle")
        .def("get_status", &GenerationHandleImpl::get_status)
        .def("can_read", &GenerationHandleImpl::can_read)
        .def("stop", &GenerationHandleImpl::stop)
        .def("cancel", &GenerationHandleImpl::cancel)
        .def("read", &GenerationHandleImpl::read)
        .def("read_all", &GenerationHandleImpl::read_all);
    OPENVINO_SUPPRESS_DEPRECATED_START
    generation_handle.def("drop", &GenerationHandleImpl::drop);
    OPENVINO_SUPPRESS_DEPRECATED_END

    py::enum_<AggregationMode>(m, "AggregationMode",
                            R"(Represents the mode of per-token score aggregation when determining least important tokens for eviction from cache
                               :param AggregationMode.SUM: In this mode the importance scores of each token will be summed after each step of generation
                               :param AggregationMode.NORM_SUM: Same as SUM, but the importance scores are additionally divided by the lifetime (in tokens generated) of a given token in cache)")
            .value("SUM", AggregationMode::SUM)
            .value("NORM_SUM", AggregationMode::NORM_SUM);
    py::enum_<KVCrushAnchorPointMode>(m,
                                      "KVCrushAnchorPointMode",
                                      R"(Represents the anchor point types for KVCrush cache eviction
                  :param KVCrushAnchorPointMode.RANDOM: Random binary vector will be used as anchor point
                  :param KVCrushAnchorPointMode.ZEROS: Vector of all zeros will be used as anchor point
                  :param KVCrushAnchorPointMode.ONES: Vector of all ones will be used as anchor point
                  :param KVCrushAnchorPointMode.MEAN: Mean of indicator feature vector to be used as anchor point
                  :param KVCrushAnchorPointMode.ALTERNATE: Alternating 0s and 1s will be used as anchor point)")
        .value("RANDOM", KVCrushAnchorPointMode::RANDOM)
        .value("ZEROS", KVCrushAnchorPointMode::ZEROS)
        .value("ONES", KVCrushAnchorPointMode::ONES)
        .value("MEAN", KVCrushAnchorPointMode::MEAN)
        .value("ALTERNATE", KVCrushAnchorPointMode::ALTERNATE);

    py::class_<KVCrushConfig>(m, "KVCrushConfig", "Configuration for KVCrush cache eviction algorithm")
        .def(py::init<>(), "Default constructor")
        .def(py::init<size_t, KVCrushAnchorPointMode, size_t>(),
             "Constructor with budget, anchor point mode, and RNG seed",
             py::arg("budget"),
             py::arg_v("anchor_point_mode", KVCrushAnchorPointMode::RANDOM, "KVCrushAnchorPointMode.RANDOM"),
             py::arg("rng_seed") = 0)
        .def_readwrite("budget", &KVCrushConfig::budget)
        .def_readwrite("anchor_point_mode", &KVCrushConfig::anchor_point_mode)
        .def_readwrite("rng_seed", &KVCrushConfig::rng_seed)
        .def("to_string", &KVCrushConfig::to_string);

    py::class_<CacheEvictionConfig>(m, "CacheEvictionConfig", cache_eviction_config_docstring)
            .def(py::init<>([](const size_t start_size, size_t recent_size, size_t max_cache_size, AggregationMode aggregation_mode, bool apply_rotation,
                            size_t snapkv_window_size, py::object kvcrush_config) {
                if (kvcrush_config.is_none()) {
                     return CacheEvictionConfig{start_size, recent_size, max_cache_size, aggregation_mode, apply_rotation, snapkv_window_size};
                 } else {
                     const auto& config_ref = kvcrush_config.cast<const KVCrushConfig&>();
                     return CacheEvictionConfig{start_size,
                                                recent_size,
                                                max_cache_size,
                                                aggregation_mode,
                                                apply_rotation,
                                                snapkv_window_size,
                                                config_ref};
                 } }),
                 py::arg("start_size"), py::arg("recent_size"), py::arg("max_cache_size"), py::arg("aggregation_mode"), py::arg("apply_rotation") = false,
                 py::arg("snapkv_window_size") = 8, py::arg("kvcrush_config") = py::none())
            .def_readwrite("aggregation_mode", &CacheEvictionConfig::aggregation_mode)
            .def_readwrite("apply_rotation", &CacheEvictionConfig::apply_rotation)
            .def_readwrite("snapkv_window_size", &CacheEvictionConfig::snapkv_window_size)
            .def_readwrite("kvcrush_config", &CacheEvictionConfig::kvcrush_config)
            .def("get_start_size", &CacheEvictionConfig::get_start_size)
            .def("get_recent_size", &CacheEvictionConfig::get_recent_size)
            .def("get_max_cache_size", &CacheEvictionConfig::get_max_cache_size)
            .def("get_evictable_size", &CacheEvictionConfig::get_evictable_size)
            .def("to_string", &CacheEvictionConfig::to_string);

    py::enum_<SparseAttentionMode>(m, "SparseAttentionMode",
                            R"(Represents the mode of sparse attention applied during generation.
                               :param SparseAttentionMode.TRISHAPE: Sparse attention will be applied to prefill stage only, with a configurable number of start and recent cache tokens to be retained. A number of prefill tokens in the end of the prompt can be configured to have dense attention applied to them instead, to retain generation accuracy.
                               :param SparseAttentionMode.XATTENTION: Following https://arxiv.org/pdf/2503.16428, introduces importance score threshold-based block sparsity into the prefill stage.  Computing importance scores introduces an overhead, but the total inference time is expected to be reduced even more.
)")
            .value("TRISHAPE", SparseAttentionMode::TRISHAPE)
			.value("XATTENTION", SparseAttentionMode::XATTENTION);

    py::class_<SparseAttentionConfig>(m, "SparseAttentionConfig", sparse_attention_config_docstring)
            .def(py::init<>([](SparseAttentionMode mode, size_t num_last_dense_tokens_in_prefill, size_t num_retained_start_tokens_in_cache, size_t num_retained_recent_tokens_in_cache, float xattention_threshold, size_t xattention_block_size, size_t xattention_stride) {
                 // somehow pybind cannot associate enum arg with a default value with its python counterpart,
                 // hence need to use arg_v instead of arg like everywhere else
                return SparseAttentionConfig{mode, num_last_dense_tokens_in_prefill, num_retained_start_tokens_in_cache, num_retained_recent_tokens_in_cache, xattention_threshold, xattention_block_size, xattention_stride}; }),
                 py::arg_v("mode", SparseAttentionMode::TRISHAPE, "SparseAttentionMode.TRISHAPE"),
                 py::arg("num_last_dense_tokens_in_prefill") = 100,
                 py::arg("num_retained_start_tokens_in_cache") = 128,
                 py::arg("num_retained_recent_tokens_in_cache") = 1920,
                 py::arg("xattention_threshold") = 0.8,
                 py::arg("xattention_block_size") = 64,
                 py::arg("xattention_stride") = 8)
            .def_readwrite("mode", &SparseAttentionConfig::mode)
            .def_readwrite("num_last_dense_tokens_in_prefill", &SparseAttentionConfig::num_last_dense_tokens_in_prefill)
            .def_readwrite("num_retained_start_tokens_in_cache", &SparseAttentionConfig::num_retained_start_tokens_in_cache)
            .def_readwrite("num_retained_recent_tokens_in_cache", &SparseAttentionConfig::num_retained_recent_tokens_in_cache)
            .def_readwrite("xattention_threshold", &SparseAttentionConfig::xattention_threshold)
            .def_readwrite("xattention_block_size", &SparseAttentionConfig::xattention_block_size)
            .def_readwrite("xattention_stride", &SparseAttentionConfig::xattention_stride)
            .def("to_string", &SparseAttentionConfig::to_string);

    py::class_<SchedulerConfig>(m, "SchedulerConfig", scheduler_config_docstring)
        .def(py::init<>())
        .def_readwrite("max_num_batched_tokens", &SchedulerConfig::max_num_batched_tokens)
        .def_readwrite("num_kv_blocks", &SchedulerConfig::num_kv_blocks)
        .def_readwrite("cache_size", &SchedulerConfig::cache_size)
        .def_readwrite("dynamic_split_fuse", &SchedulerConfig::dynamic_split_fuse)
        .def_readwrite("max_num_seqs", &SchedulerConfig::max_num_seqs)
        .def_readwrite("enable_prefix_caching", &SchedulerConfig::enable_prefix_caching)
        .def_readwrite("use_cache_eviction", &SchedulerConfig::use_cache_eviction)
        .def_readwrite("cache_eviction_config", &SchedulerConfig::cache_eviction_config)
        .def_readwrite("use_sparse_attention", &SchedulerConfig::use_sparse_attention)
        .def_readwrite("sparse_attention_config", &SchedulerConfig::sparse_attention_config)
        .def("to_string", &SchedulerConfig::to_string);

    py::class_<PipelineMetrics>(m, "PipelineMetrics", pipeline_metrics_docstring)
            .def(py::init<>())
            .def_readonly("requests", &PipelineMetrics::requests)
            .def_readonly("scheduled_requests", &PipelineMetrics::scheduled_requests)
            .def_readonly("cache_usage", &PipelineMetrics::cache_usage)
            .def_readonly("avg_cache_usage", &PipelineMetrics::avg_cache_usage)
            .def_readonly("max_cache_usage", &PipelineMetrics::max_cache_usage);

    py::class_<ContinuousBatchingPipeline>(m, "ContinuousBatchingPipeline", "This class is used for generation with LLMs with continuous batchig")
        .def(py::init([](const std::filesystem::path& models_path, const SchedulerConfig& scheduler_config, const std::string& device, const std::map<std::string, py::object>& llm_plugin_config, 
                const std::map<std::string, py::object>& tokenizer_plugin_config, const std::map<std::string, py::object>& inputs_embedder_plugin_config) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ContinuousBatchingPipeline>(models_path, scheduler_config, device, pyutils::properties_to_any_map(llm_plugin_config), 
                pyutils::properties_to_any_map(tokenizer_plugin_config), pyutils::properties_to_any_map(inputs_embedder_plugin_config));
        }),
        py::arg("models_path"),
        py::arg("scheduler_config"),
        py::arg("device"),
        py::arg("properties") = ov::AnyMap({}),
        py::arg("tokenizer_properties") = ov::AnyMap({}),
        py::arg("vision_encoder_properties") = ov::AnyMap({}))

        .def(py::init([](const std::filesystem::path& models_path, const ov::genai::Tokenizer& tokenizer, const SchedulerConfig& scheduler_config, const std::string& device, const py::kwargs& kwargs) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ContinuousBatchingPipeline>(models_path, tokenizer, scheduler_config, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"),
        py::arg("tokenizer"),
        py::arg("scheduler_config"),
        py::arg("device"))

        .def("get_tokenizer", &ContinuousBatchingPipeline::get_tokenizer)
        .def("get_config", &ContinuousBatchingPipeline::get_config)
        .def("get_metrics", &ContinuousBatchingPipeline::get_metrics)
        .def("add_request", py::overload_cast<uint64_t, const ov::Tensor&, const ov::genai::GenerationConfig&>(&ContinuousBatchingPipeline::add_request), py::arg("request_id"), py::arg("input_ids"), py::arg("generation_config"))
        .def("add_request", py::overload_cast<uint64_t, const std::string&, const ov::genai::GenerationConfig&>(&ContinuousBatchingPipeline::add_request), py::arg("request_id"), py::arg("prompt"), py::arg("generation_config"))
        .def("add_request", py::overload_cast<uint64_t, const std::string&, const std::vector<ov::Tensor>&, const std::vector<ov::Tensor>&, const ov::genai::GenerationConfig&>(&ContinuousBatchingPipeline::add_request), py::arg("request_id"), py::arg("prompt"), py::arg("images"), py::arg("videos"), py::arg("generation_config"))
        .def("add_request", py::overload_cast<uint64_t, const std::string&, const std::vector<ov::Tensor>&, const ov::genai::GenerationConfig&>(&ContinuousBatchingPipeline::add_request), py::arg("request_id"), py::arg("prompt"), py::arg("images"), py::arg("generation_config"))
        .def("step", &ContinuousBatchingPipeline::step)
        .def("has_non_finished_requests", &ContinuousBatchingPipeline::has_non_finished_requests)

        .def("start_chat", &ContinuousBatchingPipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ContinuousBatchingPipeline::finish_chat)

        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::vector<ov::Tensor>& input_ids,
               const std::vector<ov::genai::GenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& streamer
            ) -> py::typing::Union<std::vector<ov::genai::EncodedGenerationResult>> {
                return __call_cb_generate(pipe, input_ids, generation_config, streamer);
            },
            py::arg("input_ids"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        )

        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::vector<std::string>& prompts,
               const std::vector<ov::genai::GenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& streamer
            ) -> py::typing::Union<std::vector<ov::genai::GenerationResult>> {
                return __call_cb_generate(pipe, prompts, generation_config, streamer);
            },
            py::arg("prompts"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        )
        
        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::string& prompt,
               const ov::genai::GenerationConfig& generation_config,
               const pyutils::PyBindStreamerVariant& streamer
            ) -> py::typing::Union<std::vector<ov::genai::GenerationResult>> {
                std::vector<std::string> prompts = { prompts };
                std::vector<ov::genai::GenerationConfig> generation_configs = { generation_config };
                return __call_cb_generate(pipe, prompts, generation_configs, streamer);
            },
            py::arg("prompt"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        )

        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::vector<ChatHistory>& histories,
               const std::vector<ov::genai::GenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& streamer
            ) -> py::typing::Union<std::vector<ov::genai::GenerationResult>> {
                return __call_cb_generate(pipe, histories, generation_config, streamer);
            },
            py::arg("prompts"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        )

        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::vector<std::string>& prompts,
               const std::vector<std::vector<ov::Tensor>>& images,
               const std::vector<std::vector<ov::Tensor>>& videos,
               const std::vector<ov::genai::GenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& py_streamer
            ) -> py::typing::Union<std::vector<ov::genai::GenerationResult>> {
                ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
                std::vector<ov::genai::VLMDecodedResults> generated_results;
                {
                    py::gil_scoped_release rel;
                    generated_results = pipe.generate(prompts, images, videos, generation_config, streamer);
                }  
                return py::cast(generated_results);
            },
            py::arg("prompts"),
            py::arg("images"),
            py::arg("videos"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        )
        .def(
            "generate",
            [](ContinuousBatchingPipeline& pipe,
               const std::vector<std::string>& prompts,
               const std::vector<std::vector<ov::Tensor>>& images,
               const std::vector<ov::genai::GenerationConfig>& generation_config,
               const pyutils::PyBindStreamerVariant& py_streamer
            ) -> py::typing::Union<std::vector<ov::genai::GenerationResult>> {
                ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
                std::vector<ov::genai::VLMDecodedResults> generated_results;
                {
                    py::gil_scoped_release rel;
                    generated_results = pipe.generate(prompts, images, generation_config, streamer);
                }  
                return py::cast(generated_results);
            },
            py::arg("prompts"),
            py::arg("images"),
            py::arg("generation_config"),
            py::arg("streamer") = std::monostate{}
        );
}
