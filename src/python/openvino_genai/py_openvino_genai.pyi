"""
Pybind11 binding for OpenVINO GenAI library
"""
from __future__ import annotations
import collections.abc
import openvino._pyopenvino
import typing
__all__: list[str] = ['Adapter', 'AdapterConfig', 'AggregationMode', 'AutoencoderKL', 'CLIPTextModel', 'CLIPTextModelWithProjection', 'CacheEvictionConfig', 'ChatHistory', 'ChunkStreamerBase', 'ContinuousBatchingPipeline', 'CppStdGenerator', 'DecodedResults', 'EncodedGenerationResult', 'EncodedResults', 'ExtendedPerfMetrics', 'FluxTransformer2DModel', 'GenerationConfig', 'GenerationFinishReason', 'GenerationHandle', 'GenerationOutput', 'GenerationResult', 'GenerationStatus', 'Generator', 'Image2ImagePipeline', 'ImageGenerationConfig', 'ImageGenerationPerfMetrics', 'InpaintingPipeline', 'KVCrushAnchorPointMode', 'KVCrushConfig', 'LLMPipeline', 'MeanStdPair', 'PerfMetrics', 'PipelineMetrics', 'RawImageGenerationPerfMetrics', 'RawPerfMetrics', 'SD3Transformer2DModel', 'SDPerModelsPerfMetrics', 'SDPerfMetrics', 'Scheduler', 'SchedulerConfig', 'SparseAttentionConfig', 'SparseAttentionMode', 'SpeechGenerationConfig', 'SpeechGenerationPerfMetrics', 'StopCriteria', 'StreamerBase', 'StreamingStatus', 'StructuralTagItem', 'StructuralTagsConfig', 'StructuredOutputConfig', 'SummaryStats', 'T5EncoderModel', 'Text2ImagePipeline', 'Text2SpeechDecodedResults', 'Text2SpeechPipeline', 'TextEmbeddingPipeline', 'TextRerankPipeline', 'TextStreamer', 'TokenizedInputs', 'Tokenizer', 'TorchGenerator', 'UNet2DConditionModel', 'VLMDecodedResults', 'VLMPerfMetrics', 'VLMPipeline', 'VLMRawPerfMetrics', 'WhisperDecodedResultChunk', 'WhisperDecodedResults', 'WhisperGenerationConfig', 'WhisperPerfMetrics', 'WhisperPipeline', 'WhisperRawPerfMetrics', 'draft_model', 'get_version']
class Adapter:
    """
    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
    """
    def __bool__(self) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, path: os.PathLike | str | bytes) -> None:
        """
                    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
                    path (os.PathLike): Path to adapter file in safetensors format.
        """
    @typing.overload
    def __init__(self, safetensor: openvino._pyopenvino.Tensor) -> None:
        """
                    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
                    safetensor (ov.Tensor): Pre-read LoRA Adapter safetensor.
        """
class AdapterConfig:
    """
    Adapter config that defines a combination of LoRA adapters with blending parameters.
    """
    class Mode:
        """
        Members:
        
          MODE_AUTO
        
          MODE_DYNAMIC
        
          MODE_STATIC_RANK
        
          MODE_STATIC
        
          MODE_FUSE
        """
        MODE_AUTO: typing.ClassVar[AdapterConfig.Mode]  # value = <Mode.MODE_AUTO: 0>
        MODE_DYNAMIC: typing.ClassVar[AdapterConfig.Mode]  # value = <Mode.MODE_DYNAMIC: 1>
        MODE_FUSE: typing.ClassVar[AdapterConfig.Mode]  # value = <Mode.MODE_FUSE: 4>
        MODE_STATIC: typing.ClassVar[AdapterConfig.Mode]  # value = <Mode.MODE_STATIC: 3>
        MODE_STATIC_RANK: typing.ClassVar[AdapterConfig.Mode]  # value = <Mode.MODE_STATIC_RANK: 2>
        __members__: typing.ClassVar[dict[str, AdapterConfig.Mode]]  # value = {'MODE_AUTO': <Mode.MODE_AUTO: 0>, 'MODE_DYNAMIC': <Mode.MODE_DYNAMIC: 1>, 'MODE_STATIC_RANK': <Mode.MODE_STATIC_RANK: 2>, 'MODE_STATIC': <Mode.MODE_STATIC: 3>, 'MODE_FUSE': <Mode.MODE_FUSE: 4>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __bool__(self) -> bool:
        ...
    @typing.overload
    def __init__(self, mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapter: Adapter, alpha: typing.SupportsFloat, mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapter: Adapter, mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapters: collections.abc.Sequence[Adapter], mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapters: collections.abc.Sequence[tuple[Adapter, typing.SupportsFloat]], mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def add(self, adapter: Adapter, alpha: typing.SupportsFloat) -> AdapterConfig:
        ...
    @typing.overload
    def add(self, adapter: Adapter) -> AdapterConfig:
        ...
    def get_adapters(self) -> list[Adapter]:
        ...
    def get_adapters_and_alphas(self) -> list[tuple[Adapter, float]]:
        ...
    def get_alpha(self, adapter: Adapter) -> float:
        ...
    def remove(self, adapter: Adapter) -> AdapterConfig:
        ...
    def set_adapters_and_alphas(self, adapters: collections.abc.Sequence[tuple[Adapter, typing.SupportsFloat]]) -> None:
        ...
    def set_alpha(self, adapter: Adapter, alpha: typing.SupportsFloat) -> AdapterConfig:
        ...
class AggregationMode:
    """
    Represents the mode of per-token score aggregation when determining least important tokens for eviction from cache
                                   :param AggregationMode.SUM: In this mode the importance scores of each token will be summed after each step of generation
                                   :param AggregationMode.NORM_SUM: Same as SUM, but the importance scores are additionally divided by the lifetime (in tokens generated) of a given token in cache
    
    Members:
    
      SUM
    
      NORM_SUM
    """
    NORM_SUM: typing.ClassVar[AggregationMode]  # value = <AggregationMode.NORM_SUM: 1>
    SUM: typing.ClassVar[AggregationMode]  # value = <AggregationMode.SUM: 0>
    __members__: typing.ClassVar[dict[str, AggregationMode]]  # value = {'SUM': <AggregationMode.SUM: 0>, 'NORM_SUM': <AggregationMode.NORM_SUM: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AutoencoderKL:
    """
    AutoencoderKL class.
    """
    class Config:
        """
        This class is used for storing AutoencoderKL config.
        """
        def __init__(self, config_path: os.PathLike | str | bytes) -> None:
            ...
        @property
        def block_out_channels(self) -> list[int]:
            ...
        @block_out_channels.setter
        def block_out_channels(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
            ...
        @property
        def in_channels(self) -> int:
            ...
        @in_channels.setter
        def in_channels(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def latent_channels(self) -> int:
            ...
        @latent_channels.setter
        def latent_channels(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def out_channels(self) -> int:
            ...
        @out_channels.setter
        def out_channels(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def scaling_factor(self) -> float:
            ...
        @scaling_factor.setter
        def scaling_factor(self, arg0: typing.SupportsFloat) -> None:
            ...
    @typing.overload
    def __init__(self, vae_decoder_path: os.PathLike | str | bytes) -> None:
        """
                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
        """
    @typing.overload
    def __init__(self, vae_encoder_path: os.PathLike | str | bytes, vae_decoder_path: os.PathLike | str | bytes) -> None:
        """
                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (os.PathLike): VAE encoder directory.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
        """
    @typing.overload
    def __init__(self, vae_decoder_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, vae_encoder_path: os.PathLike | str | bytes, vae_decoder_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    AutoencoderKL class initialized only with both encoder and decoder models.
                    vae_encoder_path (os.PathLike): VAE encoder directory.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: AutoencoderKL) -> None:
        """
        AutoencoderKL model
                    AutoencoderKL class.
                    model (AutoencoderKL): AutoencoderKL model.
        """
    def compile(self, device: str, **kwargs) -> None:
        """
        device on which inference will be done
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def encode(self, image: openvino._pyopenvino.Tensor, generator: Generator) -> openvino._pyopenvino.Tensor:
        ...
    def export_model(self, export_path: os.PathLike | str | bytes) -> None:
        """
                        Exports compiled models to a specified directory. Can significantly reduce model load time, especially for large models.
                        export_path (os.PathLike): A path to a directory to export compiled models to.
        
                        Use `blob_path` property to load previously exported models.
        """
    def get_config(self) -> AutoencoderKL.Config:
        ...
    def get_vae_scale_factor(self) -> int:
        ...
    def reshape(self, batch_size: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt) -> AutoencoderKL:
        ...
class CLIPTextModel:
    """
    CLIPTextModel class.
    """
    class Config:
        """
        This class is used for storing CLIPTextModel config.
        """
        def __init__(self, config_path: os.PathLike | str | bytes) -> None:
            ...
        @property
        def max_position_embeddings(self) -> int:
            ...
        @max_position_embeddings.setter
        def max_position_embeddings(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def num_hidden_layers(self) -> int:
            ...
        @num_hidden_layers.setter
        def num_hidden_layers(self, arg0: typing.SupportsInt) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    CLIPTextModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    CLIPTextModel class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: CLIPTextModel) -> None:
        """
        CLIPText model
                    CLIPTextModel class
                    model (CLIPTextModel): CLIPText model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def export_model(self, export_path: os.PathLike | str | bytes) -> None:
        """
                        Exports compiled model to a specified directory. Can significantly reduce model load time, especially for large models.
                        export_path (os.PathLike): A path to a directory to export compiled model to.
        
                        Use `blob_path` property to load previously exported models.
        """
    def get_config(self) -> CLIPTextModel.Config:
        ...
    def get_output_tensor(self, idx: typing.SupportsInt) -> openvino._pyopenvino.Tensor:
        ...
    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: typing.SupportsInt) -> CLIPTextModel:
        ...
    def set_adapters(self, adapters: openvino_genai.py_openvino_genai.AdapterConfig | None) -> None:
        ...
class CLIPTextModelWithProjection(CLIPTextModel):
    """
    CLIPTextModelWithProjection class.
    """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    CLIPTextModelWithProjection class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    CLIPTextModelWithProjection class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: CLIPTextModelWithProjection) -> None:
        """
        CLIPText model
                    CLIPTextModelWithProjection class
                    model (CLIPTextModelWithProjection): CLIPText model with projection
        """
class CacheEvictionConfig:
    """
    
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
    """
    aggregation_mode: AggregationMode
    apply_rotation: bool
    kvcrush_config: KVCrushConfig
    def __init__(self, start_size: typing.SupportsInt, recent_size: typing.SupportsInt, max_cache_size: typing.SupportsInt, aggregation_mode: AggregationMode, apply_rotation: bool = False, snapkv_window_size: typing.SupportsInt = 8, kvcrush_config: typing.Any = None) -> None:
        ...
    def get_evictable_size(self) -> int:
        ...
    def get_max_cache_size(self) -> int:
        ...
    def get_recent_size(self) -> int:
        ...
    def get_start_size(self) -> int:
        ...
    def to_string(self) -> str:
        ...
    @property
    def snapkv_window_size(self) -> int:
        ...
    @snapkv_window_size.setter
    def snapkv_window_size(self, arg0: typing.SupportsInt) -> None:
        ...
class ChatHistory:
    """
    
        ChatHistory stores conversation messages and optional metadata for chat templates.
    
        Manages:
        - Message history (array of message objects)
        - Optional tools definitions array (for function calling)
        - Optional extra context object (for custom template variables)
    
        Messages are stored as JSON-like structures but accessed as Python dicts.
        Use get_messages() to retrieve the list of all messages, modify them,
        and set_messages() to update the history.
    
        Example:
            ```python
            history = ChatHistory()
            history.append({"role": "user", "content": "Hello"})
            
            # Modify messages
            messages = history.get_messages()
            messages[0]["content"] = "Updated"
            history.set_messages(messages)
            ```
    """
    def __bool__(self) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Create an empty chat history.
        """
    @typing.overload
    def __init__(self, messages: list) -> None:
        """
        Create chat history from a list of message dicts.
        """
    def __len__(self) -> int:
        ...
    def append(self, message: dict) -> None:
        """
        Add a message to the end of chat history.
        """
    def clear(self) -> None:
        ...
    def get_extra_context(self) -> dict:
        """
        Get the extra context object.
        """
    def get_messages(self) -> list:
        """
        Get all messages as a list of dicts (deep copy).
        """
    def get_tools(self) -> list:
        """
        Get the tools definitions array.
        """
    def pop(self) -> dict:
        """
        Remove and return the last message.
        """
    def set_extra_context(self, extra_context: dict) -> None:
        """
        Set the extra context object.
        """
    def set_messages(self, messages: list) -> None:
        """
        Replace all messages with a new list.
        """
    def set_tools(self, tools: list) -> None:
        """
        Set the tools definitions array.
        """
class ChunkStreamerBase(StreamerBase):
    """
    
        Base class for chunk streamers. In order to use inherit from from this class.
    """
    def __init__(self) -> None:
        ...
    def end(self) -> None:
        """
        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """
    def put(self, token: typing.SupportsInt) -> bool:
        """
        Put is called every time new token is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
    def put_chunk(self, tokens: collections.abc.Sequence[typing.SupportsInt]) -> bool:
        """
        put_chunk is called every time new token chunk is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
class ContinuousBatchingPipeline:
    """
    This class is used for generation with LLMs with continuous batchig
    """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, scheduler_config: SchedulerConfig, device: str, properties: collections.abc.Mapping[str, typing.Any] = {}, tokenizer_properties: collections.abc.Mapping[str, typing.Any] = {}, vision_encoder_properties: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, tokenizer: Tokenizer, scheduler_config: SchedulerConfig, device: str, **kwargs) -> None:
        ...
    @typing.overload
    def add_request(self, request_id: typing.SupportsInt, input_ids: openvino._pyopenvino.Tensor, generation_config: GenerationConfig) -> GenerationHandle:
        ...
    @typing.overload
    def add_request(self, request_id: typing.SupportsInt, prompt: str, generation_config: GenerationConfig) -> GenerationHandle:
        ...
    @typing.overload
    def add_request(self, request_id: typing.SupportsInt, prompt: str, images: collections.abc.Sequence[openvino._pyopenvino.Tensor], videos: collections.abc.Sequence[openvino._pyopenvino.Tensor], generation_config: GenerationConfig) -> GenerationHandle:
        ...
    @typing.overload
    def add_request(self, request_id: typing.SupportsInt, prompt: str, images: collections.abc.Sequence[openvino._pyopenvino.Tensor], generation_config: GenerationConfig) -> GenerationHandle:
        ...
    def finish_chat(self) -> None:
        ...
    @typing.overload
    def generate(self, input_ids: collections.abc.Sequence[openvino._pyopenvino.Tensor], generation_config: collections.abc.Sequence[GenerationConfig], streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[EncodedGenerationResult]:
        ...
    @typing.overload
    def generate(self, prompts: collections.abc.Sequence[str], generation_config: collections.abc.Sequence[GenerationConfig], streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[GenerationResult]:
        ...
    @typing.overload
    def generate(self, prompt: str, generation_config: GenerationConfig, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[GenerationResult]:
        ...
    @typing.overload
    def generate(self, prompts: collections.abc.Sequence[ChatHistory], generation_config: collections.abc.Sequence[GenerationConfig], streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[GenerationResult]:
        ...
    @typing.overload
    def generate(self, prompts: collections.abc.Sequence[str], images: collections.abc.Sequence[collections.abc.Sequence[openvino._pyopenvino.Tensor]], videos: collections.abc.Sequence[collections.abc.Sequence[openvino._pyopenvino.Tensor]], generation_config: collections.abc.Sequence[GenerationConfig], streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[GenerationResult]:
        ...
    @typing.overload
    def generate(self, prompts: collections.abc.Sequence[str], images: collections.abc.Sequence[collections.abc.Sequence[openvino._pyopenvino.Tensor]], generation_config: collections.abc.Sequence[GenerationConfig], streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None) -> list[GenerationResult]:
        ...
    def get_config(self) -> GenerationConfig:
        ...
    def get_metrics(self) -> PipelineMetrics:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def has_non_finished_requests(self) -> bool:
        ...
    def start_chat(self, system_message: str = '') -> None:
        ...
    def step(self) -> None:
        ...
class CppStdGenerator(Generator):
    """
    This class wraps std::mt19937 pseudo-random generator.
    """
    def __init__(self, seed: typing.SupportsInt) -> None:
        ...
    def next(self) -> float:
        ...
    def randn_tensor(self, shape: openvino._pyopenvino.Shape) -> openvino._pyopenvino.Tensor:
        ...
    def seed(self, new_seed: typing.SupportsInt) -> None:
        ...
class DecodedResults:
    """
    
        Structure to store resulting batched text outputs and scores for each batch.
        The first num_return_sequences elements correspond to the first batch element.
    
        Parameters: 
        texts:      vector of resulting sequences.
        scores:     scores for each sequence.
        metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
        extended_perf_metrics: performance pipeline specifics metrics,
                               applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
    """
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def extended_perf_metrics(self) -> ExtendedPerfMetrics:
        ...
    @property
    def perf_metrics(self) -> PerfMetrics:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def texts(self) -> list[str]:
        ...
class EncodedGenerationResult:
    """
    
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
    """
    def __init__(self) -> None:
        ...
    @property
    def extended_perf_metrics(self) -> ExtendedPerfMetrics:
        ...
    @property
    def m_generation_ids(self) -> list[list[int]]:
        ...
    @m_generation_ids.setter
    def m_generation_ids(self, arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]) -> None:
        ...
    @property
    def m_request_id(self) -> int:
        ...
    @property
    def m_scores(self) -> list[float]:
        ...
    @m_scores.setter
    def m_scores(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @property
    def perf_metrics(self) -> PerfMetrics:
        ...
class EncodedResults:
    """
    
        Structure to store resulting batched tokens and scores for each batch sequence.
        The first num_return_sequences elements correspond to the first batch element.
        In the case if results decoded with beam search and random sampling scores contain
        sum of logarithmic probabilities for each token in the sequence. In the case
        of greedy decoding scores are filled with zeros.
    
        Parameters: 
        tokens: sequence of resulting tokens.
        scores: sum of logarithmic probabilities of all tokens in the sequence.
        metrics: performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
        extended_perf_metrics: performance pipeline specifics metrics,
                               applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
    """
    @property
    def extended_perf_metrics(self) -> ExtendedPerfMetrics:
        ...
    @property
    def perf_metrics(self) -> PerfMetrics:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def tokens(self) -> list[list[int]]:
        ...
class ExtendedPerfMetrics:
    """
    
        Holds performance metrics for each generate call.
    
        PerfMetrics holds the following metrics with mean and standard deviations:
        - Time To the First Token (TTFT), ms
        - Time per Output Token (TPOT), ms/token
        - Inference time per Output Token (IPOT), ms/token
        - Generate total duration, ms
        - Inference duration, ms
        - Tokenization duration, ms
        - Detokenization duration, ms
        - Throughput, tokens/s
    
        Additional metrics include:
        - Load time, ms
        - Number of generated tokens
        - Number of tokens in the input prompt
        - Time to initialize grammar compiler for each backend, ms
        - Time to compile grammar, ms
    
        Preferable way to access metrics is via getter methods. Getter methods calculate mean and std values from raw_metrics and return pairs.
        If mean and std were already calculated, getters return cached values.
    
        :param get_load_time: Returns the load time in milliseconds.
        :type get_load_time: float
    
        :param get_num_generated_tokens: Returns the number of generated tokens.
        :type get_num_generated_tokens: int
    
        :param get_num_input_tokens: Returns the number of tokens in the input prompt.
        :type get_num_input_tokens: int
    
        :param get_ttft: Returns the mean and standard deviation of TTFT in milliseconds.
        :type get_ttft: MeanStdPair
    
        :param get_tpot: Returns the mean and standard deviation of TPOT in milliseconds.
        :type get_tpot: MeanStdPair
    
        :param get_ipot: Returns the mean and standard deviation of IPOT in milliseconds.
        :type get_ipot: MeanStdPair
    
        :param get_throughput: Returns the mean and standard deviation of throughput in tokens per second.
        :type get_throughput: MeanStdPair
    
        :param get_inference_duration: Returns the mean and standard deviation of the time spent on model inference during generate call in milliseconds.
        :type get_inference_duration: MeanStdPair
    
        :param get_generate_duration: Returns the mean and standard deviation of generate durations in milliseconds.
        :type get_generate_duration: MeanStdPair
    
        :param get_tokenization_duration: Returns the mean and standard deviation of tokenization durations in milliseconds.
        :type get_tokenization_duration: MeanStdPair
    
        :param get_detokenization_duration: Returns the mean and standard deviation of detokenization durations in milliseconds.
        :type get_detokenization_duration: MeanStdPair
    
        :param get_grammar_compiler_init_times: Returns a map with the time to initialize the grammar compiler for each backend in milliseconds.
        :type get_grammar_compiler_init_times: dict[str, float]
    
        :param get_grammar_compile_time: Returns the mean, standard deviation, min, and max of grammar compile times in milliseconds.
        :type get_grammar_compile_time: SummaryStats
    
        :param raw_metrics: A structure of RawPerfMetrics type that holds raw metrics.
        :type raw_metrics: RawPerfMetrics
    """
    def __add__(self, metrics: PerfMetrics) -> PerfMetrics:
        ...
    def __iadd__(self, right: PerfMetrics) -> PerfMetrics:
        ...
    def __init__(self) -> None:
        ...
    def get_detokenization_duration(self) -> MeanStdPair:
        ...
    def get_generate_duration(self) -> MeanStdPair:
        ...
    def get_inference_duration(self) -> MeanStdPair:
        ...
    def get_ipot(self) -> MeanStdPair:
        ...
    def get_load_time(self) -> float:
        ...
    def get_num_generated_tokens(self) -> int:
        ...
    def get_num_input_tokens(self) -> int:
        ...
    def get_throughput(self) -> MeanStdPair:
        ...
    def get_tokenization_duration(self) -> MeanStdPair:
        ...
    def get_tpot(self) -> MeanStdPair:
        ...
    def get_ttft(self) -> MeanStdPair:
        ...
    @property
    def raw_metrics(self) -> RawPerfMetrics:
        ...
class FluxTransformer2DModel:
    """
    FluxTransformer2DModel class.
    """
    class Config:
        """
        This class is used for storing FluxTransformer2DModel config.
        """
        def __init__(self, config_path: os.PathLike | str | bytes) -> None:
            ...
        @property
        def default_sample_size(self) -> int:
            ...
        @default_sample_size.setter
        def default_sample_size(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def in_channels(self) -> int:
            ...
        @in_channels.setter
        def in_channels(self, arg0: typing.SupportsInt) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    FluxTransformer2DModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    UNet2DConditionModel class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: FluxTransformer2DModel) -> None:
        """
        FluxTransformer2DModel model
                    FluxTransformer2DModel class
                    model (FluxTransformer2DModel): FluxTransformer2DModel model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def get_config(self) -> FluxTransformer2DModel.Config:
        ...
    def infer(self, latent: openvino._pyopenvino.Tensor, timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, tokenizer_model_max_length: typing.SupportsInt) -> FluxTransformer2DModel:
        ...
    def set_hidden_states(self, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None:
        ...
class GenerationConfig:
    """
    
        Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
        and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
        be used while greedy and beam search parameters will not affect decoding at all.
    
        Parameters:
        max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                       max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
        min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
        ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
        eos_token_id:  token_id of <eos> (end of sentence)
        stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
        include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
        stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
        echo:           if set to true, the model will echo the prompt in the output.
        logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                        Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
        apply_chat_template: whether to apply chat_template for non-chat scenarios
    
        repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
        presence_penalty: reduces absolute log prob if the token was generated at least once.
        frequency_penalty: reduces absolute log prob as many times as the token was generated.
    
        Beam search specific parameters:
        num_beams:         number of beams for beam search. 1 disables beam search.
        num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
        diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
        length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
            length_penalty < 0.0 encourages shorter sequences.
        num_return_sequences: the number of sequences to return for grouped beam search decoding.
        no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
        stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
            "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
            "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
            "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
    
        Random sampling parameters:
        temperature:        the value used to modulate token probabilities for random sampling.
        top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
        do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
        num_return_sequences: the number of sequences to generate from a single prompt.
    """
    adapters: openvino_genai.py_openvino_genai.AdapterConfig | None
    apply_chat_template: bool
    do_sample: bool
    echo: bool
    ignore_eos: bool
    include_stop_str_in_output: bool
    stop_criteria: StopCriteria
    structured_output_config: openvino_genai.py_openvino_genai.StructuredOutputConfig | None
    @typing.overload
    def __init__(self, json_path: os.PathLike | str | bytes) -> None:
        """
        path where generation_config.json is stored
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def is_assisting_generation(self) -> bool:
        ...
    def is_beam_search(self) -> bool:
        ...
    def is_greedy_decoding(self) -> bool:
        ...
    def is_multinomial(self) -> bool:
        ...
    def is_prompt_lookup(self) -> bool:
        ...
    def set_eos_token_id(self, tokenizer_eos_token_id: typing.SupportsInt) -> None:
        ...
    def update_generation_config(self, **kwargs) -> None:
        ...
    def validate(self) -> None:
        ...
    @property
    def assistant_confidence_threshold(self) -> float:
        ...
    @assistant_confidence_threshold.setter
    def assistant_confidence_threshold(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def diversity_penalty(self) -> float:
        ...
    @diversity_penalty.setter
    def diversity_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def eos_token_id(self) -> int:
        ...
    @eos_token_id.setter
    def eos_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def frequency_penalty(self) -> float:
        ...
    @frequency_penalty.setter
    def frequency_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def length_penalty(self) -> float:
        ...
    @length_penalty.setter
    def length_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def logprobs(self) -> int:
        ...
    @logprobs.setter
    def logprobs(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_length(self) -> int:
        ...
    @max_length.setter
    def max_length(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_new_tokens(self) -> int:
        ...
    @max_new_tokens.setter
    def max_new_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_ngram_size(self) -> int:
        ...
    @max_ngram_size.setter
    def max_ngram_size(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def min_new_tokens(self) -> int:
        ...
    @min_new_tokens.setter
    def min_new_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def no_repeat_ngram_size(self) -> int:
        ...
    @no_repeat_ngram_size.setter
    def no_repeat_ngram_size(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_assistant_tokens(self) -> int:
        ...
    @num_assistant_tokens.setter
    def num_assistant_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_beam_groups(self) -> int:
        ...
    @num_beam_groups.setter
    def num_beam_groups(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_beams(self) -> int:
        ...
    @num_beams.setter
    def num_beams(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_return_sequences(self) -> int:
        ...
    @num_return_sequences.setter
    def num_return_sequences(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def presence_penalty(self) -> float:
        ...
    @presence_penalty.setter
    def presence_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def repetition_penalty(self) -> float:
        ...
    @repetition_penalty.setter
    def repetition_penalty(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def rng_seed(self) -> int:
        ...
    @rng_seed.setter
    def rng_seed(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def stop_strings(self) -> set[str]:
        ...
    @stop_strings.setter
    def stop_strings(self, arg0: collections.abc.Set[str]) -> None:
        ...
    @property
    def stop_token_ids(self) -> set[int]:
        ...
    @stop_token_ids.setter
    def stop_token_ids(self, arg0: collections.abc.Set[typing.SupportsInt]) -> None:
        ...
    @property
    def temperature(self) -> float:
        ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def top_k(self) -> int:
        ...
    @top_k.setter
    def top_k(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def top_p(self) -> float:
        ...
    @top_p.setter
    def top_p(self, arg0: typing.SupportsFloat) -> None:
        ...
class GenerationFinishReason:
    """
    Members:
    
      NONE
    
      STOP
    
      LENGTH
    """
    LENGTH: typing.ClassVar[GenerationFinishReason]  # value = <GenerationFinishReason.LENGTH: 2>
    NONE: typing.ClassVar[GenerationFinishReason]  # value = <GenerationFinishReason.NONE: 0>
    STOP: typing.ClassVar[GenerationFinishReason]  # value = <GenerationFinishReason.STOP: 1>
    __members__: typing.ClassVar[dict[str, GenerationFinishReason]]  # value = {'NONE': <GenerationFinishReason.NONE: 0>, 'STOP': <GenerationFinishReason.STOP: 1>, 'LENGTH': <GenerationFinishReason.LENGTH: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GenerationHandle:
    def can_read(self) -> bool:
        ...
    def cancel(self) -> None:
        ...
    def drop(self) -> None:
        ...
    def get_status(self) -> GenerationStatus:
        ...
    def read(self) -> dict[int, GenerationOutput]:
        ...
    def read_all(self) -> list[GenerationOutput]:
        ...
    def stop(self) -> None:
        ...
class GenerationOutput:
    finish_reason: GenerationFinishReason
    @property
    def generated_ids(self) -> list[int]:
        ...
    @generated_ids.setter
    def generated_ids(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def generated_log_probs(self) -> list[float]:
        ...
    @generated_log_probs.setter
    def generated_log_probs(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @property
    def score(self) -> float:
        ...
    @score.setter
    def score(self, arg0: typing.SupportsFloat) -> None:
        ...
class GenerationResult:
    """
    
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
    """
    m_status: GenerationStatus
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_generation_ids(self) -> list[str]:
        ...
    @property
    def extended_perf_metrics(self) -> ExtendedPerfMetrics:
        ...
    @property
    def m_generation_ids(self) -> list[str]:
        ...
    @m_generation_ids.setter
    def m_generation_ids(self, arg1: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def m_request_id(self) -> int:
        ...
    @property
    def m_scores(self) -> list[float]:
        ...
    @m_scores.setter
    def m_scores(self, arg0: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @property
    def perf_metrics(self) -> PerfMetrics:
        ...
class GenerationStatus:
    """
    Members:
    
      RUNNING
    
      FINISHED
    
      IGNORED
    
      CANCEL
    
      STOP
    """
    CANCEL: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.CANCEL: 3>
    FINISHED: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.FINISHED: 1>
    IGNORED: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.IGNORED: 2>
    RUNNING: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.RUNNING: 0>
    STOP: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.STOP: 4>
    __members__: typing.ClassVar[dict[str, GenerationStatus]]  # value = {'RUNNING': <GenerationStatus.RUNNING: 0>, 'FINISHED': <GenerationStatus.FINISHED: 1>, 'IGNORED': <GenerationStatus.IGNORED: 2>, 'CANCEL': <GenerationStatus.CANCEL: 3>, 'STOP': <GenerationStatus.STOP: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Generator:
    """
    This class is used for storing pseudo-random generator.
    """
    def __init__(self) -> None:
        ...
class Image2ImagePipeline:
    """
    This class is used for generation with image-to-image models.
    """
    @staticmethod
    def flux(scheduler: Scheduler, clip_text_model: CLIPTextModel, t5_encoder_model: T5EncoderModel, transformer: FluxTransformer2DModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @staticmethod
    def latent_consistency_model(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, t5_encoder_model: T5EncoderModel, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion_xl(scheduler: Scheduler, clip_text_model: CLIPTextModel, clip_text_model_with_projection: CLIPTextModelWithProjection, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Image2ImagePipeline:
        ...
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes) -> None:
        """
                    Image2ImagePipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    Image2ImagePipeline class constructor.
                    models_path (os.PathLike): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: Image2ImagePipeline properties
        """
    @typing.overload
    def __init__(self, pipe: InpaintingPipeline) -> None:
        ...
    @typing.overload
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    @typing.overload
    def compile(self, text_encode_device: str, denoise_device: str, vae_device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        text_encode_device (str): Device to run the text encoder(s) on (e.g., CPU, GPU).
                        denoise_device (str): Device to run denoise steps on.
                        vae_device (str): Device to run vae encoder / decoder on.
                        kwargs: Device properties.
        """
    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def generate(self, prompt: str, image: openvino._pyopenvino.Tensor, **kwargs) -> openvino._pyopenvino.Tensor:
        """
            Generates images for text-to-image models.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.
        
            Expected parameters list:
            prompt_2: str - second prompt,
            prompt_3: str - third prompt,
            negative_prompt: str - negative prompt,
            negative_prompt_2: str - second negative prompt,
            negative_prompt_3: str - third negative prompt,
            num_images_per_prompt: int - number of images, that should be generated per prompt,
            guidance_scale: float - guidance scale,
            generation_config: GenerationConfig,
            height: int - height of resulting images,
            width: int - width of resulting images,
            num_inference_steps: int - number of inference steps,
            rng_seed: int - a seed for random numbers generator,
            generator: openvino_genai.TorchGenerator, openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator,
            adapters: LoRA adapters,
            strength: strength for image to image generation. 1.0f means initial image is fully noised,
            max_sequence_length: int - length of t5_encoder_model input
        
            :return: ov.Tensor with resulting images
            :rtype: ov.Tensor
        """
    def get_generation_config(self) -> ImageGenerationConfig:
        ...
    def get_performance_metrics(self) -> ImageGenerationPerfMetrics:
        ...
    def reshape(self, num_images_per_prompt: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, guidance_scale: typing.SupportsFloat) -> None:
        ...
    def set_generation_config(self, config: ImageGenerationConfig) -> None:
        ...
    def set_scheduler(self, scheduler: Scheduler) -> None:
        ...
class ImageGenerationConfig:
    """
    This class is used for storing generation config for image generation pipeline.
    """
    adapters: openvino_genai.py_openvino_genai.AdapterConfig | None
    generator: Generator
    negative_prompt: str | None
    negative_prompt_2: str | None
    negative_prompt_3: str | None
    prompt_2: str | None
    prompt_3: str | None
    def __init__(self) -> None:
        ...
    def update_generation_config(self, **kwargs) -> None:
        ...
    def validate(self) -> None:
        ...
    @property
    def guidance_scale(self) -> float:
        ...
    @guidance_scale.setter
    def guidance_scale(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def height(self) -> int:
        ...
    @height.setter
    def height(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_sequence_length(self) -> int:
        ...
    @max_sequence_length.setter
    def max_sequence_length(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_images_per_prompt(self) -> int:
        ...
    @num_images_per_prompt.setter
    def num_images_per_prompt(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_inference_steps(self) -> int:
        ...
    @num_inference_steps.setter
    def num_inference_steps(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def rng_seed(self) -> int:
        ...
    @rng_seed.setter
    def rng_seed(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def strength(self) -> float:
        ...
    @strength.setter
    def strength(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def width(self) -> int:
        ...
    @width.setter
    def width(self, arg0: typing.SupportsInt) -> None:
        ...
class ImageGenerationPerfMetrics:
    """
    
        Holds performance metrics for each generate call.
    
        PerfMetrics holds fields with mean and standard deviations for the following metrics:
        - Generate iteration duration, ms
        - Inference duration for unet model, ms
        - Inference duration for transformer model, ms
    
        Additional fields include:
        - Load time, ms
        - Generate total duration, ms
        - inference durations for each encoder, ms
        - inference duration of vae_encoder model, ms
        - inference duration of vae_decoder model, ms
    
        Preferable way to access values is via get functions. Getters calculate mean and std values from raw_metrics and return pairs.
        If mean and std were already calculated, getters return cached values.
    
        :param get_text_encoder_infer_duration: Returns the inference duration of every text encoder in milliseconds.
        :type get_text_encoder_infer_duration: dict[str, float]
    
        :param get_vae_encoder_infer_duration: Returns the inference duration of vae encoder in milliseconds.
        :type get_vae_encoder_infer_duration: float
    
        :param get_vae_decoder_infer_duration: Returns the inference duration of vae decoder in milliseconds.
        :type get_vae_decoder_infer_duration: float
    
        :param get_load_time: Returns the load time in milliseconds.
        :type get_load_time: float
    
        :param get_generate_duration: Returns the generate duration in milliseconds.
        :type get_generate_duration: float
    
        :param get_inference_duration: Returns the total inference durations (including encoder, unet/transformer and decoder inference) in milliseconds.
        :type get_inference_duration: float
    
        :param get_first_and_other_iter_duration: Returns the first iteration duration and the average duration of other iterations in one generation in milliseconds.
        :type get_first_and_other_iter_duration: tuple
    
        :param get_iteration_duration: Returns the mean and standard deviation of one generation iteration in milliseconds.
        :type get_iteration_duration: MeanStdPair
    
        :param get_first_and_second_unet_infer_duration: Returns the first inference duration and the average duration of other inferences in one generation in milliseconds.
        :type get_first_and_second_unet_infer_duration: tuple
    
        :param get_unet_infer_duration: Returns the mean and standard deviation of one unet inference in milliseconds.
        :type get_unet_infer_duration: MeanStdPair
    
        :param get_first_and_other_trans_infer_duration: Returns the first inference duration and the average duration of other inferences in one generation in milliseconds.
        :type get_first_and_other_trans_infer_duration: tuple
    
        :param get_transformer_infer_duration: Returns the mean and standard deviation of one transformer inference in milliseconds.
        :type get_transformer_infer_duration: MeanStdPair
    
        :param raw_metrics: A structure of RawImageGenerationPerfMetrics type that holds raw metrics.
        :type raw_metrics: RawImageGenerationPerfMetrics
    """
    def __init__(self) -> None:
        ...
    def get_first_and_other_iter_duration(self) -> tuple:
        ...
    def get_first_and_other_trans_infer_duration(self) -> tuple:
        ...
    def get_first_and_other_unet_infer_duration(self) -> tuple:
        ...
    def get_generate_duration(self) -> float:
        ...
    def get_inference_duration(self) -> float:
        ...
    def get_iteration_duration(self) -> MeanStdPair:
        ...
    def get_load_time(self) -> float:
        ...
    def get_text_encoder_infer_duration(self) -> dict[str, float]:
        ...
    def get_transformer_infer_duration(self) -> MeanStdPair:
        ...
    def get_unet_infer_duration(self) -> MeanStdPair:
        ...
    def get_vae_decoder_infer_duration(self) -> float:
        ...
    def get_vae_encoder_infer_duration(self) -> float:
        ...
    @property
    def raw_metrics(self) -> RawImageGenerationPerfMetrics:
        ...
class InpaintingPipeline:
    """
    This class is used for generation with inpainting models.
    """
    @staticmethod
    def flux(scheduler: Scheduler, clip_text_model: CLIPTextModel, t5_encoder_model: T5EncoderModel, transformer: FluxTransformer2DModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    def flux_fill(scheduler: Scheduler, clip_text_model: CLIPTextModel, t5_encoder_model: T5EncoderModel, transformer: FluxTransformer2DModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    def latent_consistency_model(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    def stable_diffusion(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, t5_encoder_model: T5EncoderModel, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @staticmethod
    def stable_diffusion_xl(scheduler: Scheduler, clip_text_model: CLIPTextModel, clip_text_model_with_projection: CLIPTextModelWithProjection, unet: UNet2DConditionModel, vae: AutoencoderKL) -> InpaintingPipeline:
        ...
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes) -> None:
        """
                    InpaintingPipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    InpaintingPipeline class constructor.
                    models_path (os.PathLike): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: InpaintingPipeline properties
        """
    @typing.overload
    def __init__(self, pipe: Image2ImagePipeline) -> None:
        ...
    @typing.overload
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    @typing.overload
    def compile(self, text_encode_device: str, denoise_device: str, vae_device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        text_encode_device (str): Device to run the text encoder(s) on (e.g., CPU, GPU).
                        denoise_device (str): Device to run denoise steps on.
                        vae_device (str): Device to run vae encoder / decoder on.
                        kwargs: Device properties.
        """
    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def generate(self, prompt: str, image: openvino._pyopenvino.Tensor, mask_image: openvino._pyopenvino.Tensor, **kwargs) -> openvino._pyopenvino.Tensor:
        """
            Generates images for text-to-image models.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.
        
            Expected parameters list:
            prompt_2: str - second prompt,
            prompt_3: str - third prompt,
            negative_prompt: str - negative prompt,
            negative_prompt_2: str - second negative prompt,
            negative_prompt_3: str - third negative prompt,
            num_images_per_prompt: int - number of images, that should be generated per prompt,
            guidance_scale: float - guidance scale,
            generation_config: GenerationConfig,
            height: int - height of resulting images,
            width: int - width of resulting images,
            num_inference_steps: int - number of inference steps,
            rng_seed: int - a seed for random numbers generator,
            generator: openvino_genai.TorchGenerator, openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator,
            adapters: LoRA adapters,
            strength: strength for image to image generation. 1.0f means initial image is fully noised,
            max_sequence_length: int - length of t5_encoder_model input
        
            :return: ov.Tensor with resulting images
            :rtype: ov.Tensor
        """
    def get_generation_config(self) -> ImageGenerationConfig:
        ...
    def get_performance_metrics(self) -> ImageGenerationPerfMetrics:
        ...
    def reshape(self, num_images_per_prompt: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, guidance_scale: typing.SupportsFloat) -> None:
        ...
    def set_generation_config(self, config: ImageGenerationConfig) -> None:
        ...
    def set_scheduler(self, scheduler: Scheduler) -> None:
        ...
class KVCrushAnchorPointMode:
    """
    Represents the anchor point types for KVCrush cache eviction
                      :param KVCrushAnchorPointMode.RANDOM: Random binary vector will be used as anchor point
                      :param KVCrushAnchorPointMode.ZEROS: Vector of all zeros will be used as anchor point
                      :param KVCrushAnchorPointMode.ONES: Vector of all ones will be used as anchor point
                      :param KVCrushAnchorPointMode.MEAN: Mean of indicator feature vector to be used as anchor point
                      :param KVCrushAnchorPointMode.ALTERNATE: Alternating 0s and 1s will be used as anchor point
    
    Members:
    
      RANDOM
    
      ZEROS
    
      ONES
    
      MEAN
    
      ALTERNATE
    """
    ALTERNATE: typing.ClassVar[KVCrushAnchorPointMode]  # value = <KVCrushAnchorPointMode.ALTERNATE: 4>
    MEAN: typing.ClassVar[KVCrushAnchorPointMode]  # value = <KVCrushAnchorPointMode.MEAN: 3>
    ONES: typing.ClassVar[KVCrushAnchorPointMode]  # value = <KVCrushAnchorPointMode.ONES: 2>
    RANDOM: typing.ClassVar[KVCrushAnchorPointMode]  # value = <KVCrushAnchorPointMode.RANDOM: 0>
    ZEROS: typing.ClassVar[KVCrushAnchorPointMode]  # value = <KVCrushAnchorPointMode.ZEROS: 1>
    __members__: typing.ClassVar[dict[str, KVCrushAnchorPointMode]]  # value = {'RANDOM': <KVCrushAnchorPointMode.RANDOM: 0>, 'ZEROS': <KVCrushAnchorPointMode.ZEROS: 1>, 'ONES': <KVCrushAnchorPointMode.ONES: 2>, 'MEAN': <KVCrushAnchorPointMode.MEAN: 3>, 'ALTERNATE': <KVCrushAnchorPointMode.ALTERNATE: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KVCrushConfig:
    """
    Configuration for KVCrush cache eviction algorithm
    """
    anchor_point_mode: KVCrushAnchorPointMode
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, budget: typing.SupportsInt, anchor_point_mode: KVCrushAnchorPointMode = ..., rng_seed: typing.SupportsInt = 0) -> None:
        """
        Constructor with budget, anchor point mode, and RNG seed
        """
    def to_string(self) -> str:
        ...
    @property
    def budget(self) -> int:
        ...
    @budget.setter
    def budget(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def rng_seed(self) -> int:
        ...
    @rng_seed.setter
    def rng_seed(self, arg0: typing.SupportsInt) -> None:
        ...
class LLMPipeline:
    """
    This class is used for generation with LLMs
    """
    def __call__(self, inputs: openvino._pyopenvino.Tensor | openvino_genai.py_openvino_genai.TokenizedInputs | str | collections.abc.Sequence[str] | openvino_genai.py_openvino_genai.ChatHistory, generation_config: openvino_genai.py_openvino_genai.GenerationConfig | None = None, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> openvino_genai.py_openvino_genai.EncodedResults | openvino_genai.py_openvino_genai.DecodedResults:
        """
            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.
        
            :param inputs: inputs in the form of string, list of strings, chat history or tokenized input_ids
            :type inputs: str, list[str], ov.genai.TokenizedInputs, or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : dict
        
            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str
         
         
            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
            be used while greedy and beam search parameters will not affect decoding at all.
        
            Parameters:
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
            apply_chat_template: whether to apply chat_template for non-chat scenarios
        
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
            presence_penalty: reduces absolute log prob if the token was generated at least once.
            frequency_penalty: reduces absolute log prob as many times as the token was generated.
        
            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
                length_penalty < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
                "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
                "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
        
            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            num_return_sequences: the number of sequences to generate from a single prompt.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, tokenizer: Tokenizer, device: str, config: collections.abc.Mapping[str, typing.Any] = {}, **kwargs) -> None:
        """
                    LLMPipeline class constructor for manually created openvino_genai.Tokenizer.
                    models_path (os.PathLike): Path to the model file.
                    tokenizer (openvino_genai.Tokenizer): tokenizer object.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, config: collections.abc.Mapping[str, typing.Any] = {}, **kwargs) -> None:
        """
                    LLMPipeline class constructor.
                    models_path (os.PathLike): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: str, weights: openvino._pyopenvino.Tensor, tokenizer: Tokenizer, device: str, generation_config: openvino_genai.py_openvino_genai.GenerationConfig | None = None, **kwargs) -> None:
        """
                    LLMPipeline class constructor.
                    model (str): Pre-read model.
                    weights (ov.Tensor): Pre-read model weights.
                    tokenizer (str): Genai Tokenizers.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    generation_config {ov_genai.GenerationConfig} Genai GenerationConfig. Default is an empty config.
                    kwargs: Device properties.
        """
    def finish_chat(self) -> None:
        ...
    def generate(self, inputs: openvino._pyopenvino.Tensor | openvino_genai.py_openvino_genai.TokenizedInputs | str | collections.abc.Sequence[str] | openvino_genai.py_openvino_genai.ChatHistory, generation_config: openvino_genai.py_openvino_genai.GenerationConfig | None = None, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> openvino_genai.py_openvino_genai.EncodedResults | openvino_genai.py_openvino_genai.DecodedResults:
        """
            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.
        
            :param inputs: inputs in the form of string, list of strings, chat history or tokenized input_ids
            :type inputs: str, list[str], ov.genai.TokenizedInputs, or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : dict
        
            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str
         
         
            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
            be used while greedy and beam search parameters will not affect decoding at all.
        
            Parameters:
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
            apply_chat_template: whether to apply chat_template for non-chat scenarios
        
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
            presence_penalty: reduces absolute log prob if the token was generated at least once.
            frequency_penalty: reduces absolute log prob as many times as the token was generated.
        
            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
                length_penalty < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
                "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
                "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
        
            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            num_return_sequences: the number of sequences to generate from a single prompt.
        """
    def get_generation_config(self) -> GenerationConfig:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def set_generation_config(self, config: GenerationConfig) -> None:
        ...
    def start_chat(self, system_message: str = '') -> None:
        ...
class MeanStdPair:
    def __init__(self) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[float]:
        ...
    @property
    def mean(self) -> float:
        ...
    @property
    def std(self) -> float:
        ...
class PerfMetrics:
    """
    
        Holds performance metrics for each generate call.
    
        PerfMetrics holds the following metrics with mean and standard deviations:
        - Time To the First Token (TTFT), ms
        - Time per Output Token (TPOT), ms/token
        - Inference time per Output Token (IPOT), ms/token
        - Generate total duration, ms
        - Inference duration, ms
        - Tokenization duration, ms
        - Detokenization duration, ms
        - Throughput, tokens/s
    
        Additional metrics include:
        - Load time, ms
        - Number of generated tokens
        - Number of tokens in the input prompt
        - Time to initialize grammar compiler for each backend, ms
        - Time to compile grammar, ms
    
        Preferable way to access metrics is via getter methods. Getter methods calculate mean and std values from raw_metrics and return pairs.
        If mean and std were already calculated, getters return cached values.
    
        :param get_load_time: Returns the load time in milliseconds.
        :type get_load_time: float
    
        :param get_num_generated_tokens: Returns the number of generated tokens.
        :type get_num_generated_tokens: int
    
        :param get_num_input_tokens: Returns the number of tokens in the input prompt.
        :type get_num_input_tokens: int
    
        :param get_ttft: Returns the mean and standard deviation of TTFT in milliseconds.
        :type get_ttft: MeanStdPair
    
        :param get_tpot: Returns the mean and standard deviation of TPOT in milliseconds.
        :type get_tpot: MeanStdPair
    
        :param get_ipot: Returns the mean and standard deviation of IPOT in milliseconds.
        :type get_ipot: MeanStdPair
    
        :param get_throughput: Returns the mean and standard deviation of throughput in tokens per second.
        :type get_throughput: MeanStdPair
    
        :param get_inference_duration: Returns the mean and standard deviation of the time spent on model inference during generate call in milliseconds.
        :type get_inference_duration: MeanStdPair
    
        :param get_generate_duration: Returns the mean and standard deviation of generate durations in milliseconds.
        :type get_generate_duration: MeanStdPair
    
        :param get_tokenization_duration: Returns the mean and standard deviation of tokenization durations in milliseconds.
        :type get_tokenization_duration: MeanStdPair
    
        :param get_detokenization_duration: Returns the mean and standard deviation of detokenization durations in milliseconds.
        :type get_detokenization_duration: MeanStdPair
    
        :param get_grammar_compiler_init_times: Returns a map with the time to initialize the grammar compiler for each backend in milliseconds.
        :type get_grammar_compiler_init_times: dict[str, float]
    
        :param get_grammar_compile_time: Returns the mean, standard deviation, min, and max of grammar compile times in milliseconds.
        :type get_grammar_compile_time: SummaryStats
    
        :param raw_metrics: A structure of RawPerfMetrics type that holds raw metrics.
        :type raw_metrics: RawPerfMetrics
    """
    def __add__(self, metrics: PerfMetrics) -> PerfMetrics:
        ...
    def __iadd__(self, right: PerfMetrics) -> PerfMetrics:
        ...
    def __init__(self) -> None:
        ...
    def get_detokenization_duration(self) -> MeanStdPair:
        ...
    def get_generate_duration(self) -> MeanStdPair:
        ...
    def get_grammar_compile_time(self) -> SummaryStats:
        ...
    def get_grammar_compiler_init_times(self) -> dict[str, float]:
        ...
    def get_inference_duration(self) -> MeanStdPair:
        ...
    def get_ipot(self) -> MeanStdPair:
        ...
    def get_load_time(self) -> float:
        ...
    def get_num_generated_tokens(self) -> int:
        ...
    def get_num_input_tokens(self) -> int:
        ...
    def get_throughput(self) -> MeanStdPair:
        ...
    def get_tokenization_duration(self) -> MeanStdPair:
        ...
    def get_tpot(self) -> MeanStdPair:
        ...
    def get_ttft(self) -> MeanStdPair:
        ...
    @property
    def raw_metrics(self) -> RawPerfMetrics:
        ...
class PipelineMetrics:
    """
    
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
    """
    def __init__(self) -> None:
        ...
    @property
    def avg_cache_usage(self) -> float:
        ...
    @property
    def cache_usage(self) -> float:
        ...
    @property
    def max_cache_usage(self) -> float:
        ...
    @property
    def requests(self) -> int:
        ...
    @property
    def scheduled_requests(self) -> int:
        ...
class RawImageGenerationPerfMetrics:
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param unet_inference_durations: Durations for each unet inference in microseconds.
        :type unet_inference_durations: list[float]
    
        :param transformer_inference_durations: Durations for each transformer inference in microseconds.
        :type transformer_inference_durations: list[float]
    
        :param iteration_durations: Durations for each step iteration in microseconds.
        :type iteration_durations: list[float]
    """
    def __init__(self) -> None:
        ...
    @property
    def iteration_durations(self) -> list[float]:
        ...
    @property
    def transformer_inference_durations(self) -> list[float]:
        ...
    @property
    def unet_inference_durations(self) -> list[float]:
        ...
class RawPerfMetrics:
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param generate_durations: Durations for each generate call in milliseconds.
        :type generate_durations: list[float]
    
        :param tokenization_durations: Durations for the tokenization process in milliseconds.
        :type tokenization_durations: list[float]
    
        :param detokenization_durations: Durations for the detokenization process in milliseconds.
        :type detokenization_durations: list[float]
    
        :param m_times_to_first_token: Times to the first token for each call in milliseconds.
        :type m_times_to_first_token: list[float]
    
        :param m_new_token_times: Timestamps of generation every token or batch of tokens in milliseconds.
        :type m_new_token_times: list[double]
    
        :param token_infer_durations : Inference time for each token in milliseconds.
        :type batch_sizes: list[float]
    
        :param m_batch_sizes: Batch sizes for each generate call.
        :type m_batch_sizes: list[int]
    
        :param m_durations: Total durations for each generate call in milliseconds.
        :type m_durations: list[float]
    
        :param inference_durations : Total inference duration for each generate call in milliseconds.
        :type batch_sizes: list[float]
    
        :param grammar_compile_times: Time to compile the grammar in milliseconds.
        :type grammar_compile_times: list[float]
    """
    def __init__(self) -> None:
        ...
    @property
    def detokenization_durations(self) -> list[float]:
        ...
    @property
    def generate_durations(self) -> list[float]:
        ...
    @property
    def grammar_compile_times(self) -> list[float]:
        ...
    @property
    def inference_durations(self) -> list[float]:
        ...
    @property
    def m_batch_sizes(self) -> list[int]:
        ...
    @property
    def m_durations(self) -> list[float]:
        ...
    @property
    def m_new_token_times(self) -> list[float]:
        ...
    @property
    def m_times_to_first_token(self) -> list[float]:
        ...
    @property
    def token_infer_durations(self) -> list[float]:
        ...
    @property
    def tokenization_durations(self) -> list[float]:
        ...
class SD3Transformer2DModel:
    """
    SD3Transformer2DModel class.
    """
    class Config:
        """
        This class is used for storing SD3Transformer2DModel config.
        """
        def __init__(self, config_path: os.PathLike | str | bytes) -> None:
            ...
        @property
        def in_channels(self) -> int:
            ...
        @in_channels.setter
        def in_channels(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def joint_attention_dim(self) -> int:
            ...
        @joint_attention_dim.setter
        def joint_attention_dim(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def patch_size(self) -> int:
            ...
        @patch_size.setter
        def patch_size(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def sample_size(self) -> int:
            ...
        @sample_size.setter
        def sample_size(self, arg0: typing.SupportsInt) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    SD3Transformer2DModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    SD3Transformer2DModel class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: SD3Transformer2DModel) -> None:
        """
        SD3Transformer2DModel model
                    SD3Transformer2DModel class
                    model (SD3Transformer2DModel): SD3Transformer2DModel model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def get_config(self) -> SD3Transformer2DModel.Config:
        ...
    def infer(self, latent: openvino._pyopenvino.Tensor, timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, tokenizer_model_max_length: typing.SupportsInt) -> SD3Transformer2DModel:
        ...
    def set_hidden_states(self, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None:
        ...
class SDPerModelsPerfMetrics(SDPerfMetrics):
    """
    
        Holds performance metrics for each generate call.
    
        :param main_model_metrics: performance metrics for main model
        :type main_model_metrics: SDPerfMetrics
    
        :param draft_model_metrics: performance metrics for draft model
        :type draft_model_metrics: SDPerfMetrics
    
        :param get_num_accepted_tokens: total number of tokens, which was generated by draft model and accepted by main model
        :type get_num_accepted_tokens: int
    """
    def get_num_accepted_tokens(self) -> int:
        ...
    @property
    def draft_model_metrics(self) -> SDPerfMetrics:
        ...
    @property
    def main_model_metrics(self) -> SDPerfMetrics:
        ...
class SDPerfMetrics(ExtendedPerfMetrics):
    """
    
        Holds performance metrics for draft and main models of SpeculativeDecoding Pipeline.
    
        SDPerfMetrics holds fields with mean and standard deviations for the all PerfMetrics fields and following metrics:
        - Time to the Second Token (TTFT), ms
        - avg_latency, ms/inference
    
        Preferable way to access values is via get functions. Getters calculate mean and std values from raw_metrics and return pairs.
        If mean and std were already calculated, getters return cached values.
    
        :param get_ttst: Returns the mean and standard deviation of TTST in milliseconds.
                         The second token is presented separately as for some plugins this can be expected to take longer than next tokens.
                         In case of GPU plugin: Async compilation of some opt kernels can be completed after second token.
                         Also, additional memory manipulation can happen at second token time.
        :type get_ttst: MeanStdPair
    
        :param get_latency: Returns the mean and standard deviation of the latency from the third token in milliseconds per inference,
                            which includes also prev and post processing. First and second token time is presented separately as ttft and ttst.
        :type get_latency: MeanStdPair
    
        Additional points:
          - TPOT is calculated from the third token. The reasons for this, please, see in the description for avg_latency.
          - `total number of iterations` of the model can be taken from raw performance metrics raw_metrics.m_durations.size().
    """
    def get_latency(self) -> MeanStdPair:
        ...
    def get_ttst(self) -> MeanStdPair:
        ...
class Scheduler:
    """
    Scheduler for image generation pipelines.
    """
    class Type:
        """
        Members:
        
          AUTO
        
          LCM
        
          DDIM
        
          EULER_DISCRETE
        
          FLOW_MATCH_EULER_DISCRETE
        
          PNDM
        
          EULER_ANCESTRAL_DISCRETE
        
          LMS_DISCRETE
        """
        AUTO: typing.ClassVar[Scheduler.Type]  # value = <Type.AUTO: 0>
        DDIM: typing.ClassVar[Scheduler.Type]  # value = <Type.DDIM: 2>
        EULER_ANCESTRAL_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.EULER_ANCESTRAL_DISCRETE: 6>
        EULER_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.EULER_DISCRETE: 3>
        FLOW_MATCH_EULER_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.FLOW_MATCH_EULER_DISCRETE: 4>
        LCM: typing.ClassVar[Scheduler.Type]  # value = <Type.LCM: 1>
        LMS_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.DDIM: 2>
        PNDM: typing.ClassVar[Scheduler.Type]  # value = <Type.PNDM: 5>
        __members__: typing.ClassVar[dict[str, Scheduler.Type]]  # value = {'AUTO': <Type.AUTO: 0>, 'LCM': <Type.LCM: 1>, 'DDIM': <Type.DDIM: 2>, 'EULER_DISCRETE': <Type.EULER_DISCRETE: 3>, 'FLOW_MATCH_EULER_DISCRETE': <Type.FLOW_MATCH_EULER_DISCRETE: 4>, 'PNDM': <Type.PNDM: 5>, 'EULER_ANCESTRAL_DISCRETE': <Type.EULER_ANCESTRAL_DISCRETE: 6>, 'LMS_DISCRETE': <Type.DDIM: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    @staticmethod
    def from_config(scheduler_config_path: os.PathLike | str | bytes, scheduler_type: Scheduler.Type = ...) -> Scheduler:
        ...
class SchedulerConfig:
    """
    
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
    """
    cache_eviction_config: CacheEvictionConfig
    dynamic_split_fuse: bool
    enable_prefix_caching: bool
    sparse_attention_config: SparseAttentionConfig
    use_cache_eviction: bool
    use_sparse_attention: bool
    def __init__(self) -> None:
        ...
    def to_string(self) -> str:
        ...
    @property
    def cache_size(self) -> int:
        ...
    @cache_size.setter
    def cache_size(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_num_batched_tokens(self) -> int:
        ...
    @max_num_batched_tokens.setter
    def max_num_batched_tokens(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def max_num_seqs(self) -> int:
        ...
    @max_num_seqs.setter
    def max_num_seqs(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_kv_blocks(self) -> int:
        ...
    @num_kv_blocks.setter
    def num_kv_blocks(self, arg0: typing.SupportsInt) -> None:
        ...
class SparseAttentionConfig:
    """
    
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
    """
    mode: SparseAttentionMode
    def __init__(self, mode: SparseAttentionMode = ..., num_last_dense_tokens_in_prefill: typing.SupportsInt = 100, num_retained_start_tokens_in_cache: typing.SupportsInt = 128, num_retained_recent_tokens_in_cache: typing.SupportsInt = 1920, xattention_threshold: typing.SupportsFloat = 0.8, xattention_block_size: typing.SupportsInt = 64, xattention_stride: typing.SupportsInt = 8) -> None:
        ...
    def to_string(self) -> str:
        ...
    @property
    def num_last_dense_tokens_in_prefill(self) -> int:
        ...
    @num_last_dense_tokens_in_prefill.setter
    def num_last_dense_tokens_in_prefill(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_retained_recent_tokens_in_cache(self) -> int:
        ...
    @num_retained_recent_tokens_in_cache.setter
    def num_retained_recent_tokens_in_cache(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_retained_start_tokens_in_cache(self) -> int:
        ...
    @num_retained_start_tokens_in_cache.setter
    def num_retained_start_tokens_in_cache(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def xattention_block_size(self) -> int:
        ...
    @xattention_block_size.setter
    def xattention_block_size(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def xattention_stride(self) -> int:
        ...
    @xattention_stride.setter
    def xattention_stride(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def xattention_threshold(self) -> float:
        ...
    @xattention_threshold.setter
    def xattention_threshold(self, arg0: typing.SupportsFloat) -> None:
        ...
class SparseAttentionMode:
    """
    Represents the mode of sparse attention applied during generation.
                                   :param SparseAttentionMode.TRISHAPE: Sparse attention will be applied to prefill stage only, with a configurable number of start and recent cache tokens to be retained. A number of prefill tokens in the end of the prompt can be configured to have dense attention applied to them instead, to retain generation accuracy.
                                   :param SparseAttentionMode.XATTENTION: Following https://arxiv.org/pdf/2503.16428, introduces importance score threshold-based block sparsity into the prefill stage.  Computing importance scores introduces an overhead, but the total inference time is expected to be reduced even more.
    
    
    Members:
    
      TRISHAPE
    
      XATTENTION
    """
    TRISHAPE: typing.ClassVar[SparseAttentionMode]  # value = <SparseAttentionMode.TRISHAPE: 0>
    XATTENTION: typing.ClassVar[SparseAttentionMode]  # value = <SparseAttentionMode.XATTENTION: 1>
    __members__: typing.ClassVar[dict[str, SparseAttentionMode]]  # value = {'TRISHAPE': <SparseAttentionMode.TRISHAPE: 0>, 'XATTENTION': <SparseAttentionMode.XATTENTION: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SpeechGenerationConfig(GenerationConfig):
    """
    
        SpeechGenerationConfig
        
        Speech-generation specific parameters:
        :param minlenratio: minimum ratio of output length to input text length; prevents output that's too short.
        :type minlenratio: float
    
        :param maxlenratio: maximum ratio of output length to input text length; prevents excessively long outputs.
        :type minlenratio: float
    
        :param threshold: probability threshold for stopping decoding; when output probability exceeds above this, generation will stop.
        :type threshold: float
    """
    @typing.overload
    def __init__(self, json_path: os.PathLike | str | bytes) -> None:
        """
        path where generation_config.json is stored
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def update_generation_config(self, **kwargs) -> None:
        ...
    @property
    def maxlenratio(self) -> float:
        ...
    @maxlenratio.setter
    def maxlenratio(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def minlenratio(self) -> float:
        ...
    @minlenratio.setter
    def minlenratio(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def threshold(self) -> float:
        ...
    @threshold.setter
    def threshold(self, arg0: typing.SupportsFloat) -> None:
        ...
class SpeechGenerationPerfMetrics(PerfMetrics):
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param num_generated_samples: Returns a number of generated samples in output
        :type num_generated_samples: int
    """
    def __init__(self) -> None:
        ...
    @property
    def generate_duration(self) -> MeanStdPair:
        ...
    @property
    def m_evaluated(self) -> bool:
        ...
    @property
    def num_generated_samples(self) -> int:
        ...
    @property
    def throughput(self) -> MeanStdPair:
        ...
class StopCriteria:
    """
    
        StopCriteria controls the stopping condition for grouped beam search.
    
        The following values are possible:
            "openvino_genai.StopCriteria.EARLY" stops as soon as there are `num_beams` complete candidates.
            "openvino_genai.StopCriteria.HEURISTIC" stops when is it unlikely to find better candidates.
            "openvino_genai.StopCriteria.NEVER" stops when there cannot be better candidates.
    
    
    Members:
    
      EARLY
    
      HEURISTIC
    
      NEVER
    """
    EARLY: typing.ClassVar[StopCriteria]  # value = <StopCriteria.EARLY: 0>
    HEURISTIC: typing.ClassVar[StopCriteria]  # value = <StopCriteria.HEURISTIC: 1>
    NEVER: typing.ClassVar[StopCriteria]  # value = <StopCriteria.NEVER: 2>
    __members__: typing.ClassVar[dict[str, StopCriteria]]  # value = {'EARLY': <StopCriteria.EARLY: 0>, 'HEURISTIC': <StopCriteria.HEURISTIC: 1>, 'NEVER': <StopCriteria.NEVER: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class StreamerBase:
    """
    
        Base class for streamers. In order to use inherit from from this class and implement write and end methods.
    """
    def __init__(self) -> None:
        ...
    def end(self) -> None:
        """
        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """
    def put(self, token: typing.SupportsInt) -> bool:
        """
        Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
    def write(self, token: typing.SupportsInt | collections.abc.Sequence[typing.SupportsInt]) -> StreamingStatus:
        """
        Write is called every time new token or vector of tokens is decoded. Returns a StreamingStatus flag to indicate whether generation should be stopped or cancelled
        """
class StreamingStatus:
    """
    Members:
    
      RUNNING
    
      CANCEL
    
      STOP
    """
    CANCEL: typing.ClassVar[StreamingStatus]  # value = <StreamingStatus.CANCEL: 2>
    RUNNING: typing.ClassVar[StreamingStatus]  # value = <StreamingStatus.RUNNING: 0>
    STOP: typing.ClassVar[StreamingStatus]  # value = <StreamingStatus.STOP: 1>
    __members__: typing.ClassVar[dict[str, StreamingStatus]]  # value = {'RUNNING': <StreamingStatus.RUNNING: 0>, 'CANCEL': <StreamingStatus.CANCEL: 2>, 'STOP': <StreamingStatus.STOP: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class StructuralTagItem:
    """
    
        Structure to keep generation config parameters for structural tags in structured output generation.
        It is used to store the configuration for a single structural tag item, which includes the begin string,
        schema, and end string.
    
        Parameters:
        begin:  the string that marks the beginning of the structural tag.
        schema: the JSON schema that defines the structure of the tag.
        end:    the string that marks the end of the structural tag.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def begin(self) -> str:
        """
        Begin string for Structural Tag Item
        """
    @begin.setter
    def begin(self, arg0: str) -> None:
        ...
    @property
    def end(self) -> str:
        """
        End string for Structural Tag Item
        """
    @end.setter
    def end(self, arg0: str) -> None:
        ...
    @property
    def schema(self) -> str:
        """
        Json schema for Structural Tag Item
        """
    @schema.setter
    def schema(self, arg0: str) -> None:
        ...
class StructuralTagsConfig:
    """
    
        Configures structured output generation by combining regular sampling with structural tags.
    
        When the model generates a trigger string, it switches to structured output mode and produces output
        based on the defined structural tags. Afterward, regular sampling resumes.
    
        Example:
          - Trigger "<func=" activates tags with begin "<func=sum>" or "<func=multiply>".
    
        Note:
          - Simple triggers like "<" may activate structured output unexpectedly if present in regular text.
          - Very specific or long triggers may be difficult for the model to generate,
          so structured output may not be triggered.
    
        Parameters:
        structural_tags: List of StructuralTagItem objects defining structural tags.
        triggers:        List of strings that trigger structured output generation.
                         Triggers may match the beginning or part of a tag's begin string.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def structural_tags(self) -> list[StructuralTagItem]:
        """
        List of structural tag items for structured output generation
        """
    @structural_tags.setter
    def structural_tags(self, arg0: collections.abc.Sequence[StructuralTagItem]) -> None:
        ...
    @property
    def triggers(self) -> list[str]:
        """
        List of strings that will trigger generation of structured output
        """
    @triggers.setter
    def triggers(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
class StructuredOutputConfig:
    """
    
        Structure to keep generation config parameters for structured output generation.
        It is used to store the configuration for structured generation, which includes
        the JSON schema and other related parameters.
    
        Structured output parameters:
        json_schema:           if set, the output will be a JSON string constraint by the specified json-schema.
        regex:          if set, the output will be constraint by specified regex.
        grammar:        if set, the output will be constraint by specified EBNF grammar.
        structural_tags_config: if set, the output will be constraint by specified structural tags configuration.
        compound_grammar:
            if set, the output will be constraint by specified compound grammar.
            Compound grammar is a combination of multiple grammars that can be used to generate structured outputs.
            It allows for more complex and flexible structured output generation.
            The compound grammar a Union or Concat of several grammars, where each grammar can be a JSON schema, regex, EBNF, Union or Concat.
    """
    class AnyText:
        """
        
            AnyText structural tag allows any text for the portion of output
            covered by this tag.
        """
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class Concat:
        """
        
            Concat composes multiple structural tags in sequence. Each element
            must be produced in the given order. Can be used indirectly with + operator.
        
            Example: Concat(ConstString("a"), ConstString("b")) produces "ab".
            ConstString("a") + ConstString("b") is equivalent.
        """
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        @typing.overload
        def __init__(self, elements: collections.abc.Iterable) -> None:
            ...
        @typing.overload
        def __init__(self, *args) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
        @property
        def elements(self) -> list[str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator]:
            ...
        @elements.setter
        def elements(self, arg0: collections.abc.Sequence[str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator]) -> None:
            ...
    class ConstString:
        """
        
            ConstString structural tag forces the generator to produce exactly
            the provided constant string value.
        """
        value: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, arg0: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class EBNF:
        """
        
            EBNF structural tag constrains output using an EBNF grammar.
        """
        value: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, arg0: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class JSONSchema:
        """
        
            JSONSchema structural tag constrains output to a JSON document that
            must conform to the provided JSON Schema string.
        """
        value: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, arg0: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class QwenXMLParametersFormat:
        """
        
            QwenXMLParametersFormat instructs the generator to output an XML
            parameters block derived from the provided JSON schema. This is a
            specialized helper for Qwen-style XML parameter formatting.
        """
        json_schema: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, arg0: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class Regex:
        """
        
            Regex structural tag constrains output using a regular expression.
        """
        value: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, arg0: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class Tag:
        """
        
            Tag defines a begin/end wrapper with constrained inner content.
        
            The generator will output `begin`, then the `content` (a StructuralTag),
            and finally `end`.
        
            Example: Tag("<think>", AnyText(), "</think>") represents thinking portion of the model output.
        """
        begin: str
        content: str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator
        end: str
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, begin: str, content: str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator, end: str) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
    class TagsWithSeparator:
        """
        
            TagsWithSeparator configures generation of a sequence of tags
            separated by a fixed separator string.
        
            Can be used to produce repeated tagged elements like "<f>A</f>;<f>B</f>"
            where `separator`=";".
        """
        at_least_one: bool
        separator: str
        stop_after_first: bool
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, tags: collections.abc.Sequence[StructuredOutputConfig.Tag], separator: str, at_least_one: bool = False, stop_after_first: bool = False) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
        @property
        def tags(self) -> list[StructuredOutputConfig.Tag]:
            ...
        @tags.setter
        def tags(self, arg0: collections.abc.Sequence[StructuredOutputConfig.Tag]) -> None:
            ...
    class TriggeredTags:
        """
        
            TriggeredTags associates a set of `triggers` with multiple `tags`.
        
            When the model generates any of the trigger strings the structured
            generation activates to produce configured tags. Flags allow requiring
            at least one tag and stopping structured generation after the first tag.
        """
        at_least_one: bool
        stop_after_first: bool
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        def __init__(self, triggers: collections.abc.Sequence[str], tags: collections.abc.Sequence[StructuredOutputConfig.Tag], at_least_one: bool = False, stop_after_first: bool = False) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
        @property
        def tags(self) -> list[StructuredOutputConfig.Tag]:
            ...
        @tags.setter
        def tags(self, arg0: collections.abc.Sequence[StructuredOutputConfig.Tag]) -> None:
            ...
        @property
        def triggers(self) -> list[str]:
            ...
        @triggers.setter
        def triggers(self, arg0: collections.abc.Sequence[str]) -> None:
            ...
    class Union:
        """
        
            Union composes multiple structural tags as alternatives. The
            model may produce any one of the provided elements. Can be used indirectly
            with | operator.
        """
        def __add__(self, arg0: typing.Any) -> StructuredOutputConfig.Concat:
            ...
        @typing.overload
        def __init__(self, elements: collections.abc.Iterable) -> None:
            ...
        @typing.overload
        def __init__(self, *args) -> None:
            ...
        def __or__(self, arg0: typing.Any) -> StructuredOutputConfig.Union:
            ...
        def __repr__(self) -> str:
            ...
        @property
        def elements(self) -> list[str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator]:
            ...
        @elements.setter
        def elements(self, arg0: collections.abc.Sequence[str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator]) -> None:
            ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def compound_grammar(self) -> str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator | None:
        """
        Compound grammar for structured output generation
        """
    @compound_grammar.setter
    def compound_grammar(self, arg0: str | openvino_genai.py_openvino_genai.StructuredOutputConfig.Regex | openvino_genai.py_openvino_genai.StructuredOutputConfig.JSONSchema | openvino_genai.py_openvino_genai.StructuredOutputConfig.EBNF | openvino_genai.py_openvino_genai.StructuredOutputConfig.ConstString | openvino_genai.py_openvino_genai.StructuredOutputConfig.AnyText | openvino_genai.py_openvino_genai.StructuredOutputConfig.QwenXMLParametersFormat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Concat | openvino_genai.py_openvino_genai.StructuredOutputConfig.Union | openvino_genai.py_openvino_genai.StructuredOutputConfig.Tag | openvino_genai.py_openvino_genai.StructuredOutputConfig.TriggeredTags | openvino_genai.py_openvino_genai.StructuredOutputConfig.TagsWithSeparator | None) -> None:
        ...
    @property
    def grammar(self) -> str | None:
        """
        Grammar for structured output generation
        """
    @grammar.setter
    def grammar(self, arg0: str | None) -> None:
        ...
    @property
    def json_schema(self) -> str | None:
        """
        JSON schema for structured output generation
        """
    @json_schema.setter
    def json_schema(self, arg0: str | None) -> None:
        ...
    @property
    def regex(self) -> str | None:
        """
        Regular expression for structured output generation
        """
    @regex.setter
    def regex(self, arg0: str | None) -> None:
        ...
    @property
    def structural_tags_config(self) -> typing.Any:
        """
        Configuration for structural tags in structured output generation (can be StructuralTagsConfig or StructuralTag)
        """
    @structural_tags_config.setter
    def structural_tags_config(self, arg1: typing.Any) -> None:
        ...
class SummaryStats:
    def __init__(self) -> None:
        ...
    def as_tuple(self) -> tuple:
        ...
    @property
    def max(self) -> float:
        ...
    @property
    def mean(self) -> float:
        ...
    @property
    def min(self) -> float:
        ...
    @property
    def std(self) -> float:
        ...
class T5EncoderModel:
    """
    T5EncoderModel class.
    """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    T5EncoderModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    T5EncoderModel class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: T5EncoderModel) -> None:
        """
        T5EncoderModel model
                    T5EncoderModel class
                    model (T5EncoderModel): T5EncoderModel model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def get_output_tensor(self, idx: typing.SupportsInt) -> openvino._pyopenvino.Tensor:
        ...
    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool, max_sequence_length: typing.SupportsInt) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: typing.SupportsInt, max_sequence_length: typing.SupportsInt) -> T5EncoderModel:
        ...
class Text2ImagePipeline:
    """
    This class is used for generation with text-to-image models.
    """
    @staticmethod
    def flux(scheduler: Scheduler, clip_text_model: CLIPTextModel, t5_encoder_model: T5EncoderModel, transformer: FluxTransformer2DModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    def latent_consistency_model(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, t5_encoder_model: T5EncoderModel, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    @typing.overload
    def stable_diffusion_3(scheduler: Scheduler, clip_text_model_1: CLIPTextModelWithProjection, clip_text_model_2: CLIPTextModelWithProjection, transformer: SD3Transformer2DModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion_xl(scheduler: Scheduler, clip_text_model: CLIPTextModel, clip_text_model_with_projection: CLIPTextModelWithProjection, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes) -> None:
        """
                    Text2ImagePipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    Text2ImagePipeline class constructor.
                    models_path (os.PathLike): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: Text2ImagePipeline properties
        """
    @typing.overload
    def __init__(self, pipe: Image2ImagePipeline) -> None:
        ...
    @typing.overload
    def __init__(self, pipe: InpaintingPipeline) -> None:
        ...
    @typing.overload
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    @typing.overload
    def compile(self, text_encode_device: str, denoise_device: str, vae_device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        text_encode_device (str): Device to run the text encoder(s) on (e.g., CPU, GPU).
                        denoise_device (str): Device to run denoise steps on.
                        vae_device (str): Device to run vae decoder on.
                        kwargs: Device properties.
        """
    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def export_model(self, export_path: os.PathLike | str | bytes) -> None:
        """
                        Exports compiled models to a specified directory. Can significantly reduce model load time, especially for large models.
                        export_path (os.PathLike): A path to a directory to export compiled models to.
        
                        Use `blob_path` property to load previously exported models.
        """
    def generate(self, prompt: str, **kwargs) -> openvino._pyopenvino.Tensor:
        """
            Generates images for text-to-image models.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.
        
            Expected parameters list:
            prompt_2: str - second prompt,
            prompt_3: str - third prompt,
            negative_prompt: str - negative prompt,
            negative_prompt_2: str - second negative prompt,
            negative_prompt_3: str - third negative prompt,
            num_images_per_prompt: int - number of images, that should be generated per prompt,
            guidance_scale: float - guidance scale,
            generation_config: GenerationConfig,
            height: int - height of resulting images,
            width: int - width of resulting images,
            num_inference_steps: int - number of inference steps,
            rng_seed: int - a seed for random numbers generator,
            generator: openvino_genai.TorchGenerator, openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator,
            adapters: LoRA adapters,
            strength: strength for image to image generation. 1.0f means initial image is fully noised,
            max_sequence_length: int - length of t5_encoder_model input
        
            :return: ov.Tensor with resulting images
            :rtype: ov.Tensor
        """
    def get_generation_config(self) -> ImageGenerationConfig:
        ...
    def get_performance_metrics(self) -> ImageGenerationPerfMetrics:
        ...
    def reshape(self, num_images_per_prompt: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, guidance_scale: typing.SupportsFloat) -> None:
        ...
    def set_generation_config(self, config: ImageGenerationConfig) -> None:
        ...
    def set_scheduler(self, scheduler: Scheduler) -> None:
        ...
class Text2SpeechDecodedResults:
    """
    
        Structure that stores the result from the generate method, including a list of waveform tensors
        sampled at 16 kHz, along with performance metrics
    
        :param speeches: a list of waveform tensors sampled at 16 kHz
        :type speeches: list
    
        :param perf_metrics: performance metrics
        :type perf_metrics: SpeechGenerationPerfMetrics
    """
    def __init__(self) -> None:
        ...
    @property
    def perf_metrics(self) -> SpeechGenerationPerfMetrics:
        ...
    @property
    def speeches(self) -> list[openvino._pyopenvino.Tensor]:
        ...
class Text2SpeechPipeline:
    """
    Text-to-speech pipeline
    """
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    Text2SpeechPipeline class constructor.
                    models_path (os.PathLike): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU).
        """
    @typing.overload
    def generate(self, text: str, speaker_embedding: typing.Any = None, **kwargs) -> Text2SpeechDecodedResults:
        """
            Generates speeches based on input texts
        
            :param text(s): input text(s) for which to generate speech
            :type text(s): str or list[str]
        
            :param speaker_embedding optional speaker embedding tensor representing the unique characteristics of a speaker's
                                     voice. If not provided for SpeechT5 TSS model, the 7306-th vector from the validation set of the
                                     `Matthijs/cmu-arctic-xvectors` dataset is used by default.
            :type speaker_embedding: openvino.Tensor or None
        
            :param properties: speech generation parameters specified as properties
            :type properties: dict
        
            :returns: raw audios of the input texts spoken in the specified speaker's voice, with a sample rate of 16 kHz
            :rtype: Text2SpeechDecodedResults
         
         
            SpeechGenerationConfig
            
            Speech-generation specific parameters:
            :param minlenratio: minimum ratio of output length to input text length; prevents output that's too short.
            :type minlenratio: float
        
            :param maxlenratio: maximum ratio of output length to input text length; prevents excessively long outputs.
            :type minlenratio: float
        
            :param threshold: probability threshold for stopping decoding; when output probability exceeds above this, generation will stop.
            :type threshold: float
        """
    @typing.overload
    def generate(self, texts: collections.abc.Sequence[str], speaker_embedding: typing.Any = None, **kwargs) -> Text2SpeechDecodedResults:
        """
            Generates speeches based on input texts
        
            :param text(s): input text(s) for which to generate speech
            :type text(s): str or list[str]
        
            :param speaker_embedding optional speaker embedding tensor representing the unique characteristics of a speaker's
                                     voice. If not provided for SpeechT5 TSS model, the 7306-th vector from the validation set of the
                                     `Matthijs/cmu-arctic-xvectors` dataset is used by default.
            :type speaker_embedding: openvino.Tensor or None
        
            :param properties: speech generation parameters specified as properties
            :type properties: dict
        
            :returns: raw audios of the input texts spoken in the specified speaker's voice, with a sample rate of 16 kHz
            :rtype: Text2SpeechDecodedResults
         
         
            SpeechGenerationConfig
            
            Speech-generation specific parameters:
            :param minlenratio: minimum ratio of output length to input text length; prevents output that's too short.
            :type minlenratio: float
        
            :param maxlenratio: maximum ratio of output length to input text length; prevents excessively long outputs.
            :type minlenratio: float
        
            :param threshold: probability threshold for stopping decoding; when output probability exceeds above this, generation will stop.
            :type threshold: float
        """
    def get_generation_config(self) -> SpeechGenerationConfig:
        ...
    def set_generation_config(self, config: SpeechGenerationConfig) -> None:
        ...
class TextEmbeddingPipeline:
    """
    Text embedding pipeline
    """
    class Config:
        """
        
        Structure to keep TextEmbeddingPipeline configuration parameters.
        
        Attributes:
            max_length (int, optional):
                Maximum length of tokens passed to the embedding model.
            pad_to_max_length (bool, optional):
                If 'True', model input tensors are padded to the maximum length.
            batch_size (int, optional):
                Batch size for the embedding model.
                Useful for database population. If set, the pipeline will fix model shape for inference optimization.
                Number of documents passed to pipeline should be equal to batch_size.
                For query embeddings, batch_size should be set to 1 or not set.
            pooling_type (TextEmbeddingPipeline.PoolingType, optional):
                Pooling strategy applied to the model output tensor. Defaults to PoolingType.CLS.
            normalize (bool, optional):
                If True, L2 normalization is applied to embeddings. Defaults to True.
            query_instruction (str, optional):
                Instruction to use for embedding a query.
            embed_instruction (str, optional):
                Instruction to use for embedding a document.
            padding_side (str, optional):
                Side to use for padding "left" or "right"
        """
        embed_instruction: str | None
        normalize: bool
        pad_to_max_length: bool | None
        padding_side: str | None
        pooling_type: TextEmbeddingPipeline.PoolingType
        query_instruction: str | None
        @typing.overload
        def __init__(self) -> None:
            ...
        @typing.overload
        def __init__(self, **kwargs) -> None:
            ...
        def validate(self) -> None:
            """
            Checks that are no conflicting parameters. Raises exception if config is invalid.
            """
        @property
        def batch_size(self) -> int | None:
            ...
        @batch_size.setter
        def batch_size(self, arg0: typing.SupportsInt | None) -> None:
            ...
        @property
        def max_length(self) -> int | None:
            ...
        @max_length.setter
        def max_length(self, arg0: typing.SupportsInt | None) -> None:
            ...
    class PoolingType:
        """
        Members:
        
          CLS : First token embeddings
        
          MEAN : The average of all token embeddings
        
          LAST_TOKEN : Last token embeddings
        """
        CLS: typing.ClassVar[TextEmbeddingPipeline.PoolingType]  # value = <PoolingType.CLS: 0>
        LAST_TOKEN: typing.ClassVar[TextEmbeddingPipeline.PoolingType]  # value = <PoolingType.LAST_TOKEN: 2>
        MEAN: typing.ClassVar[TextEmbeddingPipeline.PoolingType]  # value = <PoolingType.MEAN: 1>
        __members__: typing.ClassVar[dict[str, TextEmbeddingPipeline.PoolingType]]  # value = {'CLS': <PoolingType.CLS: 0>, 'MEAN': <PoolingType.MEAN: 1>, 'LAST_TOKEN': <PoolingType.LAST_TOKEN: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, config: openvino_genai.py_openvino_genai.TextEmbeddingPipeline.Config | None = None, **kwargs) -> None:
        """
        Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir
        models_path (os.PathLike): Path to the directory containing model xml/bin files and tokenizer
        device (str): Device to run the model on (e.g., CPU, GPU).
        config: (TextEmbeddingPipeline.Config): Optional pipeline configuration
        kwargs: Plugin and/or config properties
        """
    def embed_documents(self, texts: collections.abc.Sequence[str]) -> list[list[float]] | list[list[int]] | list[list[int]]:
        """
        Computes embeddings for a vector of texts
        """
    def embed_query(self, text: str) -> list[float] | list[int] | list[int]:
        """
        Computes embeddings for a query
        """
    def start_embed_documents_async(self, texts: collections.abc.Sequence[str]) -> None:
        """
        Asynchronously computes embeddings for a vector of texts
        """
    def start_embed_query_async(self, text: str) -> None:
        """
        Asynchronously computes embeddings for a query
        """
    def wait_embed_documents(self) -> list[list[float]] | list[list[int]] | list[list[int]]:
        """
        Waits computed embeddings of a vector of texts
        """
    def wait_embed_query(self) -> list[float] | list[int] | list[int]:
        """
        Waits computed embeddings for a query
        """
class TextRerankPipeline:
    """
    Text rerank pipeline
    """
    class Config:
        """
        
        Structure to keep TextRerankPipeline configuration parameters.
        Attributes:
            top_n (int, optional):
                Number of documents to return sorted by score.
            max_length (int, optional):
                Maximum length of tokens passed to the embedding model.
        """
        @typing.overload
        def __init__(self) -> None:
            ...
        @typing.overload
        def __init__(self, **kwargs) -> None:
            ...
        @property
        def max_length(self) -> int | None:
            ...
        @max_length.setter
        def max_length(self, arg0: typing.SupportsInt | None) -> None:
            ...
        @property
        def top_n(self) -> int:
            ...
        @top_n.setter
        def top_n(self, arg0: typing.SupportsInt) -> None:
            ...
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, config: openvino_genai.py_openvino_genai.TextRerankPipeline.Config | None = None, **kwargs) -> None:
        """
        Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir
        models_path (os.PathLike): Path to the directory containing model xml/bin files and tokenizer
        device (str): Device to run the model on (e.g., CPU, GPU).
        config: (TextRerankPipeline.Config): Optional pipeline configuration
        kwargs: Plugin and/or config properties
        """
    def rerank(self, query: str, texts: collections.abc.Sequence[str]) -> list[tuple[int, float]]:
        """
        Reranks a vector of texts based on the query.
        """
    def start_rerank_async(self, query: str, texts: collections.abc.Sequence[str]) -> None:
        """
        Asynchronously reranks a vector of texts based on the query.
        """
    def wait_rerank(self) -> list[tuple[int, float]]:
        """
        Waits for reranked texts.
        """
class TextStreamer(StreamerBase):
    """
    
    TextStreamer is used to decode tokens into text and call a user-defined callback function.
    
    tokenizer: Tokenizer object to decode tokens into text.
    callback: User-defined callback function to process the decoded text, callback should return either boolean flag or StreamingStatus.
    detokenization_params: AnyMap with detokenization parameters, e.g. ov::genai::skip_special_tokens(...)
    """
    def __init__(self, tokenizer: Tokenizer, callback: collections.abc.Callable[[str], bool | openvino_genai.py_openvino_genai.StreamingStatus], detokenization_params: collections.abc.Mapping[str, typing.Any] = {}) -> None:
        ...
    def end(self) -> None:
        ...
    def write(self, token: typing.SupportsInt | collections.abc.Sequence[typing.SupportsInt]) -> StreamingStatus:
        ...
class TokenizedInputs:
    attention_mask: openvino._pyopenvino.Tensor
    input_ids: openvino._pyopenvino.Tensor
    def __init__(self, input_ids: openvino._pyopenvino.Tensor, attention_mask: openvino._pyopenvino.Tensor) -> None:
        ...
class Tokenizer:
    """
    
        The class is used to encode prompts and decode resulting tokens
    
        Chat template is initialized from sources in the following order
        overriding the previous value:
        1. chat_template entry from tokenizer_config.json
        2. chat_template entry from processor_config.json
        3. chat_template entry from chat_template.json
        4. chat_template entry from rt_info section of openvino.Model
        5. If the template is known to be not supported by GenAI, it's
            replaced with a simplified supported version.
    """
    chat_template: str
    @typing.overload
    def __init__(self, tokenizer_path: os.PathLike | str | bytes, properties: collections.abc.Mapping[str, typing.Any] = {}, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, tokenizer_model: str, tokenizer_weights: openvino._pyopenvino.Tensor, detokenizer_model: str, detokenizer_weights: openvino._pyopenvino.Tensor, **kwargs) -> None:
        ...
    def apply_chat_template(self, history: openvino_genai.py_openvino_genai.ChatHistory | collections.abc.Sequence[dict], add_generation_prompt: bool, chat_template: str = '', tools: collections.abc.Sequence[dict] | None = None, extra_context: dict | None = None) -> str:
        """
        Applies a chat template to format chat history into a prompt string.
        """
    @typing.overload
    def decode(self, tokens: collections.abc.Sequence[typing.SupportsInt], skip_special_tokens: bool = True) -> str:
        """
        Decode a sequence into a string prompt.
        """
    @typing.overload
    def decode(self, tokens: openvino._pyopenvino.Tensor, skip_special_tokens: bool = True) -> list[str]:
        """
        Decode tensor into a list of string prompts.
        """
    @typing.overload
    def decode(self, tokens: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], skip_special_tokens: bool = True) -> list[str]:
        """
        Decode a batch of tokens into a list of string prompt.
        """
    @typing.overload
    def encode(self, prompts: collections.abc.Sequence[str], add_special_tokens: bool = True, pad_to_max_length: bool = False, max_length: typing.SupportsInt | None = None, padding_side: str | None = None) -> TokenizedInputs:
        """
        Encodes a list of prompts into tokenized inputs.
        Args:
         'prompts' - list of prompts to encode
         'add_special_tokens' - whether to add special tokens like BOS, EOS, PAD. Default is True.
         'pad_to_max_length' - whether to pad the sequence to the maximum length. Default is False.
         'max_length' - maximum length of the sequence. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
         'padding_side' - side to pad the sequence, can be 'left' or 'right'. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
        Returns:
         TokenizedInputs object containing input_ids and attention_mask tensors.
        """
    @typing.overload
    def encode(self, prompt: str, add_special_tokens: bool = True, pad_to_max_length: bool = False, max_length: typing.SupportsInt | None = None, padding_side: str | None = None) -> TokenizedInputs:
        """
        Encodes a single prompt into tokenized input.
        Args:
         'prompt' - prompt to encode
         'add_special_tokens' - whether to add special tokens like BOS, EOS, PAD. Default is True.
         'pad_to_max_length' - whether to pad the sequence to the maximum length. Default is False.
         'max_length' - maximum length of the sequence. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
         'padding_side' - side to pad the sequence, can be 'left' or 'right'. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
        Returns:
         TokenizedInputs object containing input_ids and attention_mask tensors.
        """
    @typing.overload
    def encode(self, prompts_1: collections.abc.Sequence[str], prompts_2: collections.abc.Sequence[str], add_special_tokens: bool = True, pad_to_max_length: bool = False, max_length: typing.SupportsInt | None = None, padding_side: str | None = None) -> TokenizedInputs:
        """
        Encodes a list of prompts into tokenized inputs. The number of strings must be the same, or one of the inputs can contain one string.
        In the latter case, the single-string input will be broadcast into the shape of the other input, which is more efficient than repeating the string in pairs.)
        Args:
         'prompts_1' - list of prompts to encode
         'prompts_2' - list of prompts to encode
         'add_special_tokens' - whether to add special tokens like BOS, EOS, PAD. Default is True.
         'pad_to_max_length' - whether to pad the sequence to the maximum length. Default is False.
         'max_length' - maximum length of the sequence. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
         'padding_side' - side to pad the sequence, can be 'left' or 'right'. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
        Returns:
         TokenizedInputs object containing input_ids and attention_mask tensors.
        """
    @typing.overload
    def encode(self, prompts: list, add_special_tokens: bool = True, pad_to_max_length: bool = False, max_length: typing.SupportsInt | None = None, padding_side: str | None = None) -> TokenizedInputs:
        """
        Encodes a list of paired prompts into tokenized inputs. Input format is same as for HF paired input [[prompt_1, prompt_2], ...].
        Args:
         'prompts' - list of prompts to encode\\n
         'add_special_tokens' - whether to add special tokens like BOS, EOS, PAD. Default is True.
         'pad_to_max_length' - whether to pad the sequence to the maximum length. Default is False.
         'max_length' - maximum length of the sequence. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
         'padding_side' - side to pad the sequence, can be 'left' or 'right'. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
        Returns:
         TokenizedInputs object containing input_ids and attention_mask tensors.
        """
    def get_bos_token(self) -> str:
        ...
    def get_bos_token_id(self) -> int:
        ...
    def get_eos_token(self) -> str:
        ...
    def get_eos_token_id(self) -> int:
        ...
    def get_original_chat_template(self) -> str:
        ...
    def get_pad_token(self) -> str:
        ...
    def get_pad_token_id(self) -> int:
        ...
    def get_vocab(self) -> dict:
        """
        Returns the vocabulary as a Python dictionary with bytes keys and integer values. 
                     Bytes are used for keys because not all vocabulary entries might be valid UTF-8 strings.
        """
    def get_vocab_vector(self) -> list[str]:
        """
        Returns the vocabulary as list of strings, where position of a string represents token ID.
        """
    def set_chat_template(self, chat_template: str) -> None:
        """
        Override a chat_template read from tokenizer_config.json.
        """
    def supports_paired_input(self) -> bool:
        """
        Returns true if the tokenizer supports paired input, false otherwise.
        """
class TorchGenerator(CppStdGenerator):
    """
    This class provides OpenVINO GenAI Generator wrapper for torch.Generator
    """
    def __init__(self, seed: typing.SupportsInt) -> None:
        ...
    def next(self) -> float:
        ...
    def randn_tensor(self, shape: openvino._pyopenvino.Shape) -> openvino._pyopenvino.Tensor:
        ...
    def seed(self, new_seed: typing.SupportsInt) -> None:
        ...
class UNet2DConditionModel:
    """
    UNet2DConditionModel class.
    """
    class Config:
        """
        This class is used for storing UNet2DConditionModel config.
        """
        def __init__(self, config_path: os.PathLike | str | bytes) -> None:
            ...
        @property
        def in_channels(self) -> int:
            ...
        @in_channels.setter
        def in_channels(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def sample_size(self) -> int:
            ...
        @sample_size.setter
        def sample_size(self, arg0: typing.SupportsInt) -> None:
            ...
        @property
        def time_cond_proj_dim(self) -> int:
            ...
        @time_cond_proj_dim.setter
        def time_cond_proj_dim(self, arg0: typing.SupportsInt) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes) -> None:
        """
                    UNet2DConditionModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    UNet2DConditionModel class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: UNet2DConditionModel) -> None:
        """
        UNet2DConditionModel model
                    UNet2DConditionModel class
                    model (UNet2DConditionModel): UNet2DConditionModel model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def do_classifier_free_guidance(self, guidance_scale: typing.SupportsFloat) -> bool:
        ...
    def export_model(self, export_path: os.PathLike | str | bytes) -> None:
        """
                        Exports compiled model to a specified directory. Can significantly reduce model load time, especially for large models.
                        export_path (os.PathLike): A path to a directory to export compiled model to.
        
                        Use `blob_path` property to load previously exported models.
        """
    def get_config(self) -> UNet2DConditionModel.Config:
        ...
    def infer(self, sample: openvino._pyopenvino.Tensor, timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: typing.SupportsInt, height: typing.SupportsInt, width: typing.SupportsInt, tokenizer_model_max_length: typing.SupportsInt) -> UNet2DConditionModel:
        ...
    def set_adapters(self, adapters: openvino_genai.py_openvino_genai.AdapterConfig | None) -> None:
        ...
    def set_hidden_states(self, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None:
        ...
class VLMDecodedResults(DecodedResults):
    """
    
        Structure to store resulting batched text outputs and scores for each batch.
        The first num_return_sequences elements correspond to the first batch element.
    
        Parameters:
        texts:      vector of resulting sequences.
        scores:     scores for each sequence.
        metrics:    performance metrics with tpot, ttft, etc. of type openvino_genai.VLMPerfMetrics.
    """
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def perf_metrics(self) -> VLMPerfMetrics:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def texts(self) -> list[str]:
        ...
class VLMPerfMetrics(PerfMetrics):
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param get_prepare_embeddings_duration: Returns mean and standard deviation of embeddings preparation duration in milliseconds
        :type get_prepare_embeddings_duration: MeanStdPair
    
        :param vlm_raw_metrics: VLM specific raw metrics
        :type VLMRawPerfMetrics:
    """
    def __init__(self) -> None:
        ...
    def get_prepare_embeddings_duration(self) -> MeanStdPair:
        ...
    @property
    def vlm_raw_metrics(self) -> VLMRawPerfMetrics:
        ...
class VLMPipeline:
    """
    This class is used for generation with VLMs
    """
    @typing.overload
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    VLMPipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    kwargs: Device properties
        """
    @typing.overload
    def __init__(self, models: collections.abc.Mapping[str, tuple[str, openvino._pyopenvino.Tensor]], tokenizer: Tokenizer, config_dir_path: os.PathLike | str | bytes, device: str, generation_config: openvino_genai.py_openvino_genai.GenerationConfig | None = None, **kwargs) -> None:
        """
                    VLMPipeline class constructor.
                    models (dict[str, tuple[str, openvino.Tensor]]): A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
                    tokenizer (Tokenizer): Genai Tokenizers.
                    config_dir_path (os.PathLike): Path to folder with model configs.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    generation_config (GenerationConfig | None): Device properties.
                    kwargs: Device properties
        """
    def finish_chat(self) -> None:
        ...
    @typing.overload
    def generate(self, prompt: str, images: collections.abc.Sequence[openvino._pyopenvino.Tensor], videos: collections.abc.Sequence[openvino._pyopenvino.Tensor], generation_config: GenerationConfig, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
            The prompt can contain <ov_genai_image_i> with i replaced with
            an actual zero based index to refer to an image. Reference to
            images used in previous prompts isn't implemented.
            A model's native image tag can be used instead of
            <ov_genai_image_i>. These tags are:
            InternVL2: <image>\\n
            llava-1.5-7b-hf: <image>
            LLaVA-NeXT: <image>
            LLaVa-NeXT-Video: <image>
            nanoLLaVA: <image>\\n
            nanoLLaVA-1.5: <image>\\n
            MiniCPM-o-2_6: <image>./</image>\\n
            MiniCPM-V-2_6: <image>./</image>\\n
            Phi-3-vision: <|image_i|>\\n - the index starts with one
            Phi-4-multimodal-instruct: <|image_i|>\\n - the index starts with one
            Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
            Qwen2.5-VL: <|vision_start|><|image_pad|><|vision_end|>
            gemma-3-4b-it: <start_of_image>
            Model's native video tag can be used to refer to a video:
            LLaVa-NeXT-Video: <video>
            If the prompt doesn't contain image or video tags, but images or videos are
            provided, the tags are prepended to the prompt.
        
            :param images: image or list of images
            :type images: list[ov.Tensor] or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : dict
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    @typing.overload
    def generate(self, prompt: str, images: collections.abc.Sequence[openvino._pyopenvino.Tensor], generation_config: GenerationConfig, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
            The prompt can contain <ov_genai_image_i> with i replaced with
            an actual zero based index to refer to an image. Reference to
            images used in previous prompts isn't implemented.
            A model's native image tag can be used instead of
            <ov_genai_image_i>. These tags are:
            InternVL2: <image>\\n
            llava-1.5-7b-hf: <image>
            LLaVA-NeXT: <image>
            LLaVa-NeXT-Video: <image>
            nanoLLaVA: <image>\\n
            nanoLLaVA-1.5: <image>\\n
            MiniCPM-o-2_6: <image>./</image>\\n
            MiniCPM-V-2_6: <image>./</image>\\n
            Phi-3-vision: <|image_i|>\\n - the index starts with one
            Phi-4-multimodal-instruct: <|image_i|>\\n - the index starts with one
            Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
            Qwen2.5-VL: <|vision_start|><|image_pad|><|vision_end|>
            gemma-3-4b-it: <start_of_image>
            Model's native video tag can be used to refer to a video:
            LLaVa-NeXT-Video: <video>
            If the prompt doesn't contain image or video tags, but images or videos are
            provided, the tags are prepended to the prompt.
        
            :param images: image or list of images
            :type images: list[ov.Tensor] or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : dict
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    @typing.overload
    def generate(self, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: GenerationConfig, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
            The prompt can contain <ov_genai_image_i> with i replaced with
            an actual zero based index to refer to an image. Reference to
            images used in previous prompts isn't implemented.
            A model's native image tag can be used instead of
            <ov_genai_image_i>. These tags are:
            InternVL2: <image>\\n
            llava-1.5-7b-hf: <image>
            LLaVA-NeXT: <image>
            LLaVa-NeXT-Video: <image>
            nanoLLaVA: <image>\\n
            nanoLLaVA-1.5: <image>\\n
            MiniCPM-o-2_6: <image>./</image>\\n
            MiniCPM-V-2_6: <image>./</image>\\n
            Phi-3-vision: <|image_i|>\\n - the index starts with one
            Phi-4-multimodal-instruct: <|image_i|>\\n - the index starts with one
            Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
            Qwen2.5-VL: <|vision_start|><|image_pad|><|vision_end|>
            gemma-3-4b-it: <start_of_image>
            Model's native video tag can be used to refer to a video:
            LLaVa-NeXT-Video: <video>
            If the prompt doesn't contain image or video tags, but images or videos are
            provided, the tags are prepended to the prompt.
        
            :param images: image or list of images
            :type images: list[ov.Tensor] or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : dict
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    @typing.overload
    def generate(self, prompt: str, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            The prompt can contain <ov_genai_image_i> with i replaced with
            an actual zero based index to refer to an image. Reference to
            images used in previous prompts isn't implemented.
            A model's native image tag can be used instead of
            <ov_genai_image_i>. These tags are:
            InternVL2: <image>\\n
            llava-1.5-7b-hf: <image>
            LLaVA-NeXT: <image>
            LLaVa-NeXT-Video: <image>
            nanoLLaVA: <image>\\n
            nanoLLaVA-1.5: <image>\\n
            MiniCPM-o-2_6: <image>./</image>\\n
            MiniCPM-V-2_6: <image>./</image>\\n
            Phi-3-vision: <|image_i|>\\n - the index starts with one
            Phi-4-multimodal-instruct: <|image_i|>\\n - the index starts with one
            Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
            Qwen2.5-VL: <|vision_start|><|image_pad|><|vision_end|>
            gemma-3-4b-it: <start_of_image>
            Model's native video tag can be used to refer to a video:
            LLaVa-NeXT-Video: <video>
            If the prompt doesn't contain image or video tags, but images or videos are
            provided, the tags are prepended to the prompt.
        
            :type prompt: str
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.
        
            Expected parameters list:
            image: ov.Tensor - input image,
            images: list[ov.Tensor] - input images,
            generation_config: GenerationConfig,
            streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    def get_generation_config(self) -> GenerationConfig:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def set_chat_template(self, chat_template: str) -> None:
        ...
    def set_generation_config(self, config: GenerationConfig) -> None:
        ...
    def start_chat(self, system_message: str = '') -> None:
        ...
class VLMRawPerfMetrics:
    """
    
        Structure with VLM specific raw performance metrics for each generation before any statistics are calculated.
    
        :param prepare_embeddings_durations: Durations of embeddings preparation.
        :type prepare_embeddings_durations: list[MicroSeconds]
    """
    def __init__(self) -> None:
        ...
    @property
    def prepare_embeddings_durations(self) -> list[float]:
        ...
class WhisperDecodedResultChunk:
    """
    
        Structure to store decoded text with corresponding timestamps
    
        :param start_ts chunk start time in seconds
        :param end_ts   chunk end time in seconds
        :param text     chunk text
    """
    def __init__(self) -> None:
        ...
    @property
    def end_ts(self) -> float:
        ...
    @property
    def start_ts(self) -> float:
        ...
    @property
    def text(self) -> str:
        ...
class WhisperDecodedResults:
    """
    
        Structure to store resulting text outputs and scores.
    
        Parameters:
        texts:      vector of resulting sequences.
        scores:     scores for each sequence.
        metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
        shunks:     optional chunks of resulting sequences with timestamps
    """
    def __str__(self) -> str:
        ...
    @property
    def chunks(self) -> list[WhisperDecodedResultChunk] | None:
        ...
    @property
    def perf_metrics(self) -> WhisperPerfMetrics:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def texts(self) -> list[str]:
        ...
class WhisperGenerationConfig(GenerationConfig):
    """
    
        WhisperGenerationConfig
        
        Whisper specific parameters:
        :param decoder_start_token_id: Corresponds to the ”<|startoftranscript|>” token.
        :type decoder_start_token_id: int
    
        :param pad_token_id: Padding token id.
        :type pad_token_id: int
    
        :param translate_token_id: Translate token id.
        :type translate_token_id: int
    
        :param transcribe_token_id: Transcribe token id.
        :type transcribe_token_id: int
    
        :param no_timestamps_token_id: No timestamps token id.
        :type no_timestamps_token_id: int
    
        :param prev_sot_token_id: Corresponds to the ”<|startofprev|>” token.
        :type prev_sot_token_id: int
    
        :param is_multilingual:
        :type is_multilingual: bool
    
        :param begin_suppress_tokens: A list containing tokens that will be suppressed at the beginning of the sampling process.
        :type begin_suppress_tokens: list[int]
    
        :param suppress_tokens: A list containing the non-speech tokens that will be suppressed during generation.
        :type suppress_tokens: list[int]
    
        :param language: Language token to use for generation in the form of <|en|>.
                         You can find all the possible language tokens in the generation_config.json lang_to_id dictionary.
        :type language: Optional[str]
    
        :param lang_to_id: Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
        :type lang_to_id: dict[str, int]
    
        :param task: Task to use for generation, either “translate” or “transcribe”
        :type task: int
    
        :param return_timestamps: If `true` the pipeline will return timestamps along the text for *segments* of words in the text.
                           For instance, if you get
                           WhisperDecodedResultChunk
                               start_ts = 0.5
                               end_ts = 1.5
                               text = " Hi there!"
                           then it means the model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                           Note that a segment of text refers to a sequence of one or more words, rather than individual words.
        :type return_timestamps: bool
    
        :param initial_prompt: Initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing
        window. Can be used to steer the model to use particular spellings or styles.
    
        Example:
          auto result = pipeline.generate(raw_speech);
          //  He has gone and gone for good answered Paul Icrom who...
    
          auto result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
          //  He has gone and gone for good answered Polychrome who...
        :type initial_prompt: Optional[str]
    
        :param hotwords:  Hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows.
        Can be used to steer the model to use particular spellings or styles.
    
        Example:
          auto result = pipeline.generate(raw_speech);
          //  He has gone and gone for good answered Paul Icrom who...
    
          auto result = pipeline.generate(raw_speech, ov::genai::hotwords("Polychrome"));
          //  He has gone and gone for good answered Polychrome who...
        :type hotwords: Optional[str]
    
        Generic parameters:
        max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                       max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
        min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
        ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
        eos_token_id:  token_id of <eos> (end of sentence)
        stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
        include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
        stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
        echo:           if set to true, the model will echo the prompt in the output.
        logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                        Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
    
        repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
        presence_penalty: reduces absolute log prob if the token was generated at least once.
        frequency_penalty: reduces absolute log prob as many times as the token was generated.
    
        Beam search specific parameters:
        num_beams:         number of beams for beam search. 1 disables beam search.
        num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
        diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
        length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
            length_penalty < 0.0 encourages shorter sequences.
        num_return_sequences: the number of sequences to return for grouped beam search decoding.
        no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
        stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
            "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
            "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
            "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
    
        Random sampling parameters:
        temperature:        the value used to modulate token probabilities for random sampling.
        top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
        do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
        num_return_sequences: the number of sequences to generate from a single prompt.
    """
    hotwords: str | None
    initial_prompt: str | None
    is_multilingual: bool
    language: str | None
    return_timestamps: bool
    task: str | None
    @typing.overload
    def __init__(self, json_path: os.PathLike | str | bytes) -> None:
        """
        path where generation_config.json is stored
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def update_generation_config(self, **kwargs) -> None:
        ...
    @property
    def begin_suppress_tokens(self) -> list[int]:
        ...
    @begin_suppress_tokens.setter
    def begin_suppress_tokens(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def decoder_start_token_id(self) -> int:
        ...
    @decoder_start_token_id.setter
    def decoder_start_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def lang_to_id(self) -> dict[str, int]:
        ...
    @lang_to_id.setter
    def lang_to_id(self, arg0: collections.abc.Mapping[str, typing.SupportsInt]) -> None:
        ...
    @property
    def max_initial_timestamp_index(self) -> int:
        ...
    @max_initial_timestamp_index.setter
    def max_initial_timestamp_index(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def no_timestamps_token_id(self) -> int:
        ...
    @no_timestamps_token_id.setter
    def no_timestamps_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def pad_token_id(self) -> int:
        ...
    @pad_token_id.setter
    def pad_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def prev_sot_token_id(self) -> int:
        ...
    @prev_sot_token_id.setter
    def prev_sot_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def suppress_tokens(self) -> list[int]:
        ...
    @suppress_tokens.setter
    def suppress_tokens(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def transcribe_token_id(self) -> int:
        ...
    @transcribe_token_id.setter
    def transcribe_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def translate_token_id(self) -> int:
        ...
    @translate_token_id.setter
    def translate_token_id(self, arg0: typing.SupportsInt) -> None:
        ...
class WhisperPerfMetrics(PerfMetrics):
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param get_features_extraction_duration: Returns mean and standard deviation of features extraction duration in milliseconds
        :type get_features_extraction_duration: MeanStdPair
    
        :param whisper_raw_metrics: Whisper specific raw metrics
        :type WhisperRawPerfMetrics:
    """
    def __init__(self) -> None:
        ...
    def get_features_extraction_duration(self) -> MeanStdPair:
        ...
    @property
    def whisper_raw_metrics(self) -> WhisperRawPerfMetrics:
        ...
class WhisperPipeline:
    """
    Automatic speech recognition pipeline
    """
    def __init__(self, models_path: os.PathLike | str | bytes, device: str, **kwargs) -> None:
        """
                    WhisperPipeline class constructor.
                    models_path (os.PathLike): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU).
        """
    def generate(self, raw_speech_input: collections.abc.Sequence[typing.SupportsFloat], generation_config: openvino_genai.py_openvino_genai.WhisperGenerationConfig | None = None, streamer: collections.abc.Callable[[str], int | None] | openvino_genai.py_openvino_genai.StreamerBase | None = None, **kwargs) -> WhisperDecodedResults:
        """
            High level generate that receives raw speech as a vector of floats and returns decoded output.
        
            :param raw_speech_input: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
            :type raw_speech_input: list[float]
        
            :param generation_config: generation_config
            :type generation_config: WhisperGenerationConfig or a dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped.
                             Streamer supported for short-form audio (< 30 seconds) with `return_timestamps=False` only
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to WhisperGenerationConfig fields.
            :type : dict
        
            :return: return results in decoded form
            :rtype: WhisperDecodedResults
         
         
            WhisperGenerationConfig
            
            Whisper specific parameters:
            :param decoder_start_token_id: Corresponds to the ”<|startoftranscript|>” token.
            :type decoder_start_token_id: int
        
            :param pad_token_id: Padding token id.
            :type pad_token_id: int
        
            :param translate_token_id: Translate token id.
            :type translate_token_id: int
        
            :param transcribe_token_id: Transcribe token id.
            :type transcribe_token_id: int
        
            :param no_timestamps_token_id: No timestamps token id.
            :type no_timestamps_token_id: int
        
            :param prev_sot_token_id: Corresponds to the ”<|startofprev|>” token.
            :type prev_sot_token_id: int
        
            :param is_multilingual:
            :type is_multilingual: bool
        
            :param begin_suppress_tokens: A list containing tokens that will be suppressed at the beginning of the sampling process.
            :type begin_suppress_tokens: list[int]
        
            :param suppress_tokens: A list containing the non-speech tokens that will be suppressed during generation.
            :type suppress_tokens: list[int]
        
            :param language: Language token to use for generation in the form of <|en|>.
                             You can find all the possible language tokens in the generation_config.json lang_to_id dictionary.
            :type language: Optional[str]
        
            :param lang_to_id: Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
            :type lang_to_id: dict[str, int]
        
            :param task: Task to use for generation, either “translate” or “transcribe”
            :type task: int
        
            :param return_timestamps: If `true` the pipeline will return timestamps along the text for *segments* of words in the text.
                               For instance, if you get
                               WhisperDecodedResultChunk
                                   start_ts = 0.5
                                   end_ts = 1.5
                                   text = " Hi there!"
                               then it means the model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                               Note that a segment of text refers to a sequence of one or more words, rather than individual words.
            :type return_timestamps: bool
        
            :param initial_prompt: Initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing
            window. Can be used to steer the model to use particular spellings or styles.
        
            Example:
              auto result = pipeline.generate(raw_speech);
              //  He has gone and gone for good answered Paul Icrom who...
        
              auto result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
              //  He has gone and gone for good answered Polychrome who...
            :type initial_prompt: Optional[str]
        
            :param hotwords:  Hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows.
            Can be used to steer the model to use particular spellings or styles.
        
            Example:
              auto result = pipeline.generate(raw_speech);
              //  He has gone and gone for good answered Paul Icrom who...
        
              auto result = pipeline.generate(raw_speech, ov::genai::hotwords("Polychrome"));
              //  He has gone and gone for good answered Polychrome who...
            :type hotwords: Optional[str]
        
            Generic parameters:
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
        
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
            presence_penalty: reduces absolute log prob if the token was generated at least once.
            frequency_penalty: reduces absolute log prob as many times as the token was generated.
        
            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
                length_penalty < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
                "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
                "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
        
            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            num_return_sequences: the number of sequences to generate from a single prompt.
        """
    def get_generation_config(self) -> WhisperGenerationConfig:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def set_generation_config(self, config: WhisperGenerationConfig) -> None:
        ...
class WhisperRawPerfMetrics:
    """
    
        Structure with whisper specific raw performance metrics for each generation before any statistics are calculated.
    
        :param features_extraction_durations: Duration for each features extraction call.
        :type features_extraction_durations: list[MicroSeconds]
    """
    def __init__(self) -> None:
        ...
    @property
    def features_extraction_durations(self) -> list[float]:
        ...
def draft_model(models_path: os.PathLike | str | bytes, device: str = '', **kwargs) -> openvino._pyopenvino.OVAny:
    """
    device on which inference will be performed
    """
def get_version() -> str:
    """
    OpenVINO GenAI version
    """
