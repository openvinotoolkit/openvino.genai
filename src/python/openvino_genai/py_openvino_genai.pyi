"""
Pybind11 binding for Whisper Pipeline
"""
from __future__ import annotations
import openvino._pyopenvino
import os
import typing
__all__ = ['Adapter', 'AdapterConfig', 'AggregationMode', 'AutoencoderKL', 'CLIPTextModel', 'CLIPTextModelWithProjection', 'CacheEvictionConfig', 'ChunkStreamerBase', 'ContinuousBatchingPipeline', 'CppStdGenerator', 'DecodedResults', 'EncodedGenerationResult', 'EncodedResults', 'GenerationConfig', 'GenerationFinishReason', 'GenerationHandle', 'GenerationOutput', 'GenerationResult', 'GenerationStatus', 'Generator', 'ImageGenerationConfig', 'LLMPipeline', 'MeanStdPair', 'PerfMetrics', 'PipelineMetrics', 'RawPerfMetrics', 'Scheduler', 'SchedulerConfig', 'StopCriteria', 'StreamerBase', 'Text2ImagePipeline', 'TokenizedInputs', 'Tokenizer', 'UNet2DConditionModel', 'VLMDecodedResults', 'VLMPerfMetrics', 'VLMPipeline', 'VLMRawPerfMetrics', 'WhisperDecodedResultChunk', 'WhisperDecodedResults', 'WhisperGenerationConfig', 'WhisperPerfMetrics', 'WhisperPipeline', 'WhisperRawPerfMetrics', 'draft_model']
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
    def __init__(self, path: os.PathLike) -> None:
        """
                    Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
                    path (os.PathLike): Path to adapter file in safetensors format.
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
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
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
    def __init__(self, adapter: Adapter, alpha: float, mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapter: Adapter, mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapters: list[Adapter], mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def __init__(self, adapters: list[tuple[Adapter, float]], mode: AdapterConfig.Mode = ...) -> None:
        ...
    @typing.overload
    def add(self, adapter: Adapter, alpha: float) -> AdapterConfig:
        ...
    @typing.overload
    def add(self, adapter: Adapter) -> AdapterConfig:
        ...
    def get_adapters(self) -> list[Adapter]:
        ...
    def get_alpha(self, adapter: Adapter) -> float:
        ...
    def remove(self, adapter: Adapter) -> AdapterConfig:
        ...
    def set_alpha(self, adapter: Adapter, alpha: float) -> AdapterConfig:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
        block_out_channels: list[int]
        in_channels: int
        latent_channels: int
        out_channels: int
        scaling_factor: float
        def __init__(self, config_path: os.PathLike) -> None:
            ...
    @typing.overload
    def __init__(self, vae_decoder_path: os.PathLike) -> None:
        """
                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
        """
    @typing.overload
    def __init__(self, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike) -> None:
        """
                    AutoencoderKL class initialized with both encoder and decoder models.
                    vae_encoder_path (os.PathLike): VAE encoder directory.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
        """
    @typing.overload
    def __init__(self, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None:
        """
                    AutoencoderKL class initialized only with decoder model.
                    vae_decoder_path (os.PathLike): VAE decoder directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, vae_encoder_path: os.PathLike, vae_decoder_path: os.PathLike, device: str, **kwargs) -> None:
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
    def get_config(self) -> AutoencoderKL.Config:
        ...
    def get_vae_scale_factor(self) -> int:
        ...
    def reshape(self, batch_size: int, height: int, width: int) -> AutoencoderKL:
        ...
class CLIPTextModel:
    """
    CLIPTextModel class.
    """
    class Config:
        """
        This class is used for storing CLIPTextModel config.
        """
        max_position_embeddings: int
        num_hidden_layers: int
        def __init__(self, config_path: str) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """
                    CLIPTextModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
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
    def get_config(self) -> CLIPTextModel.Config:
        ...
    def get_output_tensor(self, idx: int) -> openvino._pyopenvino.Tensor:
        ...
    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: int) -> CLIPTextModel:
        ...
    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        ...
class CLIPTextModelWithProjection:
    """
    CLIPTextModelWithProjection class.
    """
    class Config:
        """
        This class is used for storing CLIPTextModelWithProjection config.
        """
        max_position_embeddings: int
        num_hidden_layers: int
        def __init__(self, config_path: os.PathLike) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """
                    CLIPTextModelWithProjection class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
        """
                    CLIPTextModelWithProjection class
                    root_dir (os.PathLike): Model root directory.
                    device (str): Device on which inference will be done.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, model: CLIPTextModelWithProjection) -> None:
        """
        CLIPTextModelWithProjection model
                    CLIPTextModelWithProjection class
                    model (CLIPTextModelWithProjection): CLIPTextModelWithProjection model
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def get_config(self) -> CLIPTextModelWithProjection.Config:
        ...
    def get_output_tensor(self, idx: int) -> openvino._pyopenvino.Tensor:
        ...
    def infer(self, pos_prompt: str, neg_prompt: str, do_classifier_free_guidance: bool) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: int) -> CLIPTextModelWithProjection:
        ...
    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        ...
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
    """
    aggregation_mode: AggregationMode
    def __init__(self, start_size: int, recent_size: int, max_cache_size: int, aggregation_mode: AggregationMode) -> None:
        ...
    def get_evictable_size(self) -> int:
        ...
    def get_max_cache_size(self) -> int:
        ...
    def get_recent_size(self) -> int:
        ...
    def get_start_size(self) -> int:
        ...
class ChunkStreamerBase:
    """
    
        Base class for chunk streamers. In order to use inherit from from this class.
    """
    def __init__(self) -> None:
        ...
    def end(self) -> None:
        """
        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """
    def put(self, arg0: int) -> bool:
        """
        Put is called every time new token is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
    def put_chunk(self, arg0: list[int]) -> bool:
        """
        Put is called every time new token chunk is generated. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
class ContinuousBatchingPipeline:
    """
    This class is used for generation with LLMs with continuous batchig
    """
    @typing.overload
    def __init__(self, models_path: str, scheduler_config: SchedulerConfig, device: str, properties: dict[str, typing.Any] = {}, tokenizer_properties: dict[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def __init__(self, models_path: str, tokenizer: Tokenizer, scheduler_config: SchedulerConfig, device: str, properties: dict[str, typing.Any] = {}) -> None:
        ...
    @typing.overload
    def add_request(self, request_id: int, input_ids: openvino._pyopenvino.Tensor, sampling_params: GenerationConfig) -> GenerationHandle:
        ...
    @typing.overload
    def add_request(self, request_id: int, prompt: str, sampling_params: GenerationConfig) -> GenerationHandle:
        ...
    @typing.overload
    def generate(self, input_ids: list[openvino._pyopenvino.Tensor], sampling_params: list[GenerationConfig], streamer: typing.Callable[[str], bool] | StreamerBase | None = None) -> list[EncodedGenerationResult]:
        ...
    @typing.overload
    def generate(self, prompts: list[str], sampling_params: list[GenerationConfig], streamer: typing.Callable[[str], bool] | StreamerBase | None = None) -> list[GenerationResult]:
        ...
    def get_config(self) -> GenerationConfig:
        ...
    def get_metrics(self) -> PipelineMetrics:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def has_non_finished_requests(self) -> bool:
        ...
    def step(self) -> None:
        ...
class CppStdGenerator(Generator):
    """
    This class wraps std::mt19937 pseudo-random generator.
    """
    def __init__(self, seed: int) -> None:
        ...
    def next(self) -> float:
        ...
    def randn_tensor(self, shape: openvino._pyopenvino.Shape) -> openvino._pyopenvino.Tensor:
        ...
class DecodedResults:
    """
    
        Structure to store resulting batched text outputs and scores for each batch.
        The first num_return_sequences elements correspond to the first batch element.
    
        Parameters: 
        texts:      vector of resulting sequences.
        scores:     scores for each sequence.
        metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
    """
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
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
            DROPPED_BY_PIPELINE = 3 - Currently not used, TODO: implement abort functionality.
            DROPPED_BY_HANDLE = 4 - Status set when generation handle is dropped.
    
    """
    m_generation_ids: list[list[int]]
    m_scores: list[float]
    def __init__(self) -> None:
        ...
    @property
    def m_request_id(self) -> int:
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
    """
    @property
    def perf_metrics(self) -> PerfMetrics:
        ...
    @property
    def scores(self) -> list[float]:
        ...
    @property
    def tokens(self) -> list[list[int]]:
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
        ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
        eos_token_id:  token_id of <eos> (end of sentence)
        min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens. Ignored for non continuous batching.
        stop_strings: list of strings that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
        include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
        stop_token_ids: list of tokens that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
        echo:           if set to true, the model will echo the prompt in the output.
        logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                        Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
    
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
        repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.    
    """
    adapters: AdapterConfig | None
    assistant_confidence_threshold: float
    diversity_penalty: float
    do_sample: bool
    echo: bool
    eos_token_id: int
    frequency_penalty: float
    ignore_eos: bool
    include_stop_str_in_output: bool
    length_penalty: float
    logprobs: int
    max_length: int
    max_new_tokens: int
    min_new_tokens: int
    no_repeat_ngram_size: int
    num_assistant_tokens: int
    num_beam_groups: int
    num_beams: int
    num_return_sequences: int
    presence_penalty: float
    repetition_penalty: float
    rng_seed: int
    stop_criteria: StopCriteria
    stop_strings: set[str]
    stop_token_ids: set[int]
    temperature: float
    top_k: int
    top_p: float
    @typing.overload
    def __init__(self, json_path: os.PathLike) -> None:
        """
        path where generation_config.json is stored
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def is_beam_search(self) -> bool:
        ...
    def is_greedy_decoding(self) -> bool:
        ...
    def is_speculative_decoding(self) -> bool:
        ...
    def set_eos_token_id(self, tokenizer_eos_token_id: int) -> None:
        ...
    def update_generation_config(self, config_map: dict[str, openvino._pyopenvino.OVAny]) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    def back(self) -> dict[int, GenerationOutput]:
        ...
    def can_read(self) -> bool:
        ...
    def drop(self) -> None:
        ...
    def get_status(self) -> GenerationStatus:
        ...
    def read(self) -> dict[int, GenerationOutput]:
        ...
    def read_all(self) -> list[GenerationOutput]:
        ...
class GenerationOutput:
    finish_reason: GenerationFinishReason
    generated_ids: list[int]
    generated_log_probs: list[float]
    score: float
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
            DROPPED_BY_PIPELINE = 3 - Currently not used, TODO: implement abort functionality.
            DROPPED_BY_HANDLE = 4 - Status set when generation handle is dropped.
    
    """
    m_generation_ids: list[str]
    m_scores: list[float]
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_generation_ids(self) -> list[str]:
        ...
    @property
    def m_request_id(self) -> int:
        ...
class GenerationStatus:
    """
    Members:
    
      RUNNING
    
      FINISHED
    
      IGNORED
    
      DROPPED_BY_PIPELINE
    
      DROPPED_BY_HANDLE
    """
    DROPPED_BY_HANDLE: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.DROPPED_BY_HANDLE: 4>
    DROPPED_BY_PIPELINE: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.DROPPED_BY_PIPELINE: 3>
    FINISHED: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.FINISHED: 1>
    IGNORED: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.IGNORED: 2>
    RUNNING: typing.ClassVar[GenerationStatus]  # value = <GenerationStatus.RUNNING: 0>
    __members__: typing.ClassVar[dict[str, GenerationStatus]]  # value = {'RUNNING': <GenerationStatus.RUNNING: 0>, 'FINISHED': <GenerationStatus.FINISHED: 1>, 'IGNORED': <GenerationStatus.IGNORED: 2>, 'DROPPED_BY_PIPELINE': <GenerationStatus.DROPPED_BY_PIPELINE: 3>, 'DROPPED_BY_HANDLE': <GenerationStatus.DROPPED_BY_HANDLE: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
class ImageGenerationConfig:
    """
    This class is used for storing generation config for image generation pipeline.
    """
    adapters: AdapterConfig | None
    generator: Generator
    guidance_scale: float
    height: int
    max_sequence_length: int
    negative_prompt: str | None
    negative_prompt_2: str | None
    negative_prompt_3: str | None
    num_images_per_prompt: int
    num_inference_steps: int
    prompt_2: str | None
    prompt_3: str | None
    strength: float
    width: int
    def __init__(self) -> None:
        ...
    def update_generation_config(self, **kwargs) -> None:
        ...
    def validate(self) -> None:
        ...
class LLMPipeline:
    """
    This class is used for generation with LLMs
    """
    def __call__(self, inputs: openvino._pyopenvino.Tensor | TokenizedInputs | str | list[str], generation_config: GenerationConfig | None = None, streamer: typing.Callable[[str], bool] | StreamerBase | None = None, **kwargs) -> EncodedResults | DecodedResults:
        """
            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.
        
            :param inputs: inputs in the form of string, list of strings or tokenized input_ids
            :type inputs: str, List[str], ov.genai.TokenizedInputs, or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict
        
            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str
         
         
            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group 
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will 
            be used while greedy and beam search parameters will not affect decoding at all.
        
            Parameters: 
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens. Ignored for non continuous batching.
            stop_strings: list of strings that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: list of tokens that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
        
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
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.    
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike, tokenizer: Tokenizer, device: str, config: dict[str, typing.Any] = {}, **kwargs) -> None:
        """
                    LLMPipeline class constructor for manually created openvino_genai.Tokenizer.
                    models_path (os.PathLike): Path to the model file.
                    tokenizer (openvino_genai.Tokenizer): tokenizer object.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike, device: str, config: dict[str, typing.Any] = {}, **kwargs) -> None:
        """
                    LLMPipeline class constructor.
                    models_path (os.PathLike): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    Add {"scheduler_config": ov_genai.SchedulerConfig} to config properties to create continuous batching pipeline.
                    kwargs: Device properties.
        """
    def finish_chat(self) -> None:
        ...
    def generate(self, inputs: openvino._pyopenvino.Tensor | TokenizedInputs | str | list[str], generation_config: GenerationConfig | None = None, streamer: typing.Callable[[str], bool] | StreamerBase | None = None, **kwargs) -> EncodedResults | DecodedResults:
        """
            Generates sequences or tokens for LLMs. If input is a string or list of strings then resulting sequences will be already detokenized.
        
            :param inputs: inputs in the form of string, list of strings or tokenized input_ids
            :type inputs: str, List[str], ov.genai.TokenizedInputs, or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict
        
            :return: return results in encoded, or decoded form depending on inputs type
            :rtype: DecodedResults, EncodedResults, str
         
         
            Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group 
            and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will 
            be used while greedy and beam search parameters will not affect decoding at all.
        
            Parameters: 
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens. Ignored for non continuous batching.
            stop_strings: list of strings that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
            stop_token_ids: list of tokens that will cause pipeline to stop generating further tokens. Ignored for non continuous batching.
            echo:           if set to true, the model will echo the prompt in the output.
            logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                            Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
        
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
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.    
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
    def __iter__(self) -> typing.Iterator[float]:
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
    
        PerfMetrics holds fields with mean and standard deviations for the following metrics:
        - Time To the First Token (TTFT), ms
        - Time per Output Token (TPOT), ms/token
        - Generate total duration, ms
        - Tokenization duration, ms
        - Detokenization duration, ms
        - Throughput, tokens/s
    
        Additional fields include:
        - Load time, ms
        - Number of generated tokens
        - Number of tokens in the input prompt
    
        Preferable way to access values is via get functions. Getters calculate mean and std values from raw_metrics and return pairs.
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
    
        :param get_throughput: Returns the mean and standard deviation of throughput in tokens per second.
        :type get_throughput: MeanStdPair
    
        :param get_generate_duration: Returns the mean and standard deviation of generate durations in milliseconds.
        :type get_generate_duration: MeanStdPair
    
        :param get_tokenization_duration: Returns the mean and standard deviation of tokenization durations in milliseconds.
        :type get_tokenization_duration: MeanStdPair
    
        :param get_detokenization_duration: Returns the mean and standard deviation of detokenization durations in milliseconds.
        :type get_detokenization_duration: MeanStdPair
    
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
class RawPerfMetrics:
    """
    
        Structure with raw performance metrics for each generation before any statistics are calculated.
    
        :param generate_durations: Durations for each generate call in milliseconds.
        :type generate_durations: List[float]
    
        :param tokenization_durations: Durations for the tokenization process in milliseconds.
        :type tokenization_durations: List[float]
    
        :param detokenization_durations: Durations for the detokenization process in milliseconds.
        :type detokenization_durations: List[float]
    
        :param m_times_to_first_token: Times to the first token for each call in milliseconds.
        :type m_times_to_first_token: List[float]
    
        :param m_new_token_times: Timestamps of generation every token or batch of tokens in milliseconds.
        :type m_new_token_times: List[double]
    
        :param token_infer_durations : Inference time for each token in milliseconds.
        :type batch_sizes: List[float]
    
        :param m_batch_sizes: Batch sizes for each generate call.
        :type m_batch_sizes: List[int]
    
        :param m_durations: Total durations for each generate call in milliseconds.
        :type m_durations: List[float]
    
        :param inference_durations : Total inference duration for each generate call in milliseconds.
        :type batch_sizes: List[float]
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
class Scheduler:
    """
    Scheduler for image generation pipelines.
    """
    class Type:
        """
        Members:
        
          AUTO
        
          LCM
        
          LMS_DISCRETE
        
          DDIM
        
          EULER_DISCRETE
        
          FLOW_MATCH_EULER_DISCRETE
        """
        AUTO: typing.ClassVar[Scheduler.Type]  # value = <Type.AUTO: 0>
        DDIM: typing.ClassVar[Scheduler.Type]  # value = <Type.DDIM: 3>
        EULER_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.EULER_DISCRETE: 4>
        FLOW_MATCH_EULER_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.FLOW_MATCH_EULER_DISCRETE: 5>
        LCM: typing.ClassVar[Scheduler.Type]  # value = <Type.LCM: 1>
        LMS_DISCRETE: typing.ClassVar[Scheduler.Type]  # value = <Type.LMS_DISCRETE: 2>
        __members__: typing.ClassVar[dict[str, Scheduler.Type]]  # value = {'AUTO': <Type.AUTO: 0>, 'LCM': <Type.LCM: 1>, 'LMS_DISCRETE': <Type.LMS_DISCRETE: 2>, 'DDIM': <Type.DDIM: 3>, 'EULER_DISCRETE': <Type.EULER_DISCRETE: 4>, 'FLOW_MATCH_EULER_DISCRETE': <Type.FLOW_MATCH_EULER_DISCRETE: 5>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
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
    def from_config(scheduler_config_path: os.PathLike, scheduler_type: Scheduler.Type = ...) -> Scheduler:
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
            KV-caches can be rewritten if KV-cache limit is reached, but blocks are not released.
            This results in more RAM usage, maximum RAM usage is determined by cache_size or num_kv_blocks parameters.
            When turend off only KV-cache required for batch calculation is kept in memory and
            when a sequence has finished genegartion its cache is released.
    """
    cache_eviction_config: CacheEvictionConfig
    cache_size: int
    dynamic_split_fuse: bool
    enable_prefix_caching: bool
    max_num_batched_tokens: int
    max_num_seqs: int
    num_kv_blocks: int
    use_cache_eviction: bool
    def __init__(self) -> None:
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
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
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
    
        Base class for streamers. In order to use inherit from from this class and implement put, and methods.
    """
    def __init__(self) -> None:
        ...
    def end(self) -> None:
        """
        End is called at the end of generation. It can be used to flush cache if your own streamer has one
        """
    def put(self, token: int) -> bool:
        """
        Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
        """
class Text2ImagePipeline:
    """
    This class is used for generation with text-to-image models.
    """
    @staticmethod
    def latent_consistency_model(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion(scheduler: Scheduler, clip_text_model: CLIPTextModel, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @staticmethod
    def stable_diffusion_xl(scheduler: Scheduler, clip_text_model: CLIPTextModel, clip_text_model_with_projection: CLIPTextModelWithProjection, unet: UNet2DConditionModel, vae: AutoencoderKL) -> Text2ImagePipeline:
        ...
    @typing.overload
    def __init__(self, models_path: os.PathLike) -> None:
        """
                    Text2ImagePipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
        """
    @typing.overload
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """
                    Text2ImagePipeline class constructor.
                    models_path (os.PathLike): Path with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU).
                    kwargs: Text2ImagePipeline properties
        """
    def compile(self, device: str, **kwargs) -> None:
        """
                        Compiles the model.
                        device (str): Device to run the model on (e.g., CPU, GPU).
                        kwargs: Device properties.
        """
    def decode(self, latent: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
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
            generator: openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator,
            adapters: LoRA adapters,
            strength: strength for image to image generation. 1.0f means initial image is fully noised,
            max_sequence_length: int - length of t5_encoder_model input
        
            :return: ov.Tensor with resulting images
            :rtype: ov.Tensor
        """
    def get_generation_config(self) -> ImageGenerationConfig:
        ...
    def reshape(self, num_images_per_prompt: int, height: int, width: int, guidance_scale: float) -> None:
        ...
    def set_generation_config(self, generation_config: ImageGenerationConfig) -> None:
        ...
    def set_scheduler(self, scheduler: Scheduler) -> None:
        ...
class TokenizedInputs:
    attention_mask: openvino._pyopenvino.Tensor
    input_ids: openvino._pyopenvino.Tensor
    def __init__(self, input_ids: openvino._pyopenvino.Tensor, attention_mask: openvino._pyopenvino.Tensor) -> None:
        ...
class Tokenizer:
    """
    openvino_genai.Tokenizer object is used to initialize Tokenizer
               if it's located in a different path than the main model.
    """
    def __init__(self, tokenizer_path: os.PathLike, properties: dict[str, typing.Any] = {}, **kwargs) -> None:
        ...
    def apply_chat_template(self, history: list[dict[str, str]], add_generation_prompt: bool, chat_template: str = '') -> str:
        """
        Embeds input prompts with special tags for a chat scenario.
        """
    @typing.overload
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a sequence into a string prompt.
        """
    @typing.overload
    def decode(self, tokens: openvino._pyopenvino.Tensor, skip_special_tokens: bool = True) -> list[str]:
        """
        Decode tensor into a list of string prompts.
        """
    @typing.overload
    def decode(self, tokens: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        """
        Decode a batch of tokens into a list of string prompt.
        """
    @typing.overload
    def encode(self, prompts: list[str], add_special_tokens: bool = True) -> TokenizedInputs:
        """
        Encodes a list of prompts into tokenized inputs.
        """
    @typing.overload
    def encode(self, prompt: str, add_special_tokens: bool = True) -> TokenizedInputs:
        """
        Encodes a single prompt into tokenized input.
        """
    def get_bos_token(self) -> str:
        ...
    def get_bos_token_id(self) -> int:
        ...
    def get_eos_token(self) -> str:
        ...
    def get_eos_token_id(self) -> int:
        ...
    def get_pad_token(self) -> str:
        ...
    def get_pad_token_id(self) -> int:
        ...
    def set_chat_template(self, chat_template: str) -> None:
        """
        Override a chat_template read from tokenizer_config.json.
        """
class UNet2DConditionModel:
    """
    UNet2DConditionModel class.
    """
    class Config:
        """
        This class is used for storing UNet2DConditionModel config.
        """
        in_channels: int
        sample_size: int
        time_cond_proj_dim: int
        def __init__(self, config_path: os.PathLike) -> None:
            ...
    @typing.overload
    def __init__(self, root_dir: os.PathLike) -> None:
        """
                    UNet2DConditionModel class
                    root_dir (os.PathLike): Model root directory.
        """
    @typing.overload
    def __init__(self, root_dir: os.PathLike, device: str, **kwargs) -> None:
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
    def do_classifier_free_guidance(self, guidance_scale: float) -> bool:
        ...
    def get_config(self) -> UNet2DConditionModel.Config:
        ...
    def infer(self, sample: openvino._pyopenvino.Tensor, timestep: openvino._pyopenvino.Tensor) -> openvino._pyopenvino.Tensor:
        ...
    def reshape(self, batch_size: int, height: int, width: int, tokenizer_model_max_length: int) -> UNet2DConditionModel:
        ...
    def set_adapters(self, adapters: AdapterConfig | None) -> None:
        ...
    def set_hidden_states(self, tensor_name: str, encoder_hidden_states: openvino._pyopenvino.Tensor) -> None:
        ...
class VLMDecodedResults:
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
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """
        device on which inference will be done
                    VLMPipeline class constructor.
                    models_path (os.PathLike): Path to the folder with exported model files.
                    device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
                    kwargs: Device properties
        """
    def finish_chat(self) -> None:
        ...
    @typing.overload
    def generate(self, prompt: str, images: list[openvino._pyopenvino.Tensor], generation_config: GenerationConfig, streamer: typing.Callable[[str], bool] | StreamerBase | None = None, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    @typing.overload
    def generate(self, prompt: str, images: openvino._pyopenvino.Tensor, generation_config: GenerationConfig, streamer: typing.Callable[[str], bool] | StreamerBase | None = None, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param images: image or list of images
            :type images: List[ov.Tensor] or ov.Tensor
        
            :param generation_config: generation_config
            :type generation_config: GenerationConfig or a Dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
            :type : Dict
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    @typing.overload
    def generate(self, prompt: str, **kwargs) -> VLMDecodedResults:
        """
            Generates sequences for VLMs.
        
            :param prompt: input prompt
            :type prompt: str
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.
        
            Expected parameters list:
            image: ov.Tensor - input image,
            images: List[ov.Tensor] - input images,
            generation_config: GenerationConfig,
            streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped
        
            :return: return results in decoded form
            :rtype: VLMDecodedResults
        """
    def get_generation_config(self) -> GenerationConfig:
        ...
    def get_tokenizer(self) -> Tokenizer:
        ...
    def set_chat_template(self, new_template: str) -> None:
        ...
    def set_generation_config(self, new_config: GenerationConfig) -> None:
        ...
    def start_chat(self, system_message: str = '') -> None:
        ...
class VLMRawPerfMetrics:
    """
    
        Structure with VLM specific raw performance metrics for each generation before any statistics are calculated.
    
        :param prepare_embeddings_durations: Durations of embeddings preparation.
        :type prepare_embeddings_durations: List[MicroSeconds]
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
class WhisperGenerationConfig:
    """
    
        WhisperGenerationConfig
        :param max_length: the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                           `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        :type max_length: int
    
        :param max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
        :type max_new_tokens: int
    
        :param eos_token_id: End of stream token id.
        :type eos_token_id: int
    
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
        :type lang_to_id: Dict[str, int]
    
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
    """
    begin_suppress_tokens: list[int]
    decoder_start_token_id: int
    eos_token_id: int
    is_multilingual: bool
    lang_to_id: dict[str, int]
    language: str | None
    max_initial_timestamp_index: int
    max_length: int
    max_new_tokens: int
    no_timestamps_token_id: int
    pad_token_id: int
    return_timestamps: bool
    suppress_tokens: list[int]
    task: str | None
    transcribe_token_id: int
    translate_token_id: int
    @typing.overload
    def __init__(self, json_path: os.PathLike) -> None:
        """
        path where generation_config.json is stored
        """
    @typing.overload
    def __init__(self, **kwargs) -> None:
        ...
    def set_eos_token_id(self, tokenizer_eos_token_id: int) -> None:
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
    def __init__(self, models_path: os.PathLike, device: str, **kwargs) -> None:
        """
                    WhisperPipeline class constructor.
                    models_path (os.PathLike): Path to the model file.
                    device (str): Device to run the model on (e.g., CPU, GPU).
        """
    def generate(self, raw_speech_input: list[float], generation_config: WhisperGenerationConfig | None = None, streamer: typing.Callable[[str], bool] | ChunkStreamerBase | None = None, **kwargs) -> WhisperDecodedResults:
        """
            High level generate that receives raw speech as a vector of floats and returns decoded output.
        
            :param raw_speech_input: inputs in the form of list of floats. Required to be normalized to near [-1, 1] range and have 16k Hz sampling rate.
            :type raw_speech_input: List[float]
        
            :param generation_config: generation_config
            :type generation_config: WhisperGenerationConfig or a Dict
        
            :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped.
                             Streamer supported for short-form audio (< 30 seconds) with `return_timestamps=False` only
            :type : Callable[[str], bool], ov.genai.StreamerBase
        
            :param kwargs: arbitrary keyword arguments with keys corresponding to WhisperGenerationConfig fields.
            :type : Dict
        
            :return: return results in decoded form
            :rtype: WhisperDecodedResults
         
         
            WhisperGenerationConfig
            :param max_length: the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                               `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            :type max_length: int
        
            :param max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            :type max_new_tokens: int
        
            :param eos_token_id: End of stream token id.
            :type eos_token_id: int
        
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
            :type lang_to_id: Dict[str, int]
        
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
        :type features_extraction_durations: List[MicroSeconds]
    """
    def __init__(self) -> None:
        ...
    @property
    def features_extraction_durations(self) -> list[float]:
        ...
class draft_model:
    """
    This class is used to enable Speculative Decoding
    """
    def __init__(self, models_path: os.PathLike, device: str = '', **kwargs) -> None:
        """
        device on which inference will be performed
        """
