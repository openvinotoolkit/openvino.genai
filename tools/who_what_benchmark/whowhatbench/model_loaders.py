# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import logging
import torch
import os
import json

from packaging.version import Version

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    __version__,
)

from .embeddings_evaluator import DEFAULT_MAX_LENGTH as EMBED_DEFAULT_MAX_LENGTH, Qwen3VLEmbeddingWrapper
from .reranking_evaluator import (
    DEFAULT_MAX_LENGTH as RERANK_DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_LENGTH_QWEN as RERANK_DEFAULT_MAX_LENGTH_QWEN,
    DEFAULT_TOP_K as RERANK_DEFAULT_TOP_K,
    is_qwen3_causallm,
    is_qwen3,
)
from .utils import (
    apply_peft_adapters,
    mock_torch_cuda_is_available,
    mock_AwqQuantizer_validate_environment,
    disable_diffusers_model_progress_bar,
    get_json_config,
    normalize_lora_adapters_and_alphas,
)

# hide transformers progress bar
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_load_kwargs(model_type, use_hf, use_genai, use_llamacpp, kwargs):
    sanitized_kwargs = dict(kwargs)
    n_ctx = sanitized_kwargs.get("llamacpp_n_ctx")
    is_text_task = model_type in ("text", "text-chat")
    is_llamacpp_text_backend = is_text_task and use_llamacpp and not use_hf and not use_genai

    if not use_llamacpp:
        if n_ctx is not None:
            raise ValueError("--llamacpp-n-ctx requires --llamacpp")
        sanitized_kwargs.pop("llamacpp_n_ctx", None)
        return sanitized_kwargs

    if is_llamacpp_text_backend:
        if n_ctx is None:
            sanitized_kwargs["llamacpp_n_ctx"] = 8192
        else:
            n_ctx_int = int(n_ctx)
            if n_ctx_int <= 0:
                raise ValueError("--llamacpp-n-ctx must be a positive integer")
            sanitized_kwargs["llamacpp_n_ctx"] = n_ctx_int
        return sanitized_kwargs

    if n_ctx is not None:
        raise ValueError("--llamacpp-n-ctx is supported only when llama.cpp is the selected text backend")

    sanitized_kwargs.pop("llamacpp_n_ctx", None)
    return sanitized_kwargs


def _create_genai_adapter_config(adapters=None, alphas=None, *, none_if_empty=False):
    import openvino_genai

    adapter_config = openvino_genai.AdapterConfig()
    if adapters is None:
        return None if none_if_empty else adapter_config

    adapters, alphas = normalize_lora_adapters_and_alphas(adapters, alphas)
    for adapter, alpha in zip(adapters, alphas):
        ov_adapter = openvino_genai.Adapter(adapter)
        adapter_config.add(ov_adapter, alpha)

    return adapter_config


class GenAIModelWrapper:
    """
    A helper class to store additional attributes for GenAI models
    """

    def __init__(self, model, model_dir, model_type):
        self.model = model
        self.model_dir = model_dir
        self.model_type = model_type

        if model_type in (
            "text",
            "text-chat",
            "visual-text",
            "visual-video-text",
            "embedding",
            "text-reranking",
            "visual-text-chat",
        ):
            try:
                self.config = AutoConfig.from_pretrained(model_dir)
            except Exception:
                self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif model_type in ("text-to-image", "text-to-video"):
            from diffusers import DiffusionPipeline
            try:
                self.config = DiffusionPipeline.load_config(model_dir)
            except Exception:
                self.config = DiffusionPipeline.load_config(model_dir, trust_remote_code=True)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.model, attr)


def configure_sparse_attention(scheduler_params, scheduler_config):
    """
    Configures sparse attention settings based on scheduler parameters.
    """
    import openvino_genai
    sparse_attention_kwargs = scheduler_params.pop('sparse_attention_config', None)

    if sparse_attention_kwargs:
        # Convert mode string to enum if present
        mode = sparse_attention_kwargs.get("mode")
        if mode:
            sparse_attention_kwargs["mode"] = getattr(openvino_genai.SparseAttentionMode, mode)

        # Check if sparse attention is enabled
        if scheduler_params.pop('use_sparse_attention', True):
            scheduler_config.use_sparse_attention = True
            scheduler_config.sparse_attention_config = openvino_genai.SparseAttentionConfig(**sparse_attention_kwargs)
            logger.info("Sparse Attention mode ON")
        else:
            raise RuntimeError("==Failure==: sparse_attention_config value can't be used with use_sparse_attention=False")


def get_scheduler_config_genai(cb_config):
    import openvino_genai

    default_cb_config = {"cache_size": 1}
    scheduler_config = openvino_genai.SchedulerConfig()
    scheduler_params = cb_config or default_cb_config
    if scheduler_params:
        logger.info(f"Scheduler parameters for:\n{scheduler_params}")
        configure_sparse_attention(scheduler_params, scheduler_config)
        for param, value in scheduler_params.items():
            if param == "cache_eviction_config":
                value = openvino_genai.CacheEvictionConfig(aggregation_mode=openvino_genai.AggregationMode.NORM_SUM, **value)
            setattr(scheduler_config, param, value)

    return scheduler_config


def load_text_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError:
        logger.error(
            "Failed to import openvino_genai package. Please install it.")
        exit(-1)

    pipeline_path = model_dir
    if kwargs.get('gguf_file'):
        pipeline_path = os.path.join(model_dir, kwargs['gguf_file'])

    adapter_config = _create_genai_adapter_config(
        adapters=kwargs.get("adapters"),
        alphas=kwargs.get("alphas", None),
    )

    draft_model_path = kwargs.get("draft_model", '')
    if draft_model_path:
        if not Path(draft_model_path).exists():
            raise RuntimeError(f"Error: Draft model path does not exist: {draft_model_path}")
        draft_device = kwargs.get("draft_device", None) or device
        draft_model_load_kwargs = (
            {"scheduler_config": get_scheduler_config_genai(kwargs["draft_cb_config"])}
            if kwargs["draft_cb_config"] is not None else {}
        )
        ov_config["draft_model"] = openvino_genai.draft_model(draft_model_path, draft_device.upper(), **draft_model_load_kwargs)

    is_continuous_batching = kwargs.get("cb_config", None) is not None

    if is_continuous_batching:
        logger.info("Using OpenVINO GenAI Continuous Batching API")
        scheduler_config = get_scheduler_config_genai(kwargs["cb_config"])
        pipeline = openvino_genai.LLMPipeline(pipeline_path, device=device, adapters=adapter_config, scheduler_config=scheduler_config, **ov_config)
    else:
        logger.info("Using OpenVINO GenAI LLMPipeline API")
        pipeline = openvino_genai.LLMPipeline(pipeline_path, device=device, adapters=adapter_config, **ov_config)

    return GenAIModelWrapper(pipeline, model_dir, "text")


def load_text_llamacpp_pipeline(model_dir, **kwargs):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Failed to import llama_cpp. Please install llama-cpp-python to use --llamacpp."
        ) from exc
    n_ctx = kwargs.get("llamacpp_n_ctx", None)
    model_kwargs = {}
    if n_ctx is not None:
        model_kwargs["n_ctx"] = int(n_ctx)
    model = Llama(model_dir, **model_kwargs)
    return model


def load_text_hf_pipeline(model_id, device, **kwargs):
    model_kwargs = {}
    trust_remote_code = False
    if kwargs.get('gguf_file'):
        model_kwargs['gguf_file'] = kwargs['gguf_file']
    else:
        try:
            config = AutoConfig.from_pretrained(model_id)
        except Exception:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            trust_remote_code = True

    if not torch.cuda.is_available() or device.lower() == "cpu":
        is_gptq = False
        is_awq = False
        if not kwargs.get("gguf_file") and config and getattr(config, "quantization_config", None):
            is_gptq = config.quantization_config["quant_method"] == "gptq"
            is_awq = config.quantization_config["quant_method"] == "awq"
        with mock_AwqQuantizer_validate_environment(is_awq), mock_torch_cuda_is_available(is_gptq or is_awq):
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, device_map="cpu", **model_kwargs)
        if is_awq:
            model.is_awq = is_awq
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=False, device_map=device.lower(), **model_kwargs
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device_map=device.lower(), **model_kwargs
            )

    if kwargs.get("adapters") is not None:
        model = apply_peft_adapters(model, kwargs["adapters"], kwargs.get("alphas", None))

    model.eval()
    return model


def load_text_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, use_llamacpp=False, **kwargs,
):
    if use_hf:
        logger.info("Using HF Transformers API")
        model = load_text_hf_pipeline(model_id, device, **kwargs)
    elif use_genai:
        model = load_text_genai_pipeline(model_id, device, ov_config, **kwargs)
    elif use_llamacpp:
        logger.info("Using llama.cpp API (n_ctx=%s)", kwargs.get("llamacpp_n_ctx"))
        model = load_text_llamacpp_pipeline(model_id, **kwargs)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVModelForCausalLM
        try:
            model = OVModelForCausalLM.from_pretrained(
                model_id, device=device, ov_config=ov_config, **kwargs
            )
        except Exception:
            try:
                config = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=True)
                model = OVModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=True,
                    use_cache=True,
                    device=device,
                    ov_config=ov_config,
                    **kwargs
                )
            except Exception:
                config = AutoConfig.from_pretrained(model_id)
                model = OVModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    use_cache=True,
                    device=device,
                    ov_config=ov_config,
                    **kwargs
                )

    return model


def load_text2image_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError:
        logger.error(
            "Failed to import openvino_genai package. Please install it.")
        exit(-1)

    ov_config = ov_config or {}

    if device.upper().startswith("NPU"):
        image_size = kwargs.get("image_size")
        if image_size is None or image_size <= 0:
            raise ValueError(
                "A positive --image-size must be provided for text-to-image GenAI evaluation on NPU "
                "because the pipeline must be reshaped to static dimensions before compilation"
            )

        pipe = openvino_genai.Text2ImagePipeline(model_dir)
        guidance_scale = pipe.get_generation_config().guidance_scale
        logger.info(
            "Reshaping text-to-image pipeline to static shapes for NPU: "
            f"num_images_per_prompt=1, height={image_size}, width={image_size}, guidance_scale={guidance_scale}"
        )
        pipe.reshape(
            num_images_per_prompt=1,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
        )
        pipe.compile(device, **ov_config)

        wrapper = GenAIModelWrapper(pipe, model_dir, "text-to-image")
        if kwargs.get("adapters") is not None:
            wrapper.adapter_config = _create_genai_adapter_config(
                adapters=kwargs.get("adapters"),
                alphas=kwargs.get("alphas", None),
            )
        return wrapper
    else:
        adapter_config = _create_genai_adapter_config(
            adapters=kwargs.get("adapters"),
            alphas=kwargs.get("alphas", None),
        )

        return GenAIModelWrapper(
            openvino_genai.Text2ImagePipeline(model_dir, device=device, adapters=adapter_config, **ov_config),
            model_dir,
            "text-to-image",
        )


def load_text2image_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs
):
    if use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_text2image_genai_pipeline(model_id, device, ov_config, **kwargs)
    elif use_hf:
        from diffusers import DiffusionPipeline

        logger.info("Using HF Transformers API")
        try:
            model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        except Exception:
            model = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
        if kwargs.get("adapters") is not None:
            adapters = kwargs["adapters"]
            alphas = kwargs.get("alphas", None)
            adapters, alphas = normalize_lora_adapters_and_alphas(adapters, alphas)

            for idx, adapter in enumerate(adapters):
                model.load_lora_weights(adapter, adapter_name=f"adapter_{idx}")
            model.set_adapters([f"adapter_{idx}" for idx in range(len(adapters))], adapter_weights=alphas)
    else:
        logger.info("Using Optimum API")
        from optimum.intel import OVPipelineForText2Image
        TEXT2IMAGEPipeline = OVPipelineForText2Image

        if "adapters" in kwargs and kwargs["adapters"] is not None:
            raise ValueError("Adapters are not supported for OVPipelineForText2Image.")

        model_kwargs = {"ov_config": ov_config, "safety_checker": None}
        if kwargs.get('from_onnx'):
            model_kwargs['from_onnx'] = kwargs['from_onnx']
        try:
            model = TEXT2IMAGEPipeline.from_pretrained(model_id, device=device, **model_kwargs)
        except ValueError:
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                **model_kwargs
            )

    disable_diffusers_model_progress_bar(model)
    return model


def load_visual_text_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    is_continuous_batching = kwargs.get("cb_config", None) is not None

    adapter_config = _create_genai_adapter_config(
        adapters=kwargs.get("adapters"),
        alphas=kwargs.get("alphas", None),
        none_if_empty=True,
    )

    pipeline_kwargs = {
        "device": device,
        **ov_config,
    }

    if adapter_config is not None:
        pipeline_kwargs["adapters"] = adapter_config

    if is_continuous_batching:
        logger.info("Using OpenVINO GenAI Continuous Batching API")
        scheduler_config = get_scheduler_config_genai(kwargs["cb_config"])
        pipeline_kwargs["scheduler_config"] = scheduler_config
        pipeline_kwargs["ATTENTION_BACKEND"] = "PA"
        pipeline = openvino_genai.VLMPipeline(model_dir, **pipeline_kwargs)
    else:
        logger.info("Using OpenVINO GenAI VLMPipeline API")
        pipeline = openvino_genai.VLMPipeline(model_dir, **pipeline_kwargs)

    return GenAIModelWrapper(
        pipeline,
        model_dir,
        kwargs.get("model_type", "visual-text")
    )


def load_visual_text_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs
):
    if use_hf:
        logger.info("Using HF Transformers API")

        trust_remote_code = False
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        except Exception:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            trust_remote_code = True

        # force downloading to .cache image_processing file, as it is not happened by default
        if config.model_type.lower() in ["minicpmo"]:
            from transformers import AutoImageProcessor

            AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

        model_kwargs = {"trust_remote_code": trust_remote_code}
        try:
            model_cls = None

            # AutoModelForVision2Seq was removed in transformers 5.0.0
            # let's try to use AutoModelForImageTextToText instead first
            transformers_version = Version(__version__)
            if config.model_type == "gemma4_unified":
                if transformers_version < Version("5.10.0"):
                    raise ImportError(f"gemma4_unified requires transformers>=5.10.0, got {__version__}.")
                from transformers import AutoModelForMultimodalLM

                model_cls = AutoModelForMultimodalLM
            elif config.model_type == "gemma3n":
                model_cls = AutoModelForCausalLM
                model_kwargs.update({"torch_dtype": torch.float32})
            elif transformers_version < Version("5.0.0"):
                from transformers import AutoModelForVision2Seq

                model_cls = AutoModelForVision2Seq
            else:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText

            model = model_cls.from_pretrained(model_id, device_map=device.lower(), **model_kwargs)
        except ValueError:
            try:
                model_cls = AutoModel
                if config.model_type in ["smolvlm"]:
                    from transformers import AutoModelForImageTextToText

                    model_cls = AutoModelForImageTextToText
                elif config.model_type in ["gemma3"]:
                    model_cls = AutoModelForCausalLM

                model = model_cls.from_pretrained(model_id, device_map=device.lower(), **model_kwargs)
            except ValueError:
                if config.model_type == "phi4mm" or config.model_type == "llava-qwen2":
                    if hasattr(config, "audio_processor") and "activation_checkpointing" in config.audio_processor["config"]:
                        config.audio_processor["config"]["activation_checkpointing"] = ""
                    config._attn_implementation = "sdpa"
                    from_pretrained_kwargs = {"config": config}
                else:
                    from_pretrained_kwargs = {"_attn_implementation": "eager", "use_flash_attention_2": False}

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device.lower(),
                    **from_pretrained_kwargs,
                    **model_kwargs,
                )

                # phi4mm modality-specific LoRA adapters (handled internally by the pipeline/model)
                if config.model_type == "phi4mm":
                    use_lora = False
                    if hasattr(config, "vision_lora") and config.vision_lora is not None:
                        model.set_lora_adapter("vision")
                        use_lora = True
                    if hasattr(config, "speech_lora") and config.speech_lora is not None:
                        model.set_lora_adapter("speech")
                        use_lora = True
                    if use_lora:
                        model.unset_lora_adapter = lambda: None
                        model.set_lora_adapter = lambda _: None
                    if hasattr(model.model, "_require_grads_hook"):
                        model.model.disable_input_require_grads()

        # Common LoRA support via PEFT
        if kwargs.get("adapters") is not None:
            model = apply_peft_adapters(model, kwargs["adapters"], kwargs.get("alphas", None))

        model.eval()
        try:
            model.get_vision_tower().load_model()
        except Exception:
            pass

        if "internvl" in model.config.model_type:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model.img_context_token_id = img_context_token_id
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_visual_text_genai_pipeline(model_id, device, ov_config, **kwargs)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVModelForVisualCausalLM

        if "adapters" in kwargs and kwargs["adapters"] is not None:
            raise ValueError("Adapters are not supported for OVModelForVisualCausalLM.")
        try:
            model = OVModelForVisualCausalLM.from_pretrained(
                model_id, device=device, ov_config=ov_config
            )
        except ValueError:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = OVModelForVisualCausalLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
            )
    return model


def load_image2image_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    return GenAIModelWrapper(
        openvino_genai.Image2ImagePipeline(model_dir, device, **ov_config),
        model_dir,
        "image-to-image"
    )


def load_imagetext2image_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs
):
    if use_hf:
        from diffusers import AutoPipelineForImage2Image

        logger.info("Using HF Transformers API")
        model = AutoPipelineForImage2Image.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_image2image_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVPipelineForImage2Image

        model_kwargs = {"ov_config": ov_config, "safety_checker": None}
        if kwargs.get('from_onnx'):
            model_kwargs['from_onnx'] = kwargs['from_onnx']
        try:
            model = OVPipelineForImage2Image.from_pretrained(model_id, device=device, **model_kwargs)
        except ValueError:
            model = OVPipelineForImage2Image.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                **model_kwargs
            )

    disable_diffusers_model_progress_bar(model)
    return model


def load_inpainting_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    return GenAIModelWrapper(
        openvino_genai.InpaintingPipeline(model_dir, device, **ov_config),
        model_dir,
        "image-inpainting"
    )


def load_inpainting_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs
):
    if use_hf:
        from diffusers import AutoPipelineForInpainting

        logger.info("Using HF Transformers API")
        model = AutoPipelineForInpainting.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_inpainting_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVPipelineForInpainting

        model_kwargs = {"ov_config": ov_config, "safety_checker": None}
        if kwargs.get('from_onnx'):
            model_kwargs['from_onnx'] = kwargs['from_onnx']
        try:
            model = OVPipelineForInpainting.from_pretrained(model_id, device=device, **model_kwargs)
        except ValueError as e:
            logger.error("Failed to load inpaiting pipeline. Details:\n", e)
            model = OVPipelineForInpainting.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                **model_kwargs
            )

    disable_diffusers_model_progress_bar(model)
    return model


def apply_embedding_model_wrapper(model, use_genai):
    if not use_genai and Qwen3VLEmbeddingWrapper.is_qwen3_vl_model(model):
        return Qwen3VLEmbeddingWrapper(model)

    return model


def load_embedding_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    config = openvino_genai.TextEmbeddingPipeline.Config()
    if kwargs.get("embeds_pooling"):
        if kwargs.get("embeds_pooling") == "mean":
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
        elif kwargs.get("embeds_pooling") == "last_token":
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN
        else:
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.CLS
    elif Qwen3VLEmbeddingWrapper.is_qwen3_vl_model(model_dir):
        config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN

    config.max_length = EMBED_DEFAULT_MAX_LENGTH
    config.normalize = kwargs.get("embeds_normalize", False)
    config.pad_to_max_length = True
    config.batch_size = kwargs.get("embeds_batch_size", config.batch_size)

    logger.info("Using OpenVINO GenAI TextEmbeddingPipeline API")
    if hasattr(openvino_genai, "EmbeddingPipeline"):
        pipeline = openvino_genai.EmbeddingPipeline(
            model_dir, device.upper(), text_embedding_config=config, **ov_config
        )
    else:
        pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, device.upper(), config, **ov_config)

    return GenAIModelWrapper(pipeline, model_dir, "embedding")


def load_embedding_model(model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs):
    if use_hf:
        from transformers import AutoModel

        logger.info("Using HF Transformers API")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_embedding_genai_pipeline(model_id, device, ov_config, **kwargs)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVModelForFeatureExtraction
        try:
            model = OVModelForFeatureExtraction.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            model = OVModelForFeatureExtraction.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None
            )
    model = apply_embedding_model_wrapper(model, use_genai)
    return model


def load_reranking_genai_pipeline(model_dir, device="CPU", ov_config=None, is_qwen3_model=False):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    logger.info("Using OpenVINO GenAI TextRerankPipeline API")

    config = openvino_genai.TextRerankPipeline.Config()
    config.top_n = RERANK_DEFAULT_TOP_K
    config.max_length = RERANK_DEFAULT_MAX_LENGTH_QWEN if is_qwen3_model else RERANK_DEFAULT_MAX_LENGTH

    pipeline = openvino_genai.TextRerankPipeline(model_dir, device.upper(), config, **ov_config)

    return GenAIModelWrapper(
        pipeline,
        model_dir,
        "text-reranking"
    )


def load_reranking_model(model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False):
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if use_hf:
        logger.info("Using HF Transformers API")
        if is_qwen3_causallm(config):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        else:
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        is_qwen3_model = is_qwen3(config)
        model = load_reranking_genai_pipeline(model_id, device, ov_config, is_qwen3_model)
    else:
        logger.info("Using Optimum API")
        model_cls = None
        if is_qwen3_causallm(config):
            from optimum.intel.openvino import OVModelForCausalLM
            model_cls = OVModelForCausalLM
        else:
            from optimum.intel.openvino import OVModelForSequenceClassification
            model_cls = OVModelForSequenceClassification

        try:
            model = model_cls.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            model = model_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=False,
                device=device,
                ov_config=ov_config,
                safety_checker=None
            )

    return model


def load_text2video_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    import openvino_genai

    adapter_config = _create_genai_adapter_config(
        adapters=kwargs.get("adapters"),
        alphas=kwargs.get("alphas", None),
    )
    return GenAIModelWrapper(
        openvino_genai.Text2VideoPipeline(model_dir, device=device, adapters=adapter_config, **ov_config),
        model_dir,
        "text-to-video",
    )


def load_text2video_model(model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs):
    if use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_text2video_genai_pipeline(model_id, device, ov_config, **kwargs)
    elif use_hf:
        from diffusers import LTXPipeline

        logger.info("Using HF Transformers API")
        try:
            model = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        except ValueError:
            model = LTXPipeline.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
        if kwargs.get("adapters") is not None:
            adapters = kwargs["adapters"]
            alphas = kwargs.get("alphas", None)
            adapters, alphas = normalize_lora_adapters_and_alphas(adapters, alphas)

            for idx, adapter in enumerate(adapters):
                model.load_lora_weights(adapter, adapter_name=f"adapter_{idx}")
            model.set_adapters([f"adapter_{idx}" for idx in range(len(adapters))], adapter_weights=alphas)
    else:
        logger.info("Using Optimum API")
        from optimum.intel import OVLTXPipeline

        if "adapters" in kwargs and kwargs["adapters"] is not None:
            raise ValueError("Adapters are not supported for OVLTXPipeline.")

        model_kwargs = {"ov_config": ov_config, "safety_checker": None}
        if kwargs.get("from_onnx"):
            model_kwargs["from_onnx"] = kwargs["from_onnx"]
        try:
            model = OVLTXPipeline.from_pretrained(model_id, device=device, **model_kwargs)
        except ValueError:
            model = OVLTXPipeline.from_pretrained(
                model_id, trust_remote_code=True, use_cache=True, device=device, **model_kwargs
            )

    disable_diffusers_model_progress_bar(model)
    return model


def load_speech_generation_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    import openvino_genai

    return GenAIModelWrapper(
        openvino_genai.Text2SpeechPipeline(model_dir, device=device, **(ov_config or {})),
        model_dir,
        "speech-generation",
    )


def _load_qwen3_tts_config(model_id):
    config = None

    model_path = Path(model_id) if isinstance(model_id, str) else None
    if model_path and model_path.is_dir():
        config_path = model_path / "config.json"
        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = None
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = None

    return config


def _get_qwen3_tts_model_type(model_id):
    config = _load_qwen3_tts_config(model_id)
    if not isinstance(config, dict):
        return None
    if config.get("model_type") != "qwen3_tts":
        return None
    return str(config.get("tts_model_type", "")).strip().lower() or None


def _is_qwen3_custom_voice_model(model_id):
    return _get_qwen3_tts_model_type(model_id) == "custom_voice"


def _is_qwen3_voice_design_model(model_id):
    return _get_qwen3_tts_model_type(model_id) == "voice_design"


def _is_qwen3_base_model(model_id):
    return _get_qwen3_tts_model_type(model_id) == "base"


def _map_qwen3_device(device: str) -> str:
    normalized_device = (device or "CPU").strip().lower()
    if normalized_device == "gpu":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return normalized_device


def load_qwen3_custom_voice_hf_pipeline(model_id, device="CPU", **kwargs):
    from qwen_tts import Qwen3TTSModel

    import torch

    device_map = _map_qwen3_device(device)
    from_pretrained_kwargs = {"device_map": device_map}
    if device_map.startswith("cuda"):
        from_pretrained_kwargs["dtype"] = torch.bfloat16
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        from_pretrained_kwargs["dtype"] = torch.float32

    model = Qwen3TTSModel.from_pretrained(model_id, **from_pretrained_kwargs)

    from .speech_generation_evaluator import Qwen3CustomVoiceWrapper

    return Qwen3CustomVoiceWrapper(model)


def load_qwen3_voice_design_hf_pipeline(model_id, device="CPU", **kwargs):
    from qwen_tts import Qwen3TTSModel

    import torch

    device_map = _map_qwen3_device(device)
    from_pretrained_kwargs = {"device_map": device_map}
    if device_map.startswith("cuda"):
        from_pretrained_kwargs["dtype"] = torch.bfloat16
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        from_pretrained_kwargs["dtype"] = torch.float32

    model = Qwen3TTSModel.from_pretrained(model_id, **from_pretrained_kwargs)

    from .speech_generation_evaluator import Qwen3VoiceDesignWrapper

    return Qwen3VoiceDesignWrapper(model)


def load_qwen3_base_hf_pipeline(model_id, device="CPU", **kwargs):
    from qwen_tts import Qwen3TTSModel

    import torch

    device_map = _map_qwen3_device(device)
    from_pretrained_kwargs = {"device_map": device_map}
    if device_map.startswith("cuda"):
        from_pretrained_kwargs["dtype"] = torch.bfloat16
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        from_pretrained_kwargs["dtype"] = torch.float32

    model = Qwen3TTSModel.from_pretrained(model_id, **from_pretrained_kwargs)

    from .speech_generation_evaluator import Qwen3BaseWrapper

    return Qwen3BaseWrapper(model)


def load_qwen3_custom_voice_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    import openvino_genai

    from .speech_generation_evaluator import Qwen3CustomVoiceWrapper

    return Qwen3CustomVoiceWrapper(openvino_genai.Text2SpeechPipeline(model_dir, device=device, **(ov_config or {})))


def load_qwen3_voice_design_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    import openvino_genai

    from .speech_generation_evaluator import Qwen3VoiceDesignWrapper

    return Qwen3VoiceDesignWrapper(openvino_genai.Text2SpeechPipeline(model_dir, device=device, **(ov_config or {})))


def load_qwen3_base_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    import openvino_genai

    from .speech_generation_evaluator import Qwen3BaseWrapper

    return Qwen3BaseWrapper(openvino_genai.Text2SpeechPipeline(model_dir, device=device, **(ov_config or {})))


def _resolve_remote_code_and_config(model_id):
    remote_code = False
    try:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        remote_code = True
    return remote_code, model_config


def _load_speecht5_processor(model_id, remote_code):
    from transformers import SpeechT5Processor

    return SpeechT5Processor.from_pretrained(model_id, trust_remote_code=remote_code)


def _load_speecht5_hifigan_vocoder(vocoder_path=None):
    from transformers import SpeechT5HifiGan

    if vocoder_path is not None:
        return SpeechT5HifiGan.from_pretrained(vocoder_path)
    return SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


def _is_kokoro_model_id(model_id):
    if not isinstance(model_id, str):
        return False

    # Robust detection for local exports (directory name can be arbitrary).
    model_path = Path(model_id)
    if model_path.is_dir() and (model_path / "voices").is_dir():
        return True

    return "kokoro" in model_id.lower()


def load_speech_generation_model(model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, **kwargs):
    from .speech_generation_evaluator import KokoroModelWrapper, SpeechT5Wrapper

    vocoder_path = kwargs.get("vocoder_path")
    qwen3_tts_model_type = _get_qwen3_tts_model_type(model_id)
    is_qwen3_custom_voice = qwen3_tts_model_type == "custom_voice"
    is_qwen3_voice_design = qwen3_tts_model_type == "voice_design"
    is_qwen3_base = qwen3_tts_model_type == "base"

    if use_hf:
        if is_qwen3_custom_voice:
            logger.info("Using Qwen3 CustomVoice HF API")
            return load_qwen3_custom_voice_hf_pipeline(model_id, device, **kwargs)

        if is_qwen3_voice_design:
            logger.info("Using Qwen3 VoiceDesign HF API")
            return load_qwen3_voice_design_hf_pipeline(model_id, device, **kwargs)

        if is_qwen3_base:
            logger.info("Using Qwen3 Base HF API")
            return load_qwen3_base_hf_pipeline(model_id, device, **kwargs)

        if _is_kokoro_model_id(model_id):
            logger.info("Using Kokoro HF API")
            return KokoroModelWrapper(model_id)

        logger.info("Using HF Transformers API")
        from transformers import SpeechT5ForTextToSpeech

        remote_code, _ = _resolve_remote_code_and_config(model_id)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_id, trust_remote_code=remote_code)
        processor = _load_speecht5_processor(model_id, remote_code)

        # for HF, we need to explicitly load the vocoder.
        # Assume it's microsoft/speecht5_hifigan for now.
        vocoder = _load_speecht5_hifigan_vocoder(vocoder_path)
        return SpeechT5Wrapper(model, processor, vocoder)

    if use_genai or is_qwen3_custom_voice or is_qwen3_voice_design or is_qwen3_base:
        if is_qwen3_custom_voice:
            logger.info("Using OpenVINO GenAI API for Qwen3 CustomVoice")
            return load_qwen3_custom_voice_genai_pipeline(model_id, device, ov_config, **kwargs)

        if is_qwen3_voice_design:
            logger.info("Using OpenVINO GenAI API for Qwen3 VoiceDesign")
            return load_qwen3_voice_design_genai_pipeline(model_id, device, ov_config, **kwargs)

        if is_qwen3_base:
            logger.info("Using OpenVINO GenAI API for Qwen3 Base")
            return load_qwen3_base_genai_pipeline(model_id, device, ov_config, **kwargs)

        logger.info("Using OpenVINO GenAI API")
        return load_speech_generation_genai_pipeline(model_id, device, ov_config, **kwargs)

    logger.info("Using Optimum API")
    from optimum.intel.openvino import OVModelForTextToSpeechSeq2Seq

    if _is_kokoro_model_id(model_id):
        model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
            model_id,
            device=device,
            ov_config=ov_config,
            trust_remote_code=True,
        )
        return KokoroModelWrapper(model_id, ov_model=model)

    remote_code, model_config = _resolve_remote_code_and_config(model_id)

    from_pretrained_kwargs = {
        "device": device,
        "ov_config": ov_config,
        "config": model_config,
        "trust_remote_code": remote_code,
    }
    if vocoder_path is not None:
        # Optimum forwards extra kwargs from from_pretrained() to export.
        # Pass vocoder so that SpeechT5 export can consume it.
        from_pretrained_kwargs["vocoder"] = vocoder_path

    model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
        model_id,
        **from_pretrained_kwargs,
    )
    processor = _load_speecht5_processor(model_id, remote_code)

    # For Optimum, we don't need to load vocoder as it should pick up openvino_vocoder IR by default.
    # And this currently matches GenAI behavior, which will also pick up the same openvino_vocoder IR.
    return SpeechT5Wrapper(model, processor, None)


def load_model(
    model_type, model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, use_llamacpp=False, **kwargs
):
    if model_id is None:
        return None

    if ov_config:
        ov_options = get_json_config(ov_config)
        logger.info(f"OpenVINO Config: {ov_options}")
    else:
        ov_options = {}

    sanitized_kwargs = _sanitize_load_kwargs(model_type, use_hf, use_genai, use_llamacpp, kwargs)

    if model_type == "text" or model_type == "text-chat":
        return load_text_model(model_id, device, ov_options, use_hf, use_genai, use_llamacpp, **sanitized_kwargs)
    elif model_type == "text-to-image":
        return load_text2image_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type == "visual-text" or model_type == "visual-video-text" or model_type == "visual-text-chat":
        sanitized_kwargs["model_type"] = model_type
        return load_visual_text_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type == "image-to-image":
        return load_imagetext2image_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type == "image-inpainting":
        return load_inpainting_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type in ("text-embedding", "image-embedding", "video-embedding"):
        return load_embedding_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type == "text-reranking":
        return load_reranking_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "text-to-video":
        return load_text2video_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    elif model_type == "speech-generation":
        return load_speech_generation_model(model_id, device, ov_options, use_hf, use_genai, **sanitized_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
