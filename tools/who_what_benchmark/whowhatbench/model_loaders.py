import logging
import json
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoModelForVision2Seq, AutoTokenizer

from .embeddings_evaluator import DEFAULT_MAX_LENGTH as EMBED_DEFAULT_MAX_LENGTH
from .reranking_evaluator import DEFAULT_MAX_LENGTH as RERANK_DEFAULT_MAX_LENGTH, DEFAULT_TOP_K as RERANK_DEFAULT_TOP_K, reranking_base_on_causallm_arch
from .utils import mock_torch_cuda_is_available, mock_AwqQuantizer_validate_environment
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIModelWrapper:
    """
    A helper class to store additional attributes for GenAI models
    """

    def __init__(self, model, model_dir, model_type):
        self.model = model
        self.model_type = model_type

        if model_type in ["text", "visual-text", "text-embedding", "text-reranking"]:
            try:
                self.config = AutoConfig.from_pretrained(model_dir)
            except Exception:
                self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif model_type == "text-to-image":
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


def get_scheduler_config_genai(cb_config):
    import openvino_genai

    default_cb_config = {"cache_size": 1}
    scheduler_config = openvino_genai.SchedulerConfig()
    scheduler_params = cb_config or default_cb_config
    if scheduler_params:
        logger.info(f"Scheduler parameters for:\n{scheduler_params}")
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

    is_continuous_batching = kwargs.get("cb_config", None) is not None

    if is_continuous_batching:
        logger.info("Using OpenVINO GenAI Continuous Batching API")
        scheduler_config = get_scheduler_config_genai(kwargs["cb_config"])
        pipeline = openvino_genai.LLMPipeline(pipeline_path, device=device, scheduler_config=scheduler_config, **ov_config)
    else:
        logger.info("Using OpenVINO GenAI LLMPipeline API")
        pipeline = openvino_genai.LLMPipeline(pipeline_path, device=device, **ov_config)

    return GenAIModelWrapper(pipeline, model_dir, "text")


def load_text_llamacpp_pipeline(model_dir):
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "Failed to import llama_cpp package. Please install llama-cpp-python.")
        exit(-1)
    model = Llama(model_dir)
    return model


def load_text_hf_pipeline(model_id, device, **kwargs):
    model_kwargs = {}
    if kwargs.get('gguf_file'):
        model_kwargs['gguf_file'] = kwargs['gguf_file']
    if not torch.cuda.is_available or device.lower() == "cpu":
        trust_remote_code = False
        is_gptq = False
        is_awq = False
        if not kwargs.get('gguf_file'):
            try:
                config = AutoConfig.from_pretrained(model_id)
            except Exception:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                trust_remote_code = True

            if getattr(config, "quantization_config", None):
                is_gptq = config.quantization_config["quant_method"] == "gptq"
                is_awq = config.quantization_config["quant_method"] == "awq"
        if is_gptq or is_awq:
            # infer in FP32
            model_kwargs["torch_dtype"] = torch.float32
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
        logger.info("Using llama.cpp API")
        model = load_text_llamacpp_pipeline(model_id)
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
                )
            except Exception:
                config = AutoConfig.from_pretrained(model_id)
                model = OVModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    use_cache=True,
                    device=device,
                    ov_config=ov_config,
                )

    return model


def load_text2image_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError:
        logger.error(
            "Failed to import openvino_genai package. Please install it.")
        exit(-1)

    adapter_config = openvino_genai.AdapterConfig()
    if "adapters" in kwargs and kwargs["adapters"] is not None:
        for adapter, alpha in zip(kwargs['adapters'], kwargs['alphas']):
            ov_adapter = openvino_genai.Adapter(adapter)
            adapter_config.add(ov_adapter, alpha)

    return GenAIModelWrapper(
        openvino_genai.Text2ImagePipeline(model_dir, device=device, adapters=adapter_config, **ov_config),
        model_dir,
        "text-to-image"
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
            model = DiffusionPipeline.from_pretrained(model_id)
        except Exception:
            model = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True)
        if 'adapters' in kwargs and kwargs['adapters'] is not None:
            for idx, adapter in enumerate(kwargs['adapters']):
                model.load_lora_weights(adapter, adapter_name=f"adapter_{idx}")
            model.set_adapters([f"adapter_{idx}" for idx in range(len(kwargs['adapters']))], adapter_weights=kwargs['alphas'])
    else:
        logger.info("Using Optimum API")
        from optimum.intel import OVPipelineForText2Image
        TEXT2IMAGEPipeline = OVPipelineForText2Image

        if 'adapters' in kwargs and kwargs['adapters'] is not None:
            raise ValueError("Adapters are not supported for OVPipelineForText2Image.")

        try:
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None,
            )

    return model


def load_visual_text_genai_pipeline(model_dir, device="CPU", ov_config=None, **kwargs):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    is_continuous_batching = kwargs.get("cb_config", None) is not None

    if is_continuous_batching:
        logger.info("Using OpenVINO GenAI Continuous Batching API")
        scheduler_config = get_scheduler_config_genai(kwargs["cb_config"])
        pipeline = openvino_genai.VLMPipeline(model_dir, device=device, scheduler_config=scheduler_config, ATTENTION_BACKEND="PA", **ov_config)
    else:
        logger.info("Using OpenVINO GenAI VLMPipeline API")
        pipeline = openvino_genai.VLMPipeline(model_dir, device=device, **ov_config)

    return GenAIModelWrapper(
        pipeline,
        model_dir,
        "visual-text"
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

        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, device_map=device.lower()
            )
        except ValueError:
            try:
                model = AutoModel.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code, device_map=device.lower()
                )
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
                    trust_remote_code=trust_remote_code,
                    device_map=device.lower(),
                    **from_pretrained_kwargs,
                )

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
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if use_hf:
        from diffusers import AutoPipelineForImage2Image
        logger.info("Using HF Transformers API")
        model = AutoPipelineForImage2Image.from_pretrained(
            model_id, trust_remote_code=True
        )
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_image2image_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVPipelineForImage2Image
        try:
            model = OVPipelineForImage2Image.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            model = OVPipelineForImage2Image.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None,
            )
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
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if use_hf:
        from diffusers import AutoPipelineForInpainting
        logger.info("Using HF Transformers API")
        model = AutoPipelineForInpainting.from_pretrained(
            model_id, trust_remote_code=True
        )
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_inpainting_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVPipelineForInpainting
        try:
            model = OVPipelineForInpainting.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError as e:
            logger.error("Failed to load inpaiting pipeline. Details:\n", e)
            model = OVPipelineForInpainting.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None,
            )
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
    config.max_length = EMBED_DEFAULT_MAX_LENGTH
    config.normalize = kwargs.get("embeds_normalize", False)
    config.pad_to_max_length = True

    logger.info("Using OpenVINO GenAI TextEmbeddingPipeline API")
    pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, device.upper(), config, **ov_config)

    return GenAIModelWrapper(
        pipeline,
        model_dir,
        "text-embedding"
    )


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
        except ValueError as e:
            logger.error("Failed to load embedding pipeline. Details:\n", e)
            model = OVModelForFeatureExtraction.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None
            )
    return model


def load_reranking_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    logger.info("Using OpenVINO GenAI TextRerankPipeline API")

    config = openvino_genai.TextRerankPipeline.Config()
    config.top_n = RERANK_DEFAULT_TOP_K
    config.max_length = RERANK_DEFAULT_MAX_LENGTH

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
        if reranking_base_on_causallm_arch(config):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_reranking_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        model_cls = None
        if reranking_base_on_causallm_arch(config):
            from optimum.intel.openvino import OVModelForCausalLM
            model_cls = OVModelForCausalLM
        else:
            from optimum.intel.openvino import OVModelForSequenceClassification
            model_cls = OVModelForSequenceClassification

        try:
            model = model_cls.from_pretrained(
                model_id, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError as e:
            logger.error("Failed to load reranking pipeline, an attempt will be made again with updated parameters. Details:\n", e)
            model = model_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_cache=False,
                device=device,
                ov_config=ov_config,
                safety_checker=None
            )

    return model


def load_model(
    model_type, model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False, use_llamacpp=False, **kwargs
):
    if model_id is None:
        return None

    if ov_config:
        with open(ov_config) as f:
            ov_options = json.load(f)
    else:
        ov_options = {}

    if model_type == "text":
        return load_text_model(model_id, device, ov_options, use_hf, use_genai, use_llamacpp, **kwargs)
    elif model_type == "text-to-image":
        return load_text2image_model(
            model_id, device, ov_options, use_hf, use_genai, **kwargs
        )
    elif model_type == "visual-text":
        return load_visual_text_model(model_id, device, ov_options, use_hf, use_genai, **kwargs)
    elif model_type == "image-to-image":
        return load_imagetext2image_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "image-inpainting":
        return load_inpainting_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "text-embedding":
        return load_embedding_model(model_id, device, ov_options, use_hf, use_genai, **kwargs)
    elif model_type == "text-reranking":
        return load_reranking_model(model_id, device, ov_options, use_hf, use_genai)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
