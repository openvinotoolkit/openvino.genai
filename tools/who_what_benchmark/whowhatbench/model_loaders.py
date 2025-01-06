import logging
import json

from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoModelForVision2Seq
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIModelWrapper:
    """
    A helper class to store additional attributes for GenAI models
    """

    def __init__(self, model, model_dir, model_type):
        self.model = model
        self.model_type = model_type

        if model_type == "text" or model_type == "visual-text":
            self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif model_type == "text-to-image":
            self.config = DiffusionPipeline.load_config(
                model_dir, trust_remote_code=True)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.model, attr)


def load_text_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError:
        logger.error(
            "Failed to import openvino_genai package. Please install it.")
        exit(-1)
    return GenAIModelWrapper(openvino_genai.LLMPipeline(model_dir, device=device, **ov_config), model_dir, "text")


def load_text_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if use_hf:
        logger.info("Using HF Transformers API")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map=device.lower()
        )
        model.eval()
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_text_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVModelForCausalLM
        try:
            model = OVModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device=device, ov_config=ov_config
            )
        except ValueError:
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

    return model


def load_text2image_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError:
        logger.error(
            "Failed to import openvino_genai package. Please install it.")
        exit(-1)

    return GenAIModelWrapper(
        openvino_genai.Text2ImagePipeline(model_dir, device=device, **ov_config),
        model_dir,
        "text-to-image"
    )


def load_text2image_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if use_genai:
        logger.info("Using OpenvINO GenAI API")
        model = load_text2image_genai_pipeline(model_id, device, ov_config)
    elif use_hf:
        logger.info("Using HF Transformers API")
        model = DiffusionPipeline.from_pretrained(
            model_id, trust_remote_code=True)
    else:
        logger.info("Using Optimum API")
        from optimum.intel import OVPipelineForText2Image
        TEXT2IMAGEPipeline = OVPipelineForText2Image

        try:
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id, trust_remote_code=True, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True)
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None,
            )

    return model


def load_visual_text_genai_pipeline(model_dir, device="CPU", ov_config=None):
    try:
        import openvino_genai
    except ImportError as e:
        logger.error("Failed to import openvino_genai package. Please install it. Details:\n", e)
        exit(-1)

    return GenAIModelWrapper(
        openvino_genai.VLMPipeline(model_dir, device, **ov_config),
        model_dir,
        "visual-text"
    )


def load_visual_text_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if use_hf:
        logger.info("Using HF Transformers API")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, trust_remote_code=True, device_map=device.lower()
            )
        except ValueError:
            try:
                model = AutoModel.from_pretrained(
                    model_id, trust_remote_code=True, device_map=device.lower()
                )
            except ValueError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, device_map=device.lower(), _attn_implementation="eager", use_flash_attention_2=False
                )
        model.eval()
    elif use_genai:
        logger.info("Using OpenVINO GenAI API")
        model = load_visual_text_genai_pipeline(model_id, device, ov_config)
    else:
        logger.info("Using Optimum API")
        from optimum.intel.openvino import OVModelForVisualCausalLM
        try:
            model = OVModelForVisualCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device=device, ov_config=ov_config
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
                model_id, trust_remote_code=True, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = OVPipelineForImage2Image.from_pretrained(
                model_id,
                config=config,
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
                model_id, trust_remote_code=True, device=device, ov_config=ov_config, safety_checker=None,
            )
        except ValueError as e:
            logger.error("Failed to load inpaiting pipeline. Details:\n", e)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = OVPipelineForInpainting.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_config,
                safety_checker=None,
            )
    return model


def load_model(
    model_type, model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if model_id is None:
        return None

    if ov_config:
        with open(ov_config) as f:
            ov_options = json.load(f)
    else:
        ov_options = {}

    if model_type == "text":
        return load_text_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "text-to-image":
        return load_text2image_model(
            model_id, device, ov_options, use_hf, use_genai
        )
    elif model_type == "visual-text":
        return load_visual_text_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "image-to-image":
        return load_imagetext2image_model(model_id, device, ov_options, use_hf, use_genai)
    elif model_type == "image-inpainting":
        return load_inpainting_model(model_id, device, ov_options, use_hf, use_genai)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
