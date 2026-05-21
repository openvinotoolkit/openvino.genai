# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as _logging

__version__ = "0.1.0"

_logger = _logging.getLogger(__name__)

from .registry import register_evaluator, EVALUATOR_REGISTRY  # noqa: E402
from .text_evaluator import TextEvaluator  # noqa: E402
from .text_evaluator import TextEvaluator as Evaluator  # noqa: E402
from .visualtext_evaluator import VisualTextEvaluator  # noqa: E402
from .embeddings_evaluator import EmbeddingsEvaluator  # noqa: E402
from .reranking_evaluator import RerankingEvaluator  # noqa: E402
from .chat_text_evaluator import ChatTextEvaluator  # noqa: E402
from .chat_visualtext_evaluator import ChatVisualTextEvaluator  # noqa: E402
from .scenario import Scenario, load_scenario  # noqa: E402

# Evaluators below require openvino_genai at import time.
# Wrap in try/except so the package stays importable in environments where
# openvino_genai is unavailable (e.g. during scenario dry-runs or CI without GPU).
try:
    from .text2image_evaluator import Text2ImageEvaluator
    from .im2im_evaluator import Image2ImageEvaluator
    from .inpaint_evaluator import InpaintingEvaluator
    from .text2video_evaluator import Text2VideoEvaluator
    from .speech_generation_evaluator import SpeechGenerationEvaluator
except ImportError as _e:
    _logger.warning(
        "Some evaluators (text-to-image, text-to-video, speech-generation, image-to-image, "
        "image-inpainting) could not be loaded because openvino_genai is unavailable: %s. "
        "These task types will not be available.",
        _e,
    )
    Text2ImageEvaluator = None  # type: ignore[assignment,misc]
    Image2ImageEvaluator = None  # type: ignore[assignment,misc]
    InpaintingEvaluator = None  # type: ignore[assignment,misc]
    Text2VideoEvaluator = None  # type: ignore[assignment,misc]
    SpeechGenerationEvaluator = None  # type: ignore[assignment,misc]

    # Register stub evaluators so callers get a helpful ImportError instead of
    # a KeyError on the registry lookup or a TypeError("'NoneType' is not callable")
    # at instantiation time.
    def _make_genai_stub(_task_type: str, _exc: ImportError) -> type:
        class _GenAIStub:
            def __init__(self, *_args, **_kwargs) -> None:
                raise ImportError(
                    f"Task type {_task_type!r} requires openvino_genai. "
                    f"Install with: pip install openvino-genai. Original error: {_exc}"
                ) from _exc

        _GenAIStub.__name__ = f"_Stub_{_task_type.replace('-', '_')}"
        return _GenAIStub

    for _task in (
        "text-to-image",
        "text-to-video",
        "image-to-image",
        "image-inpainting",
        "speech-generation",
    ):
        # Guard against partial-import races: if a module managed to register
        # before failing on a deeper openvino_genai dependency, leave it alone.
        if _task not in EVALUATOR_REGISTRY:
            register_evaluator(_task)(_make_genai_stub(_task, _e))


__all__ = [
    "Evaluator",
    "register_evaluator",
    "TextEvaluator",
    "Text2ImageEvaluator",
    "VisualTextEvaluator",
    "Image2ImageEvaluator",
    "InpaintingEvaluator",
    "EmbeddingsEvaluator",
    "RerankingEvaluator",
    "Text2VideoEvaluator",
    "ChatTextEvaluator",
    "SpeechGenerationEvaluator",
    "ChatVisualTextEvaluator",
    "EVALUATOR_REGISTRY",
    "Scenario",
    "load_scenario",
]
