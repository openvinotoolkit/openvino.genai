#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import queue
import threading
from typing import Any

import openvino_genai as ov_genai
from pydantic import ConfigDict, Field, PrivateAttr

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
except ImportError as import_error:
    raise ImportError(
        "LangChain core is required for OpenVINOChatModel. Install langchain-core to use this sample wrapper."
    ) from import_error


class OpenVINOChatModel(BaseChatModel):
    """LangChain-compatible chat model built on top of ``openvino_genai.LLMPipeline``."""

    model_path: str = Field(description="Path to OpenVINO model directory")
    device: str = Field(default="CPU", description="Device used for inference")
    streaming: bool = Field(default=False, description="Enable streaming by default")
    max_new_tokens: int = Field(default=256, ge=1, description="Default max new tokens")
    temperature: float = Field(default=0.0, ge=0.0, description="Default temperature")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Default top-p value")
    num_return_sequences: int = Field(default=1, ge=1, description="Default number of returned sequences")
    repetition_penalty: float = Field(default=1.0, gt=0.0, description="Default repetition penalty")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _pipe: ov_genai.LLMPipeline = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        """Initialize OpenVINO pipeline and validate wrapper-level invariants."""
        super().__init__(**data)

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        if not os.path.isdir(self.model_path):
            raise ValueError(f"Model directory does not exist or is not a directory: {self.model_path}")

        if self.num_return_sequences != 1:
            raise ValueError(
                "OpenVINOChatModel currently supports only num_return_sequences=1. "
                "Use a single sequence for deterministic chat output."
            )

        self._pipe = ov_genai.LLMPipeline(self.model_path, self.device)

    @property
    def _llm_type(self) -> str:
        return "openvino_genai_chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return model parameters used by LangChain for cache/identity semantics."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "streaming": self.streaming,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
        }

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate runtime generation kwargs accepted by this MVP wrapper."""
        supported = {
            "max_new_tokens",
            "temperature",
            "top_p",
            "num_return_sequences",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "structured_output_config",
        }

        unsupported = [key for key in kwargs.keys() if key not in supported]
        if unsupported:
            unsupported_list = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported generation kwargs: {unsupported_list}")

        if "num_return_sequences" in kwargs and kwargs["num_return_sequences"] != 1:
            raise ValueError(
                "OpenVINOChatModel currently supports only num_return_sequences=1 in runtime kwargs."
            )

        if "structured_output_config" in kwargs:
            struct_cfg = kwargs["structured_output_config"]
            if struct_cfg is not None and not isinstance(struct_cfg, ov_genai.StructuredOutputConfig):
                raise TypeError(
                    "structured_output_config must be openvino_genai.StructuredOutputConfig or None"
                )

    def _message_content_to_text(self, message: BaseMessage) -> str:
        """Extract message text and enforce string-only content for this wrapper."""
        if not isinstance(message.content, str):
            raise TypeError(
                f"Unsupported message content type {type(message.content).__name__} for {type(message).__name__}. "
                "Only string content is supported in this wrapper."
            )
        return message.content

    def _to_chat_history(self, messages: list[BaseMessage]) -> ov_genai.ChatHistory:
        """Convert LangChain messages into OpenVINO ``ChatHistory`` format."""
        history = ov_genai.ChatHistory()
        for message in messages:
            content = self._message_content_to_text(message)
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                raise ValueError(
                    f"Unsupported message type: {type(message).__name__}. "
                    "Supported types are SystemMessage, HumanMessage and AIMessage."
                )

            history.append({"role": role, "content": content})

        return history

    def _build_generation_config(self, stop: list[str] | None = None, **kwargs: Any) -> ov_genai.GenerationConfig:
        """Build validated generation configuration from defaults and runtime overrides."""
        self._validate_kwargs(kwargs)

        config = self._pipe.get_generation_config()
        config.max_new_tokens = self.max_new_tokens
        config.temperature = self.temperature
        config.top_p = self.top_p
        config.num_return_sequences = self.num_return_sequences
        config.repetition_penalty = self.repetition_penalty

        for key, value in kwargs.items():
            setattr(config, key, value)

        if stop:
            config.stop_strings = set(stop)

        config.validate()
        return config

    def _extract_text(self, output: Any) -> str:
        """Normalize OpenVINO generation output to a single assistant text."""
        if isinstance(output, str):
            return output

        texts = getattr(output, "texts", None)
        if isinstance(texts, list) and texts:
            if not isinstance(texts[0], str):
                raise TypeError("First generated text is not a string")
            return texts[0]

        raise ValueError("Unexpected OpenVINO generation result type")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a single non-streaming chat response."""
        _ = run_manager
        history = self._to_chat_history(messages)
        config = self._build_generation_config(stop=stop, **kwargs)

        output = self._pipe.generate(history, config)
        text = self._extract_text(output)

        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        """Stream assistant text chunks while generating in a background thread."""
        history = self._to_chat_history(messages)
        config = self._build_generation_config(stop=stop, **kwargs)

        # Bounded queue applies backpressure and prevents unbounded buffering.
        chunks: queue.Queue[str | None] = queue.Queue(maxsize=128)
        generation_error: dict[str, Exception | None] = {"error": None}

        def callback(subword: str) -> ov_genai.StreamingStatus:
            chunks.put(subword)
            return ov_genai.StreamingStatus.RUNNING

        def run_generation() -> None:
            try:
                self._pipe.generate(history, config, callback)
            except Exception as ex:  # pragma: no cover - explicit propagation path
                generation_error["error"] = ex
            finally:
                chunks.put(None)

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

        while True:
            piece = chunks.get()
            if piece is None:
                break

            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager is not None:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

        generation_thread.join()
        if generation_error["error"] is not None:
            raise generation_error["error"]
