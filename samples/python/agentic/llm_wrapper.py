"""
OpenVINO GenAI LangChain Wrapper.
Provides a thread-safe, Pydantic-isolated LangChain BaseChatModel 
implementation for OpenVINO's LLMPipeline, supporting native stop-strings 
and generator-based streaming.
"""
import queue
import threading
from typing import Any, Dict, Iterator, List, Optional
from pydantic import ConfigDict, PrivateAttr
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import openvino_genai as ov_genai

class _Sentinel:
    pass

class QueueStreamer(ov_genai.StreamerBase):
    def __init__(self, token_queue: queue.Queue):
        super().__init__()
        self.token_queue = token_queue

    def put(self, token_id: int, token: str) -> bool:
        self.token_queue.put(token)
        return False

    def end(self) -> None:
        self.token_queue.put(_Sentinel())

class OpenVINOChatModel(BaseChatModel):
    model_path: str
    device: str = "CPU"
    generation_kwargs: Dict[str, Any] = {}
    
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    _pipeline: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device)

    @property
    def _llm_type(self) -> str:
        return "openvino_chat"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt_lines = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_lines.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_lines.append(f"Assistant: {msg.content}")
        return "\n".join(prompt_lines) + "\nAssistant:\n"

    def _build_config(self, stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
        config = self._pipeline.get_generation_config()
        params = {**self.generation_kwargs, **kwargs}
        for k, v in params.items():
            if hasattr(config, k):
                setattr(config, k, v)
        if stop:
            config.stop_strings = set(stop)
        return config

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        config = self._build_config(stop, **kwargs)
        
        response = self._pipeline.generate(prompt, config)
            
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        config = self._build_config(stop, **kwargs)
        token_queue: queue.Queue = queue.Queue()
        streamer = QueueStreamer(token_queue)
        
        def _run_generate() -> None:
            try:
                self._pipeline.generate(prompt, config, streamer)
            except Exception as e:
                token_queue.put(e)
                token_queue.put(_Sentinel())
        
        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        
        try:
            while True:
                token = token_queue.get()
                if isinstance(token, _Sentinel):
                    break
                if isinstance(token, Exception):
                    raise token
                
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        finally:
            thread.join(timeout=1.0)