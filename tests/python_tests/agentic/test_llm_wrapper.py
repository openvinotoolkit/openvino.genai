"""
Unit Test Suite for OpenVINO GenAI LangChain Wrapper.
Validates Pydantic isolation, stop-string mapping, and basic generation logic
using a mocked LLMPipeline to ensure CI/CD compatibility.
"""
import unittest
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_path = os.path.abspath(os.path.join(current_dir, "../../../samples/python/agentic/llm_wrapper.py"))

spec = importlib.util.spec_from_file_location("llm_wrapper", wrapper_path)
if spec and spec.loader:
    llm_wrapper = importlib.util.module_from_spec(spec)
    sys.modules["llm_wrapper"] = llm_wrapper
    spec.loader.exec_module(llm_wrapper)
    OpenVINOChatModel = llm_wrapper.OpenVINOChatModel
else:
    raise ImportError(f"Could not find llm_wrapper.py at: {wrapper_path}")

from langchain_core.messages import HumanMessage

class TestOpenVINOChatModel(unittest.TestCase):
    @patch('openvino_genai.LLMPipeline')
    def test_initialization_and_pydantic_isolation(self, mock_pipeline) -> None:
        model = OpenVINOChatModel(model_path="/fake/path", device="GPU")
        self.assertEqual(model.device, "GPU")
        self.assertTrue(hasattr(model, '_pipeline'))
        mock_pipeline.assert_called_once_with("/fake/path", "GPU")

    @patch('openvino_genai.LLMPipeline')
    def test_generate_non_streaming(self, mock_pipeline_class) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = "Mocked OpenVINO response"
        mock_pipeline_class.return_value = mock_pipeline
        
        model = OpenVINOChatModel(model_path="/fake/path")
        messages = [HumanMessage(content="Hello OpenVINO")]
        
        response = model.invoke(messages)
        self.assertEqual(response.content, "Mocked OpenVINO response")
        self.assertTrue(mock_pipeline.generate.called)

    @patch('openvino_genai.LLMPipeline')
    def test_stop_words_mapping(self, mock_pipeline_class) -> None:
        mock_pipeline = MagicMock()
        mock_config = MagicMock()
        mock_pipeline.get_generation_config.return_value = mock_config
        mock_pipeline_class.return_value = mock_pipeline
        
        model = OpenVINOChatModel(model_path="/fake/path")
        model.invoke([HumanMessage(content="Hello")], stop=["\nObservation:"])
        
        self.assertEqual(mock_config.stop_strings, {"\nObservation:"})

if __name__ == '__main__':
    unittest.main()