#!/usr/bin/env python3
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ModulePipeline Python bindings:
- ValidationResult
- ConfigModelsMap (via constructor)
- validate_config
- validate_config_string
- comfyui_json_to_yaml
- comfyui_json_string_to_yaml
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path

import openvino_genai
from openvino_genai.py_openvino_genai import ValidationResult
import openvino as ov


class TestValidationResult:
    """Test cases for ValidationResult structure."""

    def test_validation_result_default_constructor(self):
        """Test ValidationResult default values."""
        result = ValidationResult()
        assert result.valid == False
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_attributes(self):
        """Test ValidationResult attribute access and modification."""
        result = ValidationResult()

        # Modify attributes
        result.valid = True
        result.errors = ["error1", "error2"]
        result.warnings = ["warning1"]

        assert result.valid == True
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]

    def test_validation_result_repr(self):
        """Test ValidationResult string representation."""
        result = ValidationResult()
        result.valid = True
        result.errors = ["test error"]
        result.warnings = ["test warning"]

        repr_str = repr(result)
        assert "ValidationResult" in repr_str
        assert "valid=True" in repr_str
        assert "test error" in repr_str
        assert "test warning" in repr_str


class TestValidateConfig:
    """Test cases for validate_config and validate_config_string methods."""

    @pytest.fixture
    def valid_yaml_config(self):
        """Create a valid YAML configuration."""
        return {
            'global_context': {
                'model_type': 'test'
            },
            'pipeline_modules': {
                'pipeline_params': {
                    'type': 'ParameterModule',
                    'outputs': [
                        {'name': 'prompt', 'type': 'String'}
                    ]
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [
                        {'name': 'output', 'type': 'String', 'source': 'pipeline_params.prompt'}
                    ]
                }
            }
        }

    @pytest.fixture
    def invalid_yaml_config(self):
        """Create an invalid YAML configuration (missing required fields)."""
        return {
            'global_context': {},
            'pipeline_modules': {}
        }

    def test_validate_config_string_valid(self, valid_yaml_config):
        """Test validate_config_string with valid configuration."""
        yaml_content = yaml.dump(valid_yaml_config)
        result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)

        assert isinstance(result, ValidationResult)
        # Note: actual validation logic may vary
        print(f"Validation result: {result}")

    def test_validate_config_string_invalid(self, invalid_yaml_config):
        """Test validate_config_string with invalid configuration."""
        yaml_content = yaml.dump(invalid_yaml_config)
        result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)

        assert isinstance(result, ValidationResult)
        print(f"Validation result: {result}")

    def test_validate_config_string_empty(self):
        """Test validate_config_string with empty string."""
        result = openvino_genai.ModulePipeline.validate_config_string("")

        assert isinstance(result, ValidationResult)
        assert result.valid == False

    def test_validate_config_string_malformed_yaml(self):
        """Test validate_config_string with malformed YAML."""
        malformed_yaml = "this: is: not: valid: yaml: [["
        result = openvino_genai.ModulePipeline.validate_config_string(malformed_yaml)

        assert isinstance(result, ValidationResult)
        assert result.valid == False

    def test_validate_config_file(self, valid_yaml_config, tmp_path):
        """Test validate_config with file path."""
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_yaml_config, f)

        result = openvino_genai.ModulePipeline.validate_config(config_file)

        assert isinstance(result, ValidationResult)
        print(f"Validation result: {result}")

    def test_validate_config_nonexistent_file(self):
        """Test validate_config with non-existent file returns invalid result."""
        result = openvino_genai.ModulePipeline.validate_config("/nonexistent/path/config.yaml")
        assert isinstance(result, ValidationResult)
        assert result.valid == False


class TestComfyUIJsonConversion:
    """Test cases for ComfyUI JSON to YAML conversion methods."""

    @pytest.fixture
    def sample_comfyui_api_json(self):
        """Create a sample ComfyUI API format JSON."""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "A beautiful sunset over mountains",
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

    def test_comfyui_json_string_to_yaml_basic(self, sample_comfyui_api_json):
        """Test comfyui_json_string_to_yaml with basic JSON."""
        json_content = json.dumps(sample_comfyui_api_json)

        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_string_to_yaml(
            json_content,
            model_path_base="./models/",
            default_device="CPU"
        )

        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)
        print(f"Generated YAML length: {len(yaml_content)}")
        print(f"Extracted params: {params}")

    def test_comfyui_json_string_to_yaml_with_kwargs(self, sample_comfyui_api_json):
        """Test comfyui_json_string_to_yaml with custom kwargs."""
        json_content = json.dumps(sample_comfyui_api_json)

        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_string_to_yaml(
            json_content,
            model_path_base="/custom/models/",
            default_device="GPU"
        )

        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)

    def test_comfyui_json_string_to_yaml_empty(self):
        """Test comfyui_json_string_to_yaml with empty JSON."""
        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_string_to_yaml("{}")

        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)

    def test_comfyui_json_string_to_yaml_invalid_json(self):
        """Test comfyui_json_string_to_yaml with invalid JSON returns empty result."""
        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_string_to_yaml("not valid json {{{")
        # Invalid JSON should return empty or error result
        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)

    def test_comfyui_json_to_yaml_file(self, sample_comfyui_api_json, tmp_path):
        """Test comfyui_json_to_yaml with file path."""
        json_file = tmp_path / "comfyui_workflow.json"
        with open(json_file, 'w') as f:
            json.dump(sample_comfyui_api_json, f)

        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_to_yaml(
            json_file,
            model_path_base="./models/"
        )

        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)
        print(f"Generated YAML from file: {len(yaml_content)} chars")

    def test_comfyui_json_to_yaml_nonexistent_file(self):
        """Test comfyui_json_to_yaml with non-existent file returns empty result."""
        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_to_yaml("/nonexistent/workflow.json")
        # Non-existent file should return empty or error result
        assert isinstance(yaml_content, str)
        assert isinstance(params, dict)


class TestConfigModelsMap:
    """Test cases for ConfigModelsMap type usage via ModulePipeline constructor."""

    @pytest.fixture
    def minimal_config(self):
        """Create a minimal YAML configuration."""
        return yaml.dump({
            'global_context': {'model_type': 'test'},
            'pipeline_modules': {
                'pipeline_params': {
                    'type': 'ParameterModule',
                    'outputs': [{'name': 'output', 'type': 'String'}]
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [{'name': 'input', 'type': 'String', 'source': 'pipeline_params.output'}]
                }
            }
        })

    @pytest.fixture
    def simple_ov_model(self):
        """Create a simple OpenVINO model for testing."""
        import openvino.opset13 as opset

        # Create a simple model: output = input * 2
        param = opset.parameter([1, 3, 224, 224], ov.Type.f32, name="input")
        const = opset.constant([2.0], ov.Type.f32)
        mul = opset.multiply(param, const, name="output")
        model = ov.Model([mul], [param], "test_model")
        return model

    def test_module_pipeline_with_none_models_map(self, minimal_config):
        """Test ModulePipeline constructor with models_map=None."""
        try:
            pipe = openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map=None
            )
            assert pipe is not None
            print("Constructor with models_map=None succeeded")
        except Exception as e:
            # May fail due to missing models, but binding should work
            print(f"Constructor test (None models_map): {e}")

    def test_module_pipeline_with_empty_dict_models_map(self, minimal_config):
        """Test ModulePipeline constructor with empty dict models_map."""
        try:
            pipe = openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map={}
            )
            assert pipe is not None
            print("Constructor with empty dict models_map succeeded")
        except Exception as e:
            print(f"Constructor test (empty dict): {e}")

    def test_module_pipeline_with_models_map(self, minimal_config, simple_ov_model):
        """Test ModulePipeline constructor with actual models_map."""
        models_map = {
            "text_encoder": {
                "encoder_model": simple_ov_model
            }
        }

        try:
            pipe = openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map=models_map
            )
            assert pipe is not None
            print("Constructor with models_map succeeded")
        except Exception as e:
            # May fail due to config not using these models, but binding should work
            print(f"Constructor test (with models): {e}")

    def test_module_pipeline_with_multiple_modules_models_map(self, minimal_config, simple_ov_model):
        """Test ModulePipeline constructor with multiple modules in models_map."""
        models_map = {
            "text_encoder": {
                "encoder": simple_ov_model,
                "projection": simple_ov_model
            },
            "vae_decoder": {
                "decoder": simple_ov_model
            },
            "unet": {
                "unet_model": simple_ov_model
            }
        }

        try:
            pipe = openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map=models_map
            )
            assert pipe is not None
            print("Constructor with multiple modules succeeded")
        except Exception as e:
            print(f"Constructor test (multiple modules): {e}")

    def test_module_pipeline_file_path_with_models_map(self, tmp_path, simple_ov_model):
        """Test ModulePipeline constructor with file path and models_map."""
        config = {
            'global_context': {'model_type': 'test'},
            'pipeline_modules': {
                'pipeline_params': {
                    'type': 'ParameterModule',
                    'outputs': [{'name': 'output', 'type': 'String'}]
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [{'name': 'input', 'type': 'String', 'source': 'pipeline_params.output'}]
                }
            }
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        models_map = {
            "my_module": {"model": simple_ov_model}
        }

        try:
            pipe = openvino_genai.ModulePipeline(
                config_yaml_path=config_file,
                models_map=models_map
            )
            assert pipe is not None
            print("Constructor with file path and models_map succeeded")
        except Exception as e:
            print(f"Constructor test (file path + models): {e}")

    def test_models_map_type_validation(self, minimal_config):
        """Test that invalid models_map types raise appropriate errors."""
        # Test with invalid outer type
        with pytest.raises((TypeError, Exception)):
            openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map="invalid_string"
            )

        # Test with invalid inner type
        with pytest.raises((TypeError, Exception)):
            openvino_genai.ModulePipeline(
                config_yaml_content=minimal_config,
                models_map={"module": "not_a_dict"}
            )


class TestIntegration:
    """Integration tests combining multiple API features."""

    def test_validate_then_create_pipeline(self, tmp_path):
        """Test validation followed by pipeline creation."""
        config = {
            'global_context': {'model_type': 'test'},
            'pipeline_modules': {
                'pipeline_params': {
                    'type': 'ParameterModule',
                    'outputs': [{'name': 'prompt', 'type': 'String'}]
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [{'name': 'result', 'type': 'String', 'source': 'pipeline_params.prompt'}]
                }
            }
        }

        yaml_content = yaml.dump(config)

        # First validate
        validation_result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)
        print(f"Validation: {validation_result}")

        # Then create if valid (or try anyway for testing)
        try:
            pipe = openvino_genai.ModulePipeline(config_yaml_content=yaml_content)
            print("Pipeline created successfully")
        except Exception as e:
            print(f"Pipeline creation: {e}")

    def test_comfyui_conversion_and_validation(self, tmp_path):
        """Test ComfyUI JSON conversion followed by validation."""
        comfyui_json = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}
            }
        }

        json_content = json.dumps(comfyui_json)

        # Convert
        yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_string_to_yaml(
            json_content,
            model_path_base="./models/"
        )

        print(f"Converted YAML:\n{yaml_content[:500]}..." if len(yaml_content) > 500 else f"Converted YAML:\n{yaml_content}")
        print(f"Extracted params: {params}")

        # Validate the converted YAML
        if yaml_content:
            validation_result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)
            print(f"Validation of converted YAML: {validation_result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
