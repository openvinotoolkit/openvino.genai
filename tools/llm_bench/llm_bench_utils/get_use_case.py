import os
import json
import logging as log
from pathlib import Path
from typing import Optional, Tuple
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.config_class import (
    USE_CASES,
    UseCaseImageGen
)

KNOWN_FRAMEWORKS = ['pytorch', 'ov', 'dldt']

OTHER_IGNORE_MODEL_PATH_PARTS = ['compressed_weights']

IGNORE_MODEL_PATH_PARTS_SET = {
    x.lower() for x in (KNOWN_FRAMEWORKS + model_utils.KNOWN_PRECISIONS + OTHER_IGNORE_MODEL_PATH_PARTS)
}

# --- 1. Constants for maintainability ---
# Use a set for efficient 'in' checks.
DIFFUSERS_PIPELINE_TYPES = {
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusion3Pipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionXLInpaintPipeline",
    "FluxPipeline",
    "LatentConsistencyModelPipeline",
}


# --- 2. Helper function to reduce repetition (DRY Principle) ---
def log_and_return(case, model_type: str, model_name: str):
    """A helper to standardize success logging and returning."""
    log.info(f'==SUCCESS FOUND==: use_case: {case.task}, model_type: {model_type}, model_Name: {model_name}')
    return case, model_type, model_name


def safe_json_load(file_path: Path) -> Optional[dict]:
    """Safely loads a JSON file, returning None on failure."""
    try:
        if file_path.is_file():
            return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Could not read or parse JSON file at {file_path}: {e}")
    return None


def resolve_complex_model_types(config):
    model_type = config.get("model_type").lower().replace('_', '-')
    if model_type == "gemma3":
        return USE_CASES["visual_text_gen"][0], model_type
    if model_type == "gemma3-text":
        return USE_CASES["text_gen"][0], model_type
    if model_type in ["phi4mm", "phi4-multimodal"]:
        return USE_CASES["visual_text_gen"][0], model_type
    if model_type == "llama4":
        return USE_CASES["visual_text_gen"][0], model_type
    return None, None


def get_model_name(model_path: Path, task: Optional[str] = None) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    """
    Attempts to extract the model name and its use case/type from the given path.
    Falls back to extracting a name from the path if no match is found.
    """
    path = os.path.abspath(model_path)
    model_parts = path.split(os.sep)

    # Determine possible use cases based on task
    possible_use_cases = sum(USE_CASES.values(), [])
    if task:
        if task in UseCaseImageGen.TASK:
            possible_use_cases = USE_CASES.get("image_gen", [])
        else:
            possible_use_cases = USE_CASES.get(task, [])

    # Search for matching model type in path parts
    for part in reversed(model_parts):
        for use_case in possible_use_cases:
            for model_type in use_case.model_types:
                if part.lower().startswith(model_type):
                    return use_case, model_type, part

    # Fallback to extracting model name from path
    model_name = get_model_name_with_path_part(model_path)
    return None, None, model_name


def get_model_name_with_path_part(model_name_or_path: Path) -> Optional[str]:
    """
    Extracts a model name from the path, ignoring known framework/precision keywords.
    """
    model_path = Path(model_name_or_path)
    for part in reversed(model_path.parts):
        if part.lower() not in IGNORE_MODEL_PATH_PARTS_SET:
            return part
    return None


def normalize_model_ids(model_ids_list):
    return [m_id[:-1] if m_id.endswith('_') else m_id for m_id in model_ids_list]


def get_use_case_by_model_id(model_id, task=None):
    possible_use_cases = sum(list(USE_CASES.values()), [])
    if task:
        if task in list(UseCaseImageGen.TASK.keys()):
            possible_use_cases = USE_CASES["image_gen"]
        else:
            possible_use_cases = USE_CASES[task]
    for use_case in possible_use_cases:
        for m_type in normalize_model_ids(use_case.model_types):
            # TODO go to equality and raise error if use_cases is already found, as it will mean that
            # model with that task can be applicable to execute with different pipelines and user doesn't specify one
            if model_id.startswith(m_type):
                return use_case, m_type

    return None, None


def get_use_case(model_path: Path, task: Optional[str] = None):
    """
    Determines the use case, model type, and name for a given model.

    It tries several strategies in order:
    1. Diffusers: Checks for a 'model_index.json'.
    2. Transformers: Checks for a 'config.json'.
    3. GGUF: Inspects metadata if the file has a '.gguf' extension.
    4. Fallback: Uses an initial guess based on the model name.
    """
    # --- 3. Normalize input to a Path object once ---
    cur_case, cur_model_type, cur_model_name = get_model_name(model_path)

    # Strategy 1: Check for a Diffusers model via 'model_index.json'
    if (diffusers_config := safe_json_load(model_path / "model_index.json")):
        if (pipe_type := diffusers_config.get("_class_name")) in DIFFUSERS_PIPELINE_TYPES:
            model_type = pipe_type.replace("Pipeline", "")
            return log_and_return(USE_CASES["image_gen"][0], model_type, cur_model_name)

    # Strategy 2 & 3: Determine a 'model_id' from config or GGUF metadata
    model_id = None
    if (config := safe_json_load(model_path / "config.json")):
        # First, attempt resolution with more complex logic
        case, model_type = resolve_complex_model_types(config)
        if case and model_type:
            return log_and_return(case, model_type, cur_model_name)
        # Fallback to simple 'model_type' key
        if model_type_val := config.get("model_type"):
            model_id = str(model_type_val).lower().replace('_', '-')

    elif model_path.suffix == '.gguf' and model_path.is_file():
        # --- 4. Robustness: Handle missing dependency and parsing errors ---
        try:
            import gguf_parser
            parser = gguf_parser.GGUFParser(str(model_path))
            parser.parse()
            if arch := parser.metadata.get('general.architecture'):
                model_id = arch.lower()
        except ImportError:
            log.warning("Module 'gguf_parser' not found. Cannot inspect .gguf file.")
        except Exception as e:
            log.error(f"Failed to parse GGUF file {model_path}: {e}")

    # Use the derived model_id to find the use case
    if model_id:
        case, model_type = get_use_case_by_model_id(model_id, task)
        if case:
            # The original code used case.task, adjust if `case` is an object
            return log_and_return(case, model_type, cur_model_name)

    # Fallback Strategy: Use the initial guess if it was valid
    if cur_case:
        return log_and_return(cur_case, cur_model_type, cur_model_name)

    # --- 5. Simplified final error ---
    raise RuntimeError('==Failure FOUND==: no use_case found after checking all strategies.')
