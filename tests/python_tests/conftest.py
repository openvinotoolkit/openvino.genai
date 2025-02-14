import pytest


def pytest_make_parametrize_id(config, val, argname):
    if argname in ['prompt', 'prompts', 'batched_prompts']:
        # Print only first 1000 characters of long prompts.
        if isinstance(val, list):
            return ', '.join([f'{v[:100]}' for v in val])
        else:
            return f'{val[:100]}'
    elif argname == 'model_descr':
        return f"{val[0]}"
    elif argname == 'chat_config':
        return f"{val[0]}"
    elif argname in ['stop_criteria', 'generation_config']:
        return str(val)
    elif isinstance(val, (int, float, str)):
        return f'{argname}={val}'
    return None

def pytest_addoption(parser):
    parser.addoption("--model_ids", help="Select models to run")

def pytest_configure(config: pytest.Config):
    marker = 'precommit' if config.getoption('-m') == 'precommit' else 'nightly'
    pytest.run_marker = marker
    pytest.selected_model_ids = config.getoption('--model_ids', default=None)

