def pytest_make_parametrize_id(config, val, argname):
    if argname in ['prompt', 'promtps']:
        return f'{val}'
    elif argname == 'model_descr':
        return f"{val[0]}"
    elif argname in 'stop_criteria':
        return str(val)
    elif isinstance(val, (int, float, str)):
        return f'{argname}={val}'
    return None
