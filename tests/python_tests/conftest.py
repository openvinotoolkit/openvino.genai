def pytest_make_parametrize_id(config, val, argname):
    if argname in ['prompt', 'promtps']:
        return f'{val}'
    if argname in 'stop_criteria':
        return str(val)
    if isinstance(val, (int, float, str)):
        return f'{argname}={val}'
    return None
