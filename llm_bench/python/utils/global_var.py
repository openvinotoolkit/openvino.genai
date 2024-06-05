# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log


def _init():
    """init global dict."""
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    """set key value for global dict."""
    _global_dict[key] = value


def get_value(key):
    """get value from key."""
    try:
        return _global_dict[key]
    except Exception:
        log.error('get' + key + 'failed\r\n')
