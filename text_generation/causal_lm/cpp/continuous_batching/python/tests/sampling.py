# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from common import run_test_pipeline

# tested models:
# - facebook/opt-125m
# - meta-llama/Llama-2-7b-chat-hf
# - mistralai/Mistral-7B-Instruct-v0.2

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "precommit")))
def test_sampling_precommit(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)

@pytest.mark.nightly
@pytest.mark.parametrize("model_id", get_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "nightly")))
def test_sampling_nightly(tmp_path, model_id):
    run_test_pipeline(tmp_path, model_id)
