# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from whowhatbench.text_agent_evaluator import TextAgentEvaluator


@pytest.mark.parametrize("sample_limit", [None, 1, 3, 5])
@pytest.mark.parametrize("predictions_from_csv", [False, True], ids=["generated", "saved"])
def test_score_applies_sample_limit_to_saved_reference(tmp_path, sample_limit, predictions_from_csv):
    prompts = ["first prompt", "second prompt", "third prompt"]
    records = [{"messages": [{"role": "user", "content": prompt}]} for prompt in prompts]
    gt_file = str(tmp_path / "reference.csv")

    def answer_prompt(_model, _tokenizer, record, **_kwargs):
        return record["messages"][0]["content"].upper()

    reference = TextAgentEvaluator(
        base_model=object(),
        gt_data=gt_file,
        test_data=records,
        metrics="",
        gen_answer_fn=answer_prompt,
    )
    reference.dump_gt(gt_file)

    evaluator = TextAgentEvaluator(gt_data=gt_file, test_data=records, num_samples=sample_limit, metrics="")
    target = gt_file if predictions_from_csv else object()
    evaluator.score(target, gen_answer_fn=answer_prompt, output_dir=str(tmp_path / "output"))

    expected_prompts = prompts[:sample_limit]
    expected_answers = [prompt.upper() for prompt in expected_prompts]
    assert evaluator.last_cmp["prompt"].tolist() == expected_prompts
    assert evaluator.last_cmp["source_model"].tolist() == expected_answers
    assert evaluator.last_cmp["optimized_model"].tolist() == expected_answers
