# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from validate_gguf_mmproj import case_passed, validate_coverage


def test_missing_or_duplicate_modalities_fail():
    args = SimpleNamespace(
        video_frames=0, multi_media=False, chat=False, audio=None, audio_boundaries=False, api_checks=False
    )
    for modalities in ([], ["text"], ["text", "image", "image"]):
        with pytest.raises(ValueError):
            validate_coverage(args, {"cases": [{"modality": name} for name in modalities]})
    validate_coverage(args, {"cases": [{"modality": name} for name in ("text", "image")]})


def test_audio_boundary_and_api_coverage_is_required():
    args = SimpleNamespace(
        video_frames=4, multi_media=True, chat=True, audio="audio.wav", audio_boundaries=True, api_checks=True
    )
    modalities = [
        "text",
        "image",
        "video",
        "multi_image",
        "audio",
        "mixed",
        "multi_audio",
        "audio_samples_479999",
        "audio_samples_480321",
        "image_chat",
        "audio_chat",
        "modern_image",
        "modern_image_chat",
        "modern_audio",
        "modern_audio_chat",
    ]
    checks = {
        name: {"passed": True} for name in ("beam_search", "cancel_and_reset", "modern_image_chat", "modern_audio_chat")
    }
    report = {"cases": [{"modality": name} for name in modalities], "api_checks": checks}
    validate_coverage(args, report)
    report["cases"].pop(8)
    with pytest.raises(ValueError):
        validate_coverage(args, report)
    report["cases"] = [{"modality": name} for name in modalities]
    del checks["cancel_and_reset"]
    with pytest.raises(ValueError):
        validate_coverage(args, report)


@pytest.mark.parametrize(
    "tokens,choices,passed", [([], [], False), ([1, 2], [1], False), ([1, 2], [9, 2], False), ([1, 2], [1, 2], True)]
)
def test_accuracy_uses_token_histories(tokens, choices, passed):
    assert (
        case_passed(
            {
                "tokens": tokens,
                "reference_choices_on_same_history": choices,
                "first_token_matches": True,
                "matching_choice_fraction": 1,
            }
        )
        is passed
    )


def test_lfm2_audio_requires_all_scenarios_and_api_checks():
    args = SimpleNamespace(family="lfm2_audio", api_checks=True)
    names = ["text", "audio", "multi_audio", "reversed_audio", "audio_chat", "audio_reset"]
    checks = {name: {"passed": True} for name in ["beam_search", "cancel_and_reset", "invalid_audio"]}
    report = {"cases": [{"modality": name} for name in names], "api_checks": checks}
    validate_coverage(args, report)
    report["cases"].append(report["cases"][0])
    with pytest.raises(ValueError):
        validate_coverage(args, report)
    report["cases"].pop()
    del checks["invalid_audio"]
    with pytest.raises(ValueError):
        validate_coverage(args, report)
