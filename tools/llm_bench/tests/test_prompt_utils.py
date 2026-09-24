# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the prompt handler (``llm_bench_utils.prompt_utils``).

These cover the parts every task now shares — prompt loading, ``--prompt_index``
filtering, ``--subsequent`` scheduling, chat-turn expansion and the
``prompt_repr`` rendering — without loading a model or touching the network.
"""

import json
import types
import wave

import numpy as np
import pytest
from PIL import Image

from llm_bench_utils.prompt_utils import (
    BenchChatPrompt,
    BenchPrompt,
    BenchPrompter,
    _expand_media_entries,
    _text_chat_turns,
    _vlm_chat_turns,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def make_image(path, size):
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return str(path)


def make_mask(path, size, filled_fraction):
    """Write an L-mode mask whose non-zero pixels cover *filled_fraction*."""
    w, h = size
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[: int(round(h * filled_fraction)), :] = 255
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


def make_wav(path, duration_s, sample_rate):
    frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(np.zeros(frames, dtype=np.int16).tobytes())
    return str(path)


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)


def make_args(task, prompt_file=None, **overrides):
    """A minimal args dict of the shape ``model_utils.analyze_args`` produces."""
    args = {
        "use_case": types.SimpleNamespace(task=task),
        "prompt_file": [prompt_file] if prompt_file else None,
        "prompt": None,
        "prompt_index": None,
        "subsequent": False,
        "batch_size": 1,
        "video_frames": None,
        "chat_iter": None,
    }
    args.update(overrides)
    return args


# --------------------------------------------------------------------------- #
# _expand_media_entries                                                        #
# --------------------------------------------------------------------------- #


def test_expand_media_entries_scalar_list_and_none():
    assert _expand_media_entries(None) == []
    assert _expand_media_entries("a.png") == ["a.png"]
    assert _expand_media_entries(["a.png", "b.png"]) == ["a.png", "b.png"]


def test_expand_media_entries_expands_a_directory(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    make_image(d / "02.png", (8, 8))
    make_image(d / "01.png", (8, 8))
    # Sorted, so a directory yields a stable order across platforms.
    assert [p.name for p in _expand_media_entries(str(d))] == ["01.png", "02.png"]


def test_expand_media_entries_passes_urls_through():
    url = "https://example.invalid/cat.png"
    assert _expand_media_entries(url) == [url]


# --------------------------------------------------------------------------- #
# BenchPrompt.__repr__                                                         #
# --------------------------------------------------------------------------- #


def test_repr_text_only_is_a_word_count():
    assert repr(BenchPrompt("one two three")) == "text:3w"


def test_repr_empty_prompt():
    assert repr(BenchPrompt({})) == "<empty prompt>"


def test_repr_single_image(tmp_path):
    media = make_image(tmp_path / "a.png", (512, 384))
    assert repr(BenchPrompt({"prompt": "a b", "media": media})) == "text:2w + image:512x384"


def test_repr_image_without_text(tmp_path):
    media = make_image(tmp_path / "a.png", (64, 32))
    assert repr(BenchPrompt({"media": media})) == "image:64x32"


def test_repr_collapses_equally_sized_images(tmp_path):
    """A list of same-size images folds into one token with a count."""
    files = [make_image(tmp_path / f"{i}.png", (256, 256)) for i in range(3)]
    assert repr(BenchPrompt({"prompt": "x", "media": files})) == "text:1w + image:256x256 x3"


def test_repr_joins_differently_sized_images(tmp_path):
    a = make_image(tmp_path / "a.png", (512, 512))
    b = make_image(tmp_path / "b.png", (640, 480))
    assert repr(BenchPrompt({"media": [a, b]})) == "image:512x512|640x480"


def test_repr_never_emits_a_comma(tmp_path):
    """prompt_repr lands in a CSV column, so it must not need quoting."""
    a = make_image(tmp_path / "a.png", (512, 512))
    b = make_image(tmp_path / "b.png", (640, 480))
    assert "," not in repr(BenchPrompt({"prompt": "hi", "media": [a, b]}))


def test_repr_marks_only_the_unprobeable_file(tmp_path):
    """One bad entry must not discard its siblings' dimensions."""
    good = make_image(tmp_path / "a.png", (128, 96))
    bad = str(tmp_path / "missing.png")
    assert repr(BenchPrompt({"media": [good, bad]})) == "image:128x96|?x?"


def test_repr_of_a_directory_of_images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    make_image(d / "01.png", (32, 32))
    make_image(d / "02.png", (32, 32))
    assert repr(BenchPrompt({"media": str(d)})) == "image:32x32 x2"


def test_repr_appends_mask_coverage(tmp_path):
    media = make_image(tmp_path / "a.png", (100, 100))
    mask = make_mask(tmp_path / "m.png", (100, 100), 0.25)
    assert repr(BenchPrompt({"media": media, "mask_image": mask})) == "image:100x100/25.0%"


def test_mask_is_read_with_a_url_capable_loader(tmp_path, monkeypatch):
    """A mask may be an HTTP(S) URL, which ``Image.open`` cannot resolve.

    Exercising a real URL would need the network, so this asserts the routing
    instead: the path must reach ``load_image``, not ``Image.open``.
    """
    import llm_bench_utils.prompt_utils as pu

    # Stand in for what the loader would fetch over the wire.
    fetched = Image.open(make_mask(tmp_path / "m.png", (10, 10), 1.0))
    seen = []

    def spy(path):
        seen.append(path)
        return fetched

    monkeypatch.setattr(pu, "load_image", spy)

    url = "https://example.invalid/mask.png"
    # Image.open() would raise on a URL and the fraction would come back None.
    assert BenchPrompt._get_mask_fraction(url) == pytest.approx(100.0)
    assert seen == [url]


def test_repr_audio(tmp_path):
    audio = make_wav(tmp_path / "a.wav", duration_s=2.0, sample_rate=8000)
    assert repr(BenchPrompt({"audio": audio})) == "audio:2.0s@8000Hz"


def test_repr_probes_image_and_audio_independently(tmp_path):
    """A VLM prompt may carry both; neither may suppress the other."""
    media = make_image(tmp_path / "a.png", (64, 64))
    audio = make_wav(tmp_path / "a.wav", duration_s=1.0, sample_rate=16000)
    got = repr(BenchPrompt({"prompt": "what is said", "media": media, "audio": audio}))
    assert got == "text:3w + image:64x64 + audio:1.0s@16000Hz"


def test_probe_runs_once_and_repr_is_stable(tmp_path):
    media = make_image(tmp_path / "a.png", (10, 20))
    prompt = BenchPrompt({"media": media})
    first = repr(prompt)
    # A caller may overwrite the probed size (super-resolution does); a later
    # repr() must not silently revert it to the on-disk value.
    prompt._image_sizes = [(1, 2)]
    assert first == "image:10x20"
    assert repr(prompt) == "image:1x2"


def test_bench_prompt_rejects_unsupported_input():
    with pytest.raises(TypeError):
        BenchPrompt(42)


# --------------------------------------------------------------------------- #
# stamp_repr                                                                   #
# --------------------------------------------------------------------------- #


def test_stamp_repr_tags_every_record_appended_since_the_index():
    prompt = BenchPrompt("one two")
    records = [{"pre": True}]
    start = len(records)
    records.extend([{}, {}])
    prompt.stamp_repr(records, start)
    assert "prompt_repr" not in records[0]
    assert [r["prompt_repr"] for r in records[1:]] == ["text:2w", "text:2w"]


def test_stamp_repr_tags_nothing_when_the_call_appended_nothing():
    """A skipped or failed generation must not relabel an earlier record."""
    prompt = BenchPrompt("one two")
    records = [{"prompt_repr": "text:9w"}]
    prompt.stamp_repr(records, len(records))
    assert records[0]["prompt_repr"] == "text:9w"


def test_chat_stamp_repr_matches_records_to_turns_by_prompt_idx():
    chat = BenchChatPrompt(["first turn here", "second"])
    records = [{"prompt_idx": 1}, {"prompt_idx": 0}]
    chat.stamp_repr(records, 0)
    assert records[0]["prompt_repr"] == "text:1w"
    assert records[1]["prompt_repr"] == "text:3w"


def test_repr_word_count_ignores_punctuation():
    assert repr(BenchPrompt("Hello , world .")) == "text:2w"


def test_stamp_repr_appends_each_records_text_tokens(tmp_path):
    media = make_image(tmp_path / "a.png", (64, 64))
    prompt = BenchPrompt({"prompt": "one two", "media": media})
    records = [{"input_text_tokens": 5}, {"input_text_tokens": ""}, {}]
    prompt.stamp_repr(records, 0)
    assert [r["prompt_repr"] for r in records] == [
        "text:2w/5t + image:64x64",
        "text:2w + image:64x64",
        "text:2w + image:64x64",
    ]
    # The log-time repr has no token count yet.
    assert repr(prompt) == "text:2w + image:64x64"


def test_chat_stamp_repr_uses_each_turns_own_text_tokens():
    chat = BenchChatPrompt(["first turn here", "second"])
    records = [{"prompt_idx": 0, "input_text_tokens": 4}, {"prompt_idx": 1, "input_text_tokens": 1}]
    chat.stamp_repr(records, 0)
    assert [r["prompt_repr"] for r in records] == ["text:3w/4t", "text:1w/1t"]


def test_chat_stamp_repr_skips_records_with_an_unknown_turn_index():
    chat = BenchChatPrompt(["only turn"])
    records = [{"prompt_idx": 7}]
    chat.stamp_repr(records, 0)
    assert "prompt_repr" not in records[0]


# --------------------------------------------------------------------------- #
# BenchChatPrompt                                                              #
# --------------------------------------------------------------------------- #


def test_chat_prompt_exposes_turn_texts_and_repr():
    chat = BenchChatPrompt(["hello there", "bye"])
    assert chat.prompts == ["hello there", "bye"]
    assert repr(chat) == "chat:2t[text:2w | text:1w]"


def test_chat_prompt_rejects_an_empty_conversation():
    with pytest.raises(RuntimeError, match="prompts is empty"):
        BenchChatPrompt([])


def test_chat_prompt_rejects_a_non_benchprompt_turn():
    chat = BenchChatPrompt(["a"])
    with pytest.raises(TypeError):
        chat.append("raw string")


# --------------------------------------------------------------------------- #
# Chat turn expansion                                                          #
# --------------------------------------------------------------------------- #


def test_text_chat_turns_uses_a_list_verbatim_and_ignores_chat_iter():
    assert _text_chat_turns(["a", "b"], {"chat_iter": 5}) == ["a", "b"]


def test_text_chat_turns_replicates_a_scalar_by_chat_iter():
    assert _text_chat_turns("a", {"chat_iter": 3}) == ["a", "a", "a"]


def test_text_chat_turns_rejects_a_scalar_without_chat_iter():
    with pytest.raises(RuntimeError, match="Chat mode"):
        _text_chat_turns("a", {})


def test_vlm_chat_turns_wraps_a_single_dict():
    entry = {"prompt": "hi"}
    assert _vlm_chat_turns(entry, {}) == [entry]


def test_vlm_chat_turns_replicates_a_one_turn_chat():
    assert _vlm_chat_turns({"prompt": "hi"}, {"chat_iter": 2}) == [{"prompt": "hi"}] * 2


def test_vlm_chat_turns_keeps_a_multi_turn_list_despite_chat_iter():
    turns = [{"prompt": "a"}, {"prompt": "b"}]
    assert _vlm_chat_turns(turns, {"chat_iter": 9}) == turns


# --------------------------------------------------------------------------- #
# BenchPrompter: loading and selection                                         #
# --------------------------------------------------------------------------- #


def test_prompter_loads_a_text_jsonl(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "first"}, {"prompt": "second"}])
    prompter = BenchPrompter(make_args("text_gen", pf))
    assert [p["prompt"] for p in prompter] == ["first", "second"]
    assert prompter.active_indices == [0, 1]


def test_prompter_accepts_prebuilt_prompts_without_a_file():
    prompter = BenchPrompter(make_args("visual_text_gen"), prompts=[{"prompt": "a"}, {"prompt": "b"}])
    assert [p["prompt"] for p in prompter] == ["a", "b"]


def test_prompt_index_selects_a_subset_and_keeps_original_indices(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": f"p{i}"} for i in range(4)])
    prompter = BenchPrompter(make_args("text_gen", pf, prompt_index=[2, 0]))
    assert prompter.active_indices == [2, 0]
    assert [p["prompt"] for p in prompter.active_items] == ["p2", "p0"]


def test_prompt_index_ignores_out_of_range_entries(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "only"}])
    prompter = BenchPrompter(make_args("text_gen", pf, prompt_index=[0, 5, -1]))
    assert prompter.active_indices == [0]


def test_require_active_raises_when_the_selection_is_empty(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "only"}])
    prompter = BenchPrompter(make_args("text_gen", pf, prompt_index=[9]))
    assert prompter.active_indices == []
    with pytest.raises(RuntimeError, match="prompts is empty"):
        prompter.require_active()


def test_prompter_rejects_a_non_prompt_object(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "a"}])
    prompter = BenchPrompter(make_args("text_gen", pf))
    with pytest.raises(TypeError):
        prompter.append({"prompt": "raw dict"})


# --------------------------------------------------------------------------- #
# BenchPrompter: per-task specs                                                #
# --------------------------------------------------------------------------- #


def test_speech_to_text_routes_media_to_the_audio_key(tmp_path):
    """The rename is what makes probe() report a duration instead of a filename."""
    audio = make_wav(tmp_path / "a.wav", duration_s=1.0, sample_rate=16000)
    pf = write_jsonl(tmp_path / "p.jsonl", [{"media": audio}])
    prompter = BenchPrompter(make_args("speech_to_text", pf))
    assert "media" not in prompter[0]
    assert prompter[0]["audio"] == audio
    assert repr(prompter[0]) == "audio:1.0s@16000Hz"


def test_super_resolution_routes_prompt_to_the_media_key(tmp_path):
    image = make_image(tmp_path / "low.png", (64, 48))
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": image}])
    prompter = BenchPrompter(make_args("ldm_super_resolution", pf))
    assert prompter[0]["media"] == image
    assert repr(prompter[0]) == "image:64x48"


def test_media_paths_resolve_relative_to_the_prompt_file(tmp_path):
    sub = tmp_path / "assets"
    sub.mkdir()
    make_image(sub / "a.png", (16, 16))
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "hi", "media": "./assets/a.png"}])
    prompter = BenchPrompter(make_args("visual_text_gen", pf))
    # Resolved against the prompt file's directory, not the process CWD.
    assert repr(prompter[0]) == "text:1w + image:16x16"


def test_text_chat_task_builds_chat_prompts(tmp_path):
    """A text chat turn list is the *value* of the entry's "prompt" key."""
    pf = write_jsonl(tmp_path / "chat.jsonl", [{"prompt": ["turn one", "turn two"]}])
    prompter = BenchPrompter(make_args("text_gen_chat", pf))
    assert isinstance(prompter[0], BenchChatPrompt)
    assert prompter[0].prompts == ["turn one", "turn two"]


def test_text_chat_replicates_a_scalar_prompt_chat_iter_times(tmp_path):
    pf = write_jsonl(tmp_path / "chat.jsonl", [{"prompt": "just one"}])
    prompter = BenchPrompter(make_args("text_gen_chat", pf, chat_iter=3))
    assert prompter[0].prompts == ["just one"] * 3


def test_vlm_chat_task_builds_chat_prompts_from_a_json_array(tmp_path):
    """The VLM chat format differs: one JSONL line is an array of turn dicts."""
    make_image(tmp_path / "a.png", (32, 32))
    pf = tmp_path / "vlm_chat.jsonl"
    with open(pf, "w", encoding="utf-8") as f:
        f.write(json.dumps([{"prompt": "what is this", "media": ["./a.png"]}, {"prompt": "and now"}]) + "\n")
    prompter = BenchPrompter(make_args("visual_text_gen_chat", str(pf)))
    chat = prompter[0]
    assert isinstance(chat, BenchChatPrompt)
    assert chat.prompts == ["what is this", "and now"]
    # A list-valued 'media' must still be described, not reported as image:?x?.
    assert repr(chat[0]) == "text:3w + image:32x32"


def test_chat_prefix_uses_c_and_plain_prompts_use_p(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": "a"}])
    plain = BenchPrompter(make_args("text_gen", pf))
    assert plain.get_prefix(0, 3) == "[warm-up][P3]"
    assert plain.get_prefix(2, 3) == "[2][P3]"

    chat_pf = write_jsonl(tmp_path / "c.jsonl", [{"prompt": ["a", "b"]}])
    chat = BenchPrompter(make_args("text_gen_chat", chat_pf))
    assert chat.get_prefix(0, 1) == "[warm-up][C1]"


# --------------------------------------------------------------------------- #
# BenchPrompter.iter_schedule                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_prompts(tmp_path):
    return write_jsonl(tmp_path / "p.jsonl", [{"prompt": "p0"}, {"prompt": "p1"}])


def test_schedule_default_is_iteration_major(two_prompts):
    """Without --subsequent every prompt runs once per iteration."""
    prompter = BenchPrompter(make_args("text_gen", two_prompts))
    # (iteration, prompt): warm-up, then 1, then 2 — each visiting P0 before P1.
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert [(n, i) for n, i, _ in prompter.iter_schedule(2)] == expected


def test_schedule_subsequent_is_prompt_major(two_prompts):
    """With --subsequent all iterations of one prompt run before the next."""
    prompter = BenchPrompter(make_args("text_gen", two_prompts, subsequent=True))
    # (iteration, prompt): every iteration of P0, then every iteration of P1.
    expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    assert [(n, i) for n, i, _ in prompter.iter_schedule(2)] == expected


def test_schedule_includes_the_warm_up_iteration(two_prompts):
    prompter = BenchPrompter(make_args("text_gen", two_prompts))
    nums = {n for n, _, _ in prompter.iter_schedule(0)}
    assert nums == {0}, "num_iters=0 must still yield the warm-up"


def test_both_schedules_cover_the_same_pairs(two_prompts):
    interleaved = BenchPrompter(make_args("text_gen", two_prompts))
    subsequent = BenchPrompter(make_args("text_gen", two_prompts, subsequent=True))
    pairs = [(n, i) for n, i, _ in interleaved.iter_schedule(3)]
    assert sorted(pairs) == sorted((n, i) for n, i, _ in subsequent.iter_schedule(3))


def test_schedule_honours_prompt_index(tmp_path):
    pf = write_jsonl(tmp_path / "p.jsonl", [{"prompt": f"p{i}"} for i in range(3)])
    prompter = BenchPrompter(make_args("text_gen", pf, prompt_index=[2]))
    assert [(n, i) for n, i, _ in prompter.iter_schedule(1)] == [(0, 2), (1, 2)]


def test_schedule_yields_the_same_object_for_every_iteration(two_prompts):
    """Pipelines cache probe results on the prompt, so it must be reused."""
    prompter = BenchPrompter(make_args("text_gen", two_prompts))
    seen = {}
    for num, p_idx, prompt in prompter.iter_schedule(2):
        seen.setdefault(p_idx, prompt)
        assert seen[p_idx] is prompt
