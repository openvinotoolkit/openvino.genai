# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import numpy as np
from PIL import Image
import logging as log
import librosa
from transformers.image_utils import load_image
from .model_utils import get_param_from_file, resolve_media_file_path
from .parse_json_data import (
    parse_text_json_data,
    parse_vlm_json_data,
    parse_image_json_data,
    parse_video_json_data,
    parse_speech_json_data,
)
import llm_bench_utils.metrics_print as metrics_print
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import openvino as ov
import math
import cv2


def print_video_frames_number_and_convert_to_tensor(func):
    def inner(video_path, decim_frames, genai_flag):
        log.info(f"Input video file: {video_path}")
        if decim_frames is not None:
            log.info(f"Requested to reduce into {decim_frames} frames")
        out_frames = func(video_path, decim_frames)
        log.info(f"Final frames number: {len(out_frames)}")
        log.info(f"First frame shape: {out_frames[0].shape}")
        log.info(f"First frame dtype: {out_frames[0].dtype}")
        if genai_flag:
            return ov.Tensor(out_frames)
        return np.array(out_frames)
    return inner


@print_video_frames_number_and_convert_to_tensor
def make_video_tensor(video_path, decim_frames=None):
    assert os.path.exists(video_path), f"no input video file: {video_path}"
    cap = cv2.VideoCapture(video_path)

    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        np_img_array = np.array(pil_image)
        log.debug(f"Video shape: {np_img_array.shape}")
        log.debug(f"Video dtype: {np_img_array.dtype}")
        output_frames.append(np_img_array)

    if not decim_frames:
        log.info(f"Video decim: no-set: {decim_frames}: skip")
        return output_frames

    # decimation procedure
    # decim_frames is required max frame number if positive
    # or decimation factor if negative
    # e.g. if input frames number is 100 and decim_fames = 5:
    #         then number of processed frames are: 0, 20, 40, 60, 80
    #      if input frames number is 100 and decim_fames = -5:
    #         then number of processed frames are: 0, 5, 10, 15, 20, ...

    decim_frames = int(decim_frames)
    if decim_frames > 0:
        if len(output_frames) <= decim_frames:
            log.info(f"Video decim: too short to decim: crop: {decim_frames}")
            return list(output_frames[:decim_frames])
        decim_factor_f = float(len(output_frames)) / decim_frames
        decim_factor = int(math.ceil(decim_factor_f))
    else:
        decim_factor = -decim_frames
    log.info(f"Video decim factor: {decim_factor}")
    if decim_factor >= 2:
        return list(output_frames[::decim_factor])
    log.info("Video decim: too large decim factor: skip")
    return output_frames


def load_image_genai(image_path):
    pil_image = load_image(image_path)
    image_data = np.array(pil_image)[None]
    return ov.Tensor(image_data)


def load_audio_genai(audio_path):
    # GenAI pipelines expect a 16 kHz mono waveform tensor.
    audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
    return ov.Tensor(audio_data.astype(np.float32))


def load_audio_optimum(audio_path):
    # Optimum processors resample internally; return (array, native_rate).
    return librosa.load(audio_path, sr=None, mono=True)


def _normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _expand_media_entries(value):
    """Expand a media key's value into the flat list of files it refers to.

    A media key may hold a single path, a list of paths, or a directory (see
    :func:`extract_prompt_data`, which loads directories entry by entry).
    Mirrors that expansion so ``BenchPrompt.probe()`` describes exactly the
    files the pipeline will consume.  HTTP(S) URLs are passed through
    untouched — ``Path.is_dir()`` is False for them.
    """
    entries = []
    for item in _normalize_to_list(value):
        try:
            path = Path(item)
            is_dir = path.is_dir()
        except (TypeError, ValueError, OSError):
            is_dir = False
        if is_dir:
            entries.extend(sorted(path.iterdir()))
        else:
            entries.append(item)
    return entries


def extract_prompt_data(inputs, required_frames, genai_flag):
    """
    Unpack a list of prompt dicts into separate
    ``(prompts, images, videos, audios)`` lists, loading each media file on
    the fly.

    Parameters
    ----------
    inputs : list[dict] | dict
        One or more prompt dicts (each may have ``'prompt'``, ``'media'``,
        ``'video'`` and/or ``'audio'`` keys). Every media key accepts either
        a single path or a list of paths.
    required_frames : int | None
        Frame decimation target forwarded to :func:`make_video_tensor`.
    genai_flag : bool
        When ``True`` tensors are returned in the GenAI representation
        (``ov.Tensor``); when ``False`` in the optimum one.

    Returns
    -------
    prompts : list[str]
    images  : list
    videos  : list
    audios  : list
    """
    prompts, images, videos, audios = [], [], [], []
    if not isinstance(inputs, (list, tuple, set)):
        inputs = [inputs]
    for input_data in inputs:
        for video_item in _normalize_to_list(input_data.get("video")):
            entry = Path(video_item)
            if entry.is_dir():
                for filename in sorted(entry.iterdir()):
                    video_tensor = make_video_tensor(filename, required_frames, genai_flag)
                    videos.append(video_tensor)
            else:
                video_tensor = make_video_tensor(entry, required_frames, genai_flag)
                videos.append(video_tensor)

        # Load as PIL first so the BenchPrompt repr metadata can be refreshed
        # with the dimensions of the images that were really fetched, which
        # supersede the ones probe() guessed before inference.
        loaded_sizes = []
        for media_item in _expand_media_entries(input_data.get("media")):
            pil_img = load_image(str(media_item))
            loaded_sizes.append(pil_img.size)
            images.append(ov.Tensor(np.array(pil_img)[None]) if genai_flag else pil_img)
        if loaded_sizes and isinstance(input_data, BenchPrompt):
            input_data._image_sizes = loaded_sizes

        func_load_audio = load_audio_genai if genai_flag else load_audio_optimum
        for audio_item in _normalize_to_list(input_data.get("audio")):
            audios.append(func_load_audio(str(audio_item)))

        # 'prompt' is optional (e.g. pure image / audio entries); default to an
        # empty string so the returned prompts list stays aligned with inputs
        # and never raises KeyError.
        prompts.append(input_data.get("prompt", ""))
    return prompts, images, videos, audios


# ---------------------------------------------------------------------------
# BenchPrompt  &  BenchPrompter
# ---------------------------------------------------------------------------#


class BenchPrompt(dict):
    """
    Handler for a single multimedia prompt.

    Inherits from ``dict`` and stores prompt data under well-known keys:

        'prompt'          - text prompt string (optional – not all pipelines
                            require a text prompt, e.g. audio or img2img)
        'media'           - path to an image file
        'mask_image'      - path to a mask image (inpainting tasks)
        'video'           - path to a video file or directory of video frames
        'audio'           - path to an audio file
        'negative_prompt' - negative text prompt (image / video generation)

    Media sizes and shapes are probed **lazily** on the first call to
    ``__repr__`` (or to ``probe()`` explicitly). For video, optional
    decimation is applied via ``args['video_frames']`` using
    ``make_video_tensor``.

    Parameters
    ----------
    data : str | dict
        A single prompt entry. A plain ``str`` is stored as the text
        prompt. A ``dict`` may contain any combination of the keys listed
        above; unknown keys are silently ignored.
    args : dict, optional
        Global benchmark args dict (e.g. as returned by
        ``model_utils.analyze_args``). Used for:
            - ``args['video_frames']`` - frame decimation target for video
    """

    #: Keys recognised and stored by BenchPrompt
    MEDIA_KEYS = ("prompt", "media", "mask_image", "video", "audio", "negative_prompt")

    def __init__(self, data, args=None):
        dict.__init__(self)
        self._args = args or {}
        # Lazily filled by probe().  A media key may name several files (a
        # list of paths, or a directory), so each holds one entry per file;
        # an entry is None when that file could not be probed.
        self._image_sizes = []  # list[(width, height) | None]
        self._video_shapes = []  # list[(frames, height, width) | None]
        self._audio_infos = []  # list[(duration_sec, sample_rate) | None]
        self._mask_fraction = None  # float | None  cached mask coverage %
        self._probed = False
        self._load(data)

    # ------------------------------------------------------------------ #
    # Loading & structural validation                                      #
    # ------------------------------------------------------------------ #

    def _load(self, data):
        """Populate the dict from *data* and run cheap structural checks.

        The ``'prompt'`` key is **optional**: pipelines such as audio
        transcription, image-to-image and super-resolution do not require
        a text prompt.
        """
        if isinstance(data, str):
            self["prompt"] = data
        elif isinstance(data, dict):
            # Store all keys so that task-specific extra parameters
            # (e.g. language/timestamp for speech-to-text, or
            # steps/width/height for image super-resolution) are
            # preserved and accessible via normal dict lookups.
            self.update(data)
        else:
            raise TypeError(f"BenchPrompt: unsupported data type {type(data)!r}. Expected str or dict.")

    # ------------------------------------------------------------------ #
    # Lazy media probing                                                   #
    # ------------------------------------------------------------------ #

    def probe(self):
        """
        Probe all media files to fill size / shape metadata.

        Called automatically by ``__repr__``. Safe to call multiple times: it
        runs once and caches, so a repeated ``repr()`` costs nothing and the
        mask fraction is computed at most once per instance.

        ``'media'`` always holds an image. ``speech_to_text`` routes its audio
        path to the ``'audio'`` key on both the JSONL and CLI branches (see
        ``rename`` / ``nonjson_wrap`` in ``_PROMPT_SPECS``), and
        ``ldm_super_resolution`` puts its low-res input image there. A VLM
        prompt may carry an image and audio at once — doc/PROMPT.md section 5
        lists the media keys as independent — so each is probed separately.

        Every media key accepts a single path, a list of paths or a directory,
        so each is expanded (see :func:`_expand_media_entries`) and probed file
        by file.
        """
        if self._probed:
            return
        self._probed = True

        self._image_sizes = [self._get_image_size(path) for path in _expand_media_entries(self.get("media"))]

        decim = self._args.get("video_frames")
        self._video_shapes = [self._get_video_shape(path, decim) for path in _expand_media_entries(self.get("video"))]

        self._audio_infos = [self._get_audio_info(path) for path in _expand_media_entries(self.get("audio"))]

        if self.get("mask_image"):
            self._mask_fraction = self._get_mask_fraction(self["mask_image"])

    # ------------------------------------------------------------------ #
    # Static media helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_image_size(path):
        """Return ``(width, height)`` or ``None`` on failure.

        Uses :func:`~transformers.image_utils.load_image`, which resolves both
        local filesystem paths and HTTP(S) URLs. Prompt files may point at
        either, so ``Image.open()`` is not sufficient here.
        """
        try:
            return load_image(str(path)).size
        except Exception as exc:
            log.warning(f"BenchPrompt: cannot probe image '{path}': {exc}")
            return None

    @staticmethod
    def _get_video_shape(path, decim_frames=None):
        """Return ``(frames, height, width)`` or ``None`` on failure.

        Reads cv2 container metadata rather than decoding every frame via
        ``make_video_tensor``: this runs before the benchmark loop, so it must
        not cost real I/O and CPU.

        The frame count is therefore the raw container value and does NOT
        account for the decimation applied during the run. It is displayed for
        information only.
        """
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise IOError(f"cv2 could not open video: {path}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return (n, h, w)
        except Exception as exc:
            log.warning(f"BenchPrompt: cannot probe video '{path}': {exc}")
            return None

    @staticmethod
    def _get_audio_info(path):
        """Return ``(duration_sec, sample_rate)`` or ``None`` on failure."""
        try:
            import librosa

            sr = librosa.get_samplerate(path)
            dur = librosa.get_duration(path=path)
            return (dur, sr)
        except Exception as exc:
            log.warning(f"BenchPrompt: cannot probe audio '{path}': {exc}")
            return None

    @staticmethod
    def _get_mask_fraction(mask_path):
        """Return the percentage of non-zero pixels in *mask_path* or ``None``.

        Loaded with :func:`~transformers.image_utils.load_image` for the same
        reason :meth:`_get_image_size` is: a mask may be given as an HTTP(S)
        URL, which ``Image.open()`` cannot resolve.
        """
        try:
            arr = np.array(load_image(str(mask_path)).convert("L"))
            return 100.0 * float(np.count_nonzero(arr)) / arr.size
        except Exception as exc:
            log.warning(f"BenchPrompt: cannot probe mask image '{mask_path}': {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _join_media(kind, dims):
        """Render one modality's per-file dimension strings as a single token.

        A media key may name several files, so *dims* holds one string per
        file. They are collapsed to ``kind:<dims> x<N>`` when every file has
        the same dimensions, and joined with ``|`` when they differ — never
        with ``,``, which would need quoting in the CSV report.
        """
        if not dims:
            return None
        if len(dims) == 1:
            return f"{kind}:{dims[0]}"
        if len(set(dims)) == 1:
            return f"{kind}:{dims[0]} x{len(dims)}"
        return f"{kind}:" + "|".join(dims)

    def __repr__(self):
        """
        Human-readable description of the prompt and its input **sizes**.

        Format (``+`` separates modalities):

            text:7w
            text:7w + image:512x512
            text:7w + video:640x480@30f
            text:7w + audio:30.0s@44100Hz
            text:7w + image:512x512/35.2%
            text:7w + image:1024x768 + video:640x480@16f
            audio:30.0s@44100Hz      <- no text prompt

        A media key naming several files (a list of paths, or a directory) is
        rendered as one token covering all of them — ``image:512x512 x3`` when
        they share dimensions, ``image:512x512|640x480`` when they differ.

        This is a pre-tokenization *size* summary: the text is shown as a
        whitespace word count (``w`` suffix), images/videos as pixel
        dimensions, audio as duration. The tokenized length is reported
        separately by the pipelines as ``input_size``.
        """
        self.probe()
        parts = []

        # ---- text (optional) ----
        # prompt_repr describes input *sizes*; the text size is its word count.
        if self.get("prompt"):
            word_count = len(self["prompt"].split())
            parts.append(f"text:{word_count}w")

        # ---- image (optionally decorated with mask coverage fraction) ----
        # Mirrors probe(): 'media' is always an image, and is rendered
        # independently of any audio the prompt also carries.
        image_dims = [f"{size[0]}x{size[1]}" if size else "?x?" for size in self._image_sizes]
        image_part = self._join_media("image", image_dims)
        if image_part is not None:
            # The mask covers the whole 'media' key, so its fraction is
            # appended once to the joined token rather than per file.
            frac = self._mask_fraction  # pre-computed in probe()
            if self.get("mask_image") and frac is not None:
                image_part += f"/{frac:.1f}%"
            parts.append(image_part)

        # ---- video ----
        video_dims = [f"{s[2]}x{s[1]}@{s[0]}f" if s else "?x?@?f" for s in self._video_shapes]
        video_part = self._join_media("video", video_dims)
        if video_part is not None:
            parts.append(video_part)

        # ---- audio ----
        audio_dims = [f"{info[0]:.1f}s@{info[1]}Hz" if info else "?s@?Hz" for info in self._audio_infos]
        audio_part = self._join_media("audio", audio_dims)
        if audio_part is not None:
            parts.append(audio_part)

        return " + ".join(parts) if parts else "<empty prompt>"

    def introduce_in_stdout(self, num, prefix):
        if num == 0:
            if self.get("prompt"):
                metrics_print.print_unicode(
                    f"{prefix} Input text: {self['prompt']}",
                    f"{prefix} Unable print input text",
                    max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                )
        prompt_repr = repr(self)
        log.info(f"{prefix} Prompt: {prompt_repr}")

    def stamp_repr(self, iter_data_list, start_index):
        """Tag ``prompt_repr`` onto the records appended since *start_index*.

        Callers capture ``len(iter_data_list)`` **before** invoking the
        generation function, then pass that length here afterwards. This tags
        exactly the record(s) produced for the current (iteration, prompt)
        pair:

        * zero records appended (e.g. a skipped / errored call) -> nothing is
          tagged, so a previous iteration's record is never mislabelled;
        * multiple records appended (e.g. per-batch) -> all of them are tagged.

        Slicing rather than indexing ``iter_data_list[-1]`` is what makes both
        of those cases correct: a positional assignment mis-attributes the
        value whenever a call appends other than exactly one record.
        """
        self._stamp_records(iter_data_list[start_index:])

    def _stamp_records(self, records):
        """Write ``prompt_repr`` onto an explicit record list.

        Split out of :meth:`stamp_repr` so :class:`BenchChatPrompt` can stamp a
        single turn's record without re-slicing ``iter_data_list``.
        """
        prompt_repr = repr(self)
        for record in records:
            record["prompt_repr"] = prompt_repr


class BenchChatPrompt(list):
    """A multi-turn chat prompt: an ordered list of :class:`BenchPrompt` turns.

    Built by :class:`BenchPrompter` for the chat tasks (``text_gen_chat``,
    ``visual_text_gen_chat``) in place of a single :class:`BenchPrompt`, so the
    chat pipelines keep iterating with ``for turn_idx, turn in enumerate(chat)``
    and reading ``len(chat)``.

    Exposes the same ``introduce_in_stdout`` / ``stamp_repr`` pair as
    :class:`BenchPrompt`, so both kinds of prompt are interchangeable inside
    :meth:`BenchPrompter.iter_schedule`.

    Parameters
    ----------
    turns : list
        One entry per user turn — a ``str`` (text chat) or a ``dict`` (VLM
        chat), as produced by the spec's ``chat_turns`` expander.
    args : dict, optional
        Global benchmark args, forwarded to each turn's :class:`BenchPrompt`.
    """

    def __init__(self, turns, args=None):
        list.__init__(self)
        self._args = args or {}
        for turn in turns:
            self.append(BenchPrompt(turn, args))
        if not self:
            # A chat with no turns cannot be benchmarked.
            raise RuntimeError("==Failure prompts is empty ==")

    def append(self, turn):
        """Only :class:`BenchPrompt` turns may be appended."""
        if not isinstance(turn, BenchPrompt):
            raise TypeError(f"BenchChatPrompt only accepts BenchPrompt turns, got {type(turn)!r}")
        super().append(turn)

    @property
    def prompts(self):
        """Text of every turn, in order (used for the input-data dumps)."""
        return [turn.get("prompt", "") for turn in self]

    def __repr__(self):
        if not self:
            return "<empty chat>"
        return f"chat:{len(self)}t[" + " | ".join(repr(turn) for turn in self) + "]"

    def introduce_in_stdout(self, num, prefix):
        if num == 0:
            for turn_idx, turn in enumerate(self):
                if turn.get("prompt"):
                    metrics_print.print_unicode(
                        f"{prefix}[P{turn_idx}] Input text: {turn['prompt']}",
                        f"{prefix}[P{turn_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
        log.info(f"{prefix} Prompt: {repr(self)}")

    def stamp_repr(self, iter_data_list, start_index):
        """Tag each new record with **its own turn's** repr.

        The chat pipelines append exactly one record per turn and set that
        record's ``prompt_idx`` to the turn index, so records are matched back
        to turns by ``prompt_idx`` rather than by position. A record whose
        ``prompt_idx`` is not a valid turn index is left untouched, so a future
        change appending a different number of records degrades to "not
        stamped" rather than "mislabelled".
        """
        for record in iter_data_list[start_index:]:
            turn_idx = record.get("prompt_idx")
            if isinstance(turn_idx, int) and 0 <= turn_idx < len(self):
                self[turn_idx]._stamp_records([record])


# ---------------------------------------------------------------------------
# Per-task prompt specification
# ---------------------------------------------------------------------------#
#
# Each task type maps to a declarative _PromptSpec instead of a bespoke branch
# in BenchPrompter._load_prompts().  This keeps the task-specific knowledge
# (which CLI/JSONL key holds the prompt, how to parse the JSONL entries, which
# keys are media paths to resolve, and any key renames) in one readable table.


@dataclass(frozen=True)
class _PromptSpec:
    #: Key(s) passed to get_param_from_file().  A callable receives ``args``
    #: and returns the key(s) — used by image_gen whose key set is dynamic.
    input_key: Union[str, list, Callable]
    #: Parser applied to raw JSONL entries.
    parse: Callable
    #: Entry keys whose values are media file paths to resolve against the
    #: prompt file (JSONL branch only).
    path_keys: tuple = ()
    #: ``{src: dst}`` key renames applied AFTER path resolution (JSONL branch).
    #: e.g. speech_to_text stores the audio path under 'audio' (not 'media')
    #: so BenchPrompt.probe() routes it through _get_audio_info().
    rename: dict = field(default_factory=dict)
    #: Prompt file (relative to the llm_bench root) used when the task got
    #: neither a CLI media/prompt value nor an explicit --prompt_file.  Used by
    #: speech_to_text, which ships a default audio sample.
    default_prompt_file: Optional[str] = None
    #: CLI arg keys that, when set, suppress ``default_prompt_file``.
    default_prompt_file_unless: tuple = ()
    #: Wrapper applied to each raw value in the NON-JSON (CLI) branch.  When
    #: None the raw values are used as-is.  Used by speech_to_text to wrap a
    #: bare audio path into ``{"audio": path}``.
    nonjson_wrap: Optional[Callable] = None
    #: Marks a CHAT task.  ``fn(entry, args) -> list`` expands one parsed entry
    #: into its list of user turns; each entry then becomes a
    #: :class:`BenchChatPrompt` instead of a :class:`BenchPrompt`.  The two chat
    #: tasks disagree on how a non-list entry and ``--chat_iter`` interact, so
    #: each supplies its own expander rather than BenchPrompter branching on the
    #: task name.
    chat_turns: Optional[Callable] = None
    #: Letter used by :meth:`BenchPrompter.get_prefix` in log prefixes: ``"P"``
    #: for prompts, ``"C"`` for chats (matching metrics_print's chat_mode alias).
    prefix_alias: str = "P"


def _text_chat_turns(entry, args):
    """Expand one ``text_gen_chat`` entry into its user turns.

    A JSONL ``{"prompt": ["turn1", "turn2"]}`` entry is already a turn list and
    is used verbatim (``--chat_iter`` is ignored). A scalar prompt is replicated
    ``--chat_iter`` times; without ``--chat_iter`` it cannot form a chat at all.
    See doc/PROMPT.md section 1.
    """
    if isinstance(entry, list):
        return entry
    if args.get("chat_iter"):
        return [entry] * args["chat_iter"]
    raise RuntimeError("Chat mode can't be started due to incompatible input prompts")


def _vlm_chat_turns(entry, args):
    """Expand one ``visual_text_gen_chat`` entry into its user turns.

    A JSONL line that is a JSON array is already a turn list; a single turn dict
    — a bare JSONL object, or the one assembled from ``--media`` / ``--prompt``
    — becomes a one-turn chat. ``--chat_iter`` replicates a one-turn chat and is
    ignored (with a warning) for a multi-turn one. See doc/PROMPT.md section 5.
    """
    turns = entry if isinstance(entry, list) else [entry]
    if args.get("chat_iter"):
        if len(turns) == 1:
            return turns * args["chat_iter"]
        log.warning(
            f"Chat mode is enabled and chat_iter is {args['chat_iter']}, but input data is set as list."
            "`chat_iter` will be ignored. Chat will be run based on the provided list."
        )
    return turns


def _image_gen_input_key(args):
    """Resolve the dynamic input key set for the image_gen task.

    Mirrors the original branch logic: inpainting needs media+mask+prompt,
    img2img needs media+prompt, plain text-to-image needs only prompt.
    """
    use_case = args.get("use_case")
    if use_case and hasattr(use_case, "TASK"):
        inpainting_name = use_case.TASK.get("inpainting", {}).get("name")
        img2img_name = use_case.TASK.get("img2img", {}).get("name")
        if args.get("task") == inpainting_name or (
            (args.get("media") or args.get("images")) and args.get("mask_image")
        ):
            return ["media", "mask_image", "prompt"]
        if args.get("task") == img2img_name or args.get("media") or args.get("images"):
            return ["media", "prompt"]
    return ["prompt"]


def _video_gen_input_key(args):
    """Resolve the dynamic input key set for the video_gen task.

    Image-to-video additionally takes the source image under 'media'; plain
    text-to-video needs only the prompt pair.
    """
    if args.get("task") == "image-to-video":
        return ["media", "prompt", "negative_prompt"]
    return ["prompt", "negative_prompt"]


_PROMPT_SPECS = {
    "visual_text_gen": _PromptSpec(
        ["video", "media", "prompt"], parse_vlm_json_data, path_keys=("media", "video", "audio")
    ),
    # Chat tasks: one entry is a whole conversation, so it is expanded into
    # turns and wrapped in a BenchChatPrompt. Prefixes read [warm-up][C0].
    "text_gen_chat": _PromptSpec(
        "prompt",
        parse_text_json_data,
        chat_turns=_text_chat_turns,
        prefix_alias="C",
    ),
    "visual_text_gen_chat": _PromptSpec(
        ["video", "media", "prompt"],
        parse_vlm_json_data,
        path_keys=("media", "video", "audio"),
        chat_turns=_vlm_chat_turns,
        prefix_alias="C",
    ),
    # Embedding models may be multimodal (Qwen3-VL-Embedding), so they read the
    # VLM key set — but unlike text generation the text prompt is optional, an
    # entry may carry media only.
    "text_embed": _PromptSpec(
        ["video", "media", "prompt"],
        lambda data: parse_vlm_json_data(data, optional_prompt=True),
        path_keys=("media", "video"),
    ),
    "image_gen": _PromptSpec(_image_gen_input_key, parse_image_json_data, path_keys=("media", "mask_image")),
    "video_gen": _PromptSpec(_video_gen_input_key, parse_video_json_data, path_keys=("media",)),
    "speech_to_text": _PromptSpec(
        "media",
        parse_speech_json_data,
        path_keys=("media",),
        rename={"media": "audio"},
        default_prompt_file="prompts/speech_to_text_default.jsonl",
        default_prompt_file_unless=("media", "prompt_file"),
        nonjson_wrap=lambda item: {"audio": item},
    ),
    "ldm_super_resolution": _PromptSpec(
        "prompt",
        parse_image_json_data,
        path_keys=("prompt",),
        # Route the input image path to the 'media' key so BenchPrompt.__repr__
        # renders it as an image (image:WxH) instead of a text word count.
        rename={"prompt": "media"},
        nonjson_wrap=lambda item: {"media": item},
    ),
}

# text_gen, code_gen, text_embed, text2speech, text_rerank, ...
_DEFAULT_PROMPT_SPEC = _PromptSpec("prompt", parse_text_json_data)


class BenchPrompter(list):
    """
    Container for multiple :class:`BenchPrompt` objects.

    Parses command-line arguments and/or ``.jsonl`` prompt files, wraps
    every entry in a :class:`BenchPrompt` — or, for the chat tasks, a
    :class:`BenchChatPrompt` — and exposes an iterator over
    ``(iteration_num, prompt_idx, prompt)`` triples whose order
    respects the ``'subsequent'`` scheduling flag.

    Scheduling modes
    ----------------
    ``subsequent=False`` *(default)*
        Outer loop = iteration numbers, inner loop = prompts.
        All prompts are run in interleaved fashion within each iteration.
    ``subsequent=True``
        Outer loop = prompts, inner loop = iteration numbers.
        Prompts are processed in subsequent manner. All iterations for one prompt complete before moving to the next.
    In both modes ``num=0`` is the warm-up iteration.

    Parameters
    ----------
    args : dict
        Full benchmark args dict (as produced by
        ``model_utils.analyze_args``). Relevant keys:

            'use_case'      - object describing a pipeline type includes classes for handling the pipeline
                              in optimum/PyTorch scenarios and the name of the pipeline type (``.task`` attribute)
            'prompt_index'  - ``list[int]`` or ``None`` (prompt subset)
            'subsequent'    - ``bool`` (scheduling mode)
            'batch_size'    - ``int``
            'video_frames'  - ``int`` or ``None`` (video decimation)
    prompts : list, optional
        Pre-built prompt entries (``str`` or ``dict``) to wrap instead of
        loading them from ``args``. Used by pipelines that synthesise their
        prompts rather than reading a prompt file — e.g. the Qwen3-Omni
        speech path, which reuses the visual-language benchmark with audio
        entries it has already assembled.
    """

    def __init__(self, args, prompts=None):
        list.__init__(self)
        self._args = args
        # The spec is resolved here rather than in _load_prompts() so that
        # get_prefix() can read prefix_alias on the prompts=-supplied path too.
        # An unknown/absent task falls back to the default spec, i.e. "P".
        use_case = args.get("use_case")
        self._task = getattr(use_case, "task", None) if use_case else None
        self._spec = _PROMPT_SPECS.get(self._task, _DEFAULT_PROMPT_SPEC)
        if prompts is None:
            self._load_prompts()
        else:
            for entry in prompts:
                self.append(self._wrap(entry, args))
        if not self:
            raise RuntimeError("==Failure prompts is empty ==")

    def _wrap(self, entry, args):
        """Wrap one parsed entry per the task's spec: a chat or a single prompt."""
        if self._spec.chat_turns is not None:
            return BenchChatPrompt(self._spec.chat_turns(entry, args), args)
        return BenchPrompt(entry, args)

    def require_active(self):
        """Raise when ``args['prompt_index']`` selected no prompt at all.

        Returns ``self`` so it can be chained onto the constructor. Opt-in on
        purpose: only the VLM chat pipeline treats an empty selection as an
        error, while the other tasks run zero iterations instead. Calling this
        from more pipelines would change their behaviour.
        """
        if not self.active_pairs:
            raise RuntimeError("==Failure prompts is empty ==")
        return self

    def get_prefix(self, num, p_idx):
        alias = self._spec.prefix_alias
        if num == 0:
            return f"[warm-up][{alias}{p_idx}]"
        return f"[{num}][{alias}{p_idx}]"

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_prompts(self):
        """
        Populate the list with :class:`BenchPrompt` / :class:`BenchChatPrompt`
        objects.

        The task type (``args['use_case'].task``) selects a declarative
        :class:`_PromptSpec` (see ``_PROMPT_SPECS``) that drives every
        task-specific decision: which ``input_key`` to read, how to parse
        JSONL entries, which entry keys hold media paths to resolve, any key
        renames, how to wrap bare CLI values, and — for the chat tasks — how to
        expand an entry into its turns.
        """
        args = self._args
        if self._task is None:
            raise ValueError("(obligatory) task is not specified!")

        spec = self._spec
        input_key = spec.input_key(args) if callable(spec.input_key) else spec.input_key

        if spec.default_prompt_file is not None and all(
            args.get(key) is None for key in spec.default_prompt_file_unless
        ):
            default_prompt_file = Path(__file__).resolve().parents[1] / spec.default_prompt_file
            log.info(f"Default prompt file is used: {default_prompt_file}")
            args = dict(args, prompt_file=[str(default_prompt_file)])

        output_data_list, is_json_data = get_param_from_file(args, input_key)

        if is_json_data:
            # parse_text_json_data returns plain strings; all other parsers
            # return dicts — both are accepted by BenchPrompt.__init__.
            raw_list = spec.parse(output_data_list)
            # Path resolution needs the prompt file to resolve relative paths,
            # so it is applied only when one is present.  Key renames (e.g.
            # speech_to_text 'media' -> 'audio') run unconditionally instead, so
            # that downstream consumers always find the expected key even on a
            # JSON-without-prompt_file input.
            prompt_file = args.get("prompt_file")
            base = prompt_file[0] if prompt_file else None
            for entry in raw_list:
                # A chat entry is a list of turn dicts, a single-prompt entry is
                # one dict (or a plain string, from parse_text_json_data).
                # Resolving per leaf dict keeps both on one code path.  This
                # runs BEFORE the spec's chat_turns expansion, so a replicated
                # turn resolves its paths once rather than once per copy.
                for item in entry if isinstance(entry, list) else [entry]:
                    if not isinstance(item, dict):
                        continue
                    if base is not None:
                        for key in spec.path_keys:
                            if key not in item:
                                continue
                            value = item[key]
                            # A media key may hold a single path or a list of them.
                            if isinstance(value, list):
                                item[key] = [resolve_media_file_path(sub, base) for sub in value]
                            else:
                                item[key] = resolve_media_file_path(value, base)
                    for src, dst in spec.rename.items():
                        if src in item:
                            item[dst] = item.pop(src)
        elif spec.nonjson_wrap is not None:
            raw_list = [spec.nonjson_wrap(item) for item in output_data_list]
        else:
            raw_list = output_data_list

        if not raw_list:
            raise RuntimeError("BenchPrompter: prompt list is empty")

        for entry in raw_list:
            self.append(self._wrap(entry, args))

    # ------------------------------------------------------------------ #
    # List interface                                                       #
    # ------------------------------------------------------------------ #

    def append(self, prompt):
        """Only :class:`BenchPrompt` / :class:`BenchChatPrompt` may be appended."""
        if not isinstance(prompt, (BenchPrompt, BenchChatPrompt)):
            raise TypeError(f"BenchPrompter only accepts BenchPrompt or BenchChatPrompt objects, got {type(prompt)!r}")
        super().append(prompt)

    # ------------------------------------------------------------------ #
    # Prompt selection                                                     #
    # ------------------------------------------------------------------ #

    @property
    def active_pairs(self):
        """
        Return a list of ``(p_idx, BenchPrompt)`` pairs to be benchmarked.

        If ``args['prompt_index']`` is provided only the prompts at those
        positions are included (out-of-range indices are silently skipped).
        Otherwise all prompts are included and ``p_idx`` equals the
        position in this list.
        """
        prompt_index = self._args.get("prompt_index")
        if prompt_index is None:
            return list(enumerate(self))
        return [(i, self[i]) for i in prompt_index if 0 <= i < len(self)]

    @property
    def active_indices(self):
        """Return a plain list of prompt indices that will be benchmarked.

        Convenience shorthand; equivalent to
        ``[p_idx for p_idx, _ in self.active_pairs]``.
        """
        return [p_idx for p_idx, _ in self.active_pairs]

    @property
    def active_items(self):
        """Return a plain list of :class:`BenchPrompt` objects to be benchmarked.

        Convenience shorthand; equivalent to
        ``[p for _, p in self.active_pairs]``.
        """
        return [p for _, p in self.active_pairs]

    # ------------------------------------------------------------------ #
    # Iteration scheduling                                                 #
    # ------------------------------------------------------------------ #

    def iter_schedule(self, num_iters):
        """
        Yield ``(num, p_idx, prompt)`` triples in scheduling order.

        Parameters
        ----------
        num_iters : int
            Number of benchmark iterations *excluding* warm-up.
            ``num`` ranges from ``0`` (warm-up) to ``num_iters`` inclusive.

        Yields
        ------
        num : int
            Iteration number (``0`` = warm-up).
        p_idx : int
            Original index of the prompt in this list.
        prompt : BenchPrompt
            The prompt object for this (iteration, prompt) pair.

        Scheduling order
        ----------------
        ``subsequent=False``  ->  for num in iters: for (p_idx, p) in active
        ``subsequent=True``   ->  for (p_idx, p) in active: for num in iters
        """
        active = self.active_pairs
        subsequent = self._args.get("subsequent", False)

        if not subsequent:
            # All prompts inside each iteration
            for num in range(num_iters + 1):
                for p_idx, prompt in active:
                    yield num, p_idx, prompt
        else:
            # All iterations for each prompt before moving on
            for p_idx, prompt in active:
                for num in range(num_iters + 1):
                    yield num, p_idx, prompt
