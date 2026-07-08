# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import numpy as np
from PIL import Image
import logging as log
from transformers.image_utils import load_image
from .model_utils import get_param_from_file, resolve_media_file_path
from .parse_json_data import parse_text_json_data, parse_vlm_json_data, parse_image_json_data, parse_video_json_data, parse_speech_json_data
import llm_bench_utils.metrics_print as metrics_print
from pathlib import Path
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
        # Lazily filled by probe()
        self._image_size = None   # (width, height) | None
        self._video_shape = None   # (frames, height, width) | None
        self._audio_info = None    # (duration_sec, sample_rate) | None
        self._mask_fraction = None # float | None  cached mask coverage % (extra feedback fix)
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

        Called automatically by ``__repr__``. Safe to call multiple times
        (runs only once). Validates correctness of input data and applies
        decimation to video via ``make_video_tensor``.
        """
        if self._probed:
            return
        self._probed = True

        if self.get("media"):
            self._image_size = self._get_image_size(self["media"])

        if self.get("video"):
            decim = self._args.get("video_frames")
            self._video_shape = self._get_video_shape(self["video"], decim)

        if self.get("audio"):
            self._audio_info = self._get_audio_info(self["audio"])

        # Cache mask-coverage fraction so _get_mask_fraction is never called
        # more than once per BenchPrompt instance (extra feedback fix).
        if self.get("mask_image"):
            self._mask_fraction = self._get_mask_fraction(self["mask_image"])

    # ------------------------------------------------------------------ #
    # Static media helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_image_size(path):
        """Return ``(width, height)`` or ``None`` on failure.

        Uses :func:`~transformers.image_utils.load_image` which handles both
        local filesystem paths and HTTP(S) URLs.  The previous
        ``Image.open()`` call could not resolve HTTP URLs and emitted a
        misleading ``[Errno 2] No such file or directory`` warning for
        web-hosted prompt images (e.g. GitHub asset URLs in VLM benchmarks).
        """
        try:
            return load_image(str(path)).size
        except Exception as exc:
            log.warning(f"BenchPrompt: cannot probe image '{path}': {exc}")
            return None

    @staticmethod
    def _get_video_shape(path, decim_frames=None):
        """Return ``(frames, height, width)`` or ``None`` on failure.

        Uses cv2 container metadata for near-instant probing instead of
        decoding all frames via ``make_video_tensor``.  This eliminates the
        significant I/O + CPU overhead that previously occurred on the very
        first ``repr()`` / ``introduce_in_stdout()`` call before the
        benchmark loop (REDUCE-4 / CHANGE-3 from analysis).

        Note: the frame count reflects the raw container value and does NOT
        account for the decimation applied during the actual run.  It is used
        for informational / display purposes only.
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
        """Return the percentage of non-zero pixels in *mask_path* or ``None``."""
        try:
            arr = np.array(Image.open(mask_path).convert("L"))
            return 100.0 * float(np.count_nonzero(arr)) / arr.size
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        """
        Human-readable description of the prompt and its media dimensions.

        Format (``+`` separates modalities):

            text:7w
            text:7w + image:512x512
            text:7w + video:640x480@30f
            text:7w + audio:30.0s@44100Hz
            text:7w + image:512x512/35.2%
            text:7w + image:1024x768 + video:640x480@16f
            audio:30.0s@44100Hz      <- no text prompt

        Word count (whitespace-split) of the text prompt is shown with a
        ``w`` suffix.  Exact token count would require the model tokenizer.
        """
        self.probe()
        parts = []

        # ---- text (optional) ----
        if self.get("prompt"):
            word_count = len(self["prompt"].split())
            parts.append(f"text:{word_count}w")

        # ---- image (optionally decorated with mask coverage fraction) ----
        if self.get("media"):
            if self._image_size:
                w, h = self._image_size
                if self.get("mask_image"):
                    frac = self._mask_fraction  # pre-computed in probe()
                    if frac is not None:
                        parts.append(f"image:{w}x{h}/{frac:.1f}%")
                    else:
                        parts.append(f"image:{w}x{h}")
                else:
                    parts.append(f"image:{w}x{h}")
            else:
                parts.append("image:?x?")

        # ---- video ----
        if self.get("video"):
            if self._video_shape:
                frames, h, w = self._video_shape
                parts.append(f"video:{w}x{h}@{frames}f")
            else:
                parts.append("video:?x?@?f")

        # ---- audio ----
        if self.get("audio"):
            if self._audio_info:
                dur, sr = self._audio_info
                parts.append(f"audio:{dur:.1f}s@{sr}Hz")
            else:
                parts.append("audio:?s@?Hz")

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


class BenchPrompter(list):
    """
    Container for multiple :class:`BenchPrompt` objects.

    Parses command-line arguments and/or ``.jsonl`` prompt files, wraps
    every entry in a :class:`BenchPrompt`, and exposes an iterator over
    ``(iteration_num, prompt_idx, BenchPrompt)`` triples whose order
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
    """

    def __init__(self, args):
        list.__init__(self)
        self._args = args
        self._load_prompts()
        if not self:
            raise RuntimeError('==Failure prompts is empty ==')


    def get_prefix(self, num, p_idx):
        if num == 0:
            return f"[warm-up][P{p_idx}]"
        return f"[{num}][P{p_idx}]"

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_prompts(self):
        """
        Populate the list with :class:`BenchPrompt` objects.

        Selects the appropriate ``input_key`` for
        :func:`get_param_from_file` and the matching parse function based
        on the task type derived from ``args['use_case'].task``.
        """
        args = self._args
        use_case = args.get("use_case")
        task = getattr(use_case, "task", None) if use_case else None
        if task is None:
            raise ValueError("(obligatory) task is not specified!")

        # ---- pick input keys based on task ----
        if task == "visual_text_gen":
            input_key = ["video", "media", "prompt"]
        elif task == "image_gen":
            if use_case and hasattr(use_case, "TASK"):
                inpainting_name = use_case.TASK.get("inpainting", {}).get("name")
                img2img_name = use_case.TASK.get("img2img", {}).get("name")
                if args.get("task") == inpainting_name or (
                    (args.get("media") or args.get("images")) and args.get("mask_image")
                ):
                    input_key = ["media", "mask_image", "prompt"]
                elif args.get("task") == img2img_name or args.get("media") or args.get("images"):
                    input_key = ["media", "prompt"]
                else:
                    input_key = ["prompt"]
            else:
                input_key = ["prompt"]
        elif task == "video_gen":
            input_key = ["prompt", "negative_prompt"]
        elif task == "speech_to_text":
            # Speech prompts reference an audio file stored under the 'audio'
            # key (renamed from 'media' after loading so that BenchPrompt.probe()
            # correctly calls _get_audio_info() instead of _get_image_size()).
            # get_param_from_file still uses the 'media' key as the CLI/JSONL
            # schema key for the audio file path.
            input_key = "media"
        elif task == "ldm_super_resolution":
            # Super-resolution prompts reference an image file stored under
            # the 'prompt' key for backward-compatibility with the JSON schema.
            input_key = "prompt"
        else:
            # text_gen, code_gen, text_embed, text2speech, text_rerank, ...
            input_key = "prompt"

        output_data_list, is_json_data = get_param_from_file(args, input_key)

        if is_json_data:
            # Parse raw jsonl dicts according to task type.
            # Note: parse_text_json_data returns plain strings; all others
            # return dicts – both are accepted by BenchPrompt.__init__.
            if task == "visual_text_gen":
                raw_list = parse_vlm_json_data(output_data_list)
                # Resolve relative media/video paths against the prompt file.
                if args.get("prompt_file"):
                    for entry in raw_list:
                        if "media" in entry:
                            entry["media"] = resolve_media_file_path(entry["media"], args["prompt_file"][0])
                        if "video" in entry:
                            entry["video"] = resolve_media_file_path(entry["video"], args["prompt_file"][0])
            elif task == "image_gen":
                raw_list = parse_image_json_data(output_data_list)
                # Resolve relative media/mask_image paths against the prompt file.
                if args.get("prompt_file"):
                    for entry in raw_list:
                        if "media" in entry:
                            entry["media"] = resolve_media_file_path(entry.get("media"), args["prompt_file"][0])
                        if "mask_image" in entry:
                            entry["mask_image"] = resolve_media_file_path(entry.get("mask_image"), args["prompt_file"][0])
            elif task == "video_gen":
                # prompt/negative_prompt are plain text — no path resolution needed.
                raw_list = parse_video_json_data(output_data_list)
            elif task == "speech_to_text":
                raw_list = parse_speech_json_data(output_data_list)
                if args.get("prompt_file"):
                    for entry in raw_list:
                        if "media" in entry:
                            # Rename 'media' -> 'audio' so BenchPrompt.probe()
                            # correctly calls _get_audio_info() instead of
                            # _get_image_size() on the audio file path.
                            entry["audio"] = resolve_media_file_path(
                                entry.pop("media"), args["prompt_file"][0]
                            )
            elif task == "ldm_super_resolution":
                raw_list = parse_image_json_data(output_data_list)
                if args.get("prompt_file"):
                    for entry in raw_list:
                        if "prompt" in entry:
                            entry["prompt"] = resolve_media_file_path(
                                entry["prompt"], args["prompt_file"][0]
                            )
            else:
                raw_list = parse_text_json_data(output_data_list)
        else:
            # For speech_to_text the raw value is an audio-file path which
            # must be stored under 'audio' (not 'prompt') so that
            # BenchPrompt.probe() calls _get_audio_info() on it correctly.
            if task == "speech_to_text":
                raw_list = [{"audio": item} for item in output_data_list]
            else:
                raw_list = output_data_list

        if not raw_list:
            raise RuntimeError("BenchPrompter: prompt list is empty")

        for entry in raw_list:
            self.append(BenchPrompt(entry, args))

    # ------------------------------------------------------------------ #
    # List interface                                                       #
    # ------------------------------------------------------------------ #

    def append(self, prompt):
        """Only :class:`BenchPrompt` objects may be appended."""
        if not isinstance(prompt, BenchPrompt):
            raise TypeError(f"BenchPrompter only accepts BenchPrompt objects, got {type(prompt)!r}")
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

    # ------------------------------------------------------------------ #
    # Static helper                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_prompt_data(inputs, required_frames, genai_flag):
        """
        Unpack a list of prompt dicts into separate ``(prompts, images, videos)``
        lists, loading each media file on the fly.

        Parameters
        ----------
        inputs : list[dict] | dict
            One or more prompt dicts (each may have ``'prompt'``, ``'media'``,
            and/or ``'video'`` keys).
        required_frames : int | None
            Frame decimation target forwarded to :func:`make_video_tensor`.
        genai_flag : bool
            When ``True`` video tensors are returned as ``ov.Tensor``; when
            ``False`` as ``np.ndarray``.

        Returns
        -------
        prompts : list[str]
        images  : list
        videos  : list
        """
        prompts, images, videos = [], [], []
        if not isinstance(inputs, (list, tuple, set)):
            inputs = [inputs]
        for input_data in inputs:
            if input_data.get("video") is not None:
                entry = Path(input_data["video"])
                if entry.is_dir():
                    for filename in sorted(entry.iterdir()):
                        video_tensor = make_video_tensor(filename, required_frames, genai_flag)
                        videos.append(video_tensor)
                else:
                    video_tensor = make_video_tensor(entry, required_frames, genai_flag)
                    videos.append(video_tensor)
            if input_data.get("media") is not None:
                entry = Path(input_data["media"])
                if entry.is_dir():
                    for file in sorted(entry.iterdir()):
                        pil_img = load_image(str(file))
                        img = ov.Tensor(np.array(pil_img)[None]) if genai_flag else pil_img
                        images.append(img)
                else:
                    # Always load as PIL first so we can update the BenchPrompt
                    # repr metadata with actual image dimensions after the real
                    # media file has been fetched (handles HTTP(S) URLs and local
                    # paths uniformly).  This satisfies Sofia's review point:
                    # "representation should be updated after real media files
                    # are loaded" — the _image_size cached by probe() (which
                    # runs before inference in introduce_in_stdout) is
                    # overwritten here with the value from the truly loaded
                    # image, so the final prompt_repr in the JSON report is
                    # always accurate.
                    pil_img = load_image(input_data["media"])
                    if isinstance(input_data, BenchPrompt):
                        input_data._image_size = pil_img.size
                    img = ov.Tensor(np.array(pil_img)[None]) if genai_flag else pil_img
                    images.append(img)
            prompts.append(input_data["prompt"])
        return prompts, images, videos
