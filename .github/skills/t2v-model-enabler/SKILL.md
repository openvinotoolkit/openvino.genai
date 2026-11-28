---
name: t2v-model-enabler
description: "Enables a text-to-video (and image-/video-conditioned) diffusion model in the GenAI video generation pipelines (Text2VideoPipeline, Image2VideoPipeline). Use when: adding a new video model family or a new version of a supported one (e.g. LTX-Video, LTX-2, LTX-2.3), porting a diffusers video pipeline to C++, or debugging a GenAI-vs-optimum-intel/diffusers mismatch in a video pipeline."
argument-hint: "HuggingFace model_id of the diffusers pipeline (e.g. diffusers/LTX-2.3-Diffusers), or a path to an OpenVINO export made by optimum-intel."
---

# Text-to-Video Model Enabler

Enables a diffusers video pipeline in GenAI. The GenAI pipeline is a C++ port of the diffusers `__call__`: the
models run as OpenVINO IRs exported by optimum-intel, and everything between them (noise, schedule, guidance math,
positions, packing, post-processing) is host code that must reproduce the reference exactly. The workflow is built
around proving that, stage by stage.

## Input

- The diffusers repo id (e.g. `diffusers/LTX-2.3-Diffusers`) and its tiny test fixture if one exists
  (`optimum-intel-internal-testing/tiny-random-<model>`), or a path to an existing OpenVINO export.
- The optimum-intel version that exports the model. GenAI cannot run what optimum-intel cannot export: if the export
  does not exist yet, it is a prerequisite, not part of this skill.

## Working Directory

Keep all scripts, reference dumps and notes for one model in `.model_enabler/<model_type>/`, where `<model_type>` is
the transformer's `_class_name` family or the pipeline `_class_name` from `model_index.json`. Never share it between
models.

## Rules

- Write every experiment as a `.py` file in the working directory; keep shell commands to build/run steps.
- Validate on the tiny fixture first (seconds per run), on the full model last (tens of minutes per video on CPU).
- Compare against **optimum-intel's OpenVINO pipeline on the same export**, not against diffusers directly: both
  then run the same IRs, so any difference is host code. Compare optimum-intel against diffusers separately.
- Never reason about a mismatch from end results alone. Bisect it: tokens → text embeddings → per-step latents →
  decoded video (see [video-validation.md](video-validation.md)).
- After each step, check the checkpoint exists before moving on. Track steps with the todo list.

## Reference

- [genai-t2v-architecture.md](genai-t2v-architecture.md): how the video pipelines are put together, the interfaces a
  new model implements, and the checklist of files to touch. Read it before Step 2.
- [video-validation.md](video-validation.md): exporting, generating references, comparing videos and audio, bisecting
  mismatches, full-size wwb runs. Read it before Step 3.
- `.github/skills/tests-and-cache/SKILL.md`: how the Python tests find and cache exported models.

---

## Step 1 — Analyze the reference pipeline and the export

**Goal:** know exactly what the diffusers pipeline computes and what the OpenVINO export provides.

1. Export the tiny fixture with optimum-intel, tokenizer included (`optimum-cli export openvino --model <id>
   --task text-to-video <dir>`; from Python, `main_export(..., convert_tokenizer=True)` with a `Path` output).
2. Inventory the export: `model_index.json` (component class names; `[null, null]` entries are absent components),
   every `*/openvino_model.xml` with its input/output names, element types and partial shapes, every component
   `config.json`, the scheduler config. Write a script that prints this; do not read XML by hand.
   Diff it against the original diffusers `model_index.json`: every non-null component there must be exported,
   including optional ones the text-to-video path does not use (VAE encoders, prompt enhancer, processor). Report
   missing ones as an optimum-intel prerequisite.
3. Read the diffusers pipeline `__call__` end to end and list, in order:
   - text encoding: tokenizer settings (padding side, max length, BOS), which hidden states are used and how they
     are packed;
   - every model call with its exact inputs, including host-computed ones (positions, RoPE, masks, timesteps);
   - the noise: shape and layout at the moment `randn_tensor` is called, and the order of all random draws;
   - the schedule: `set_timesteps` arguments, dynamic shifting, `mu`, custom `sigmas`;
   - guidance: which passes run, in which space (velocity or x0) the terms are combined, rescaling;
   - decoding and post-processing, audio if any;
   - every default of `__call__`, and which of them the model card says to override (distilled checkpoints).
4. Read optimum-intel's OpenVINO pipeline for the model (`optimum/intel/openvino/modeling_diffusion.py`): what it
   computes on the host, which flags it **baked into the IR at trace time** (anything the patcher fixed rather than
   exposed as an input), and which IR inputs it feeds.
5. Find the closest existing GenAI pipeline (`ltx_pipeline.hpp` for LTX-Video, `ltx2_pipeline.hpp` for LTX-2.x) and
   diff the reference against it.

### Checkpoint

`.model_enabler/<model_type>/design.md` with:

```
## Reference
- Components and IR interfaces: <table>
- Host-side computations: <list, with the diffusers code they port>
- Random draws: <shape, layout, order>
- Defaults: <__call__ defaults, and the recommended overrides>
- Baked-in export choices: <flags fixed at trace time>
## GenAI design
- Closest pipeline: <file> — differences: <list>
- New or changed files: <list, per the architecture checklist>
- Version detection: <how the pipeline tells this model/version apart, from config keys or model_index>
```

---

## Step 2 — Implement the components

**Goal:** every model wrapper produces optimum-intel's outputs for identical inputs.

1. For each IR without a GenAI wrapper, add a class in `src/cpp/include/openvino/genai/video_generation/` and
   `src/cpp/src/video_generation/models/`, following an existing one (`ltx2_vocoder` is the smallest): constructor
   reading the model and `config.json`, `compile`, `clone`, `reshape`, `infer`. Return owned tensors where the next
   call would overwrite the request's output.
2. Port host-side computations next to the wrapper that consumes them (e.g. RoPE in the transformer wrapper).
   Reproduce the reference's precision exactly (fp64 where it uses fp64, Python's round-half-to-even where it
   rounds).
3. Bind the wrapper in `src/python/py_video_generation_models.cpp`, so it can be fed identical inputs from Python.

### Checkpoint

`.model_enabler/<model_type>/components.md`: per component, the max abs diff against optimum-intel's wrapper with
identical inputs. Expect ~1e-6 at f32 (`INFERENCE_PRECISION_HINT=f32`).

---

## Step 3 — Implement the pipeline

**Goal:** `Text2VideoPipeline` reproduces optimum-intel's video and audio for the same seed.

1. Dispatch: add the `model_index.json` class to `text2video_pipeline.cpp` (and `image2video_pipeline.cpp` if
   supported), or extend the existing pipeline's version detection.
2. Defaults: add a `<MODEL>_DEFAULT_CONFIG`. `VideoGenerationConfig` defaults are positional aggregates — append new
   config fields at the end of the struct, never in the middle. Select the defaults from a config key that tells the
   versions apart, and say in a comment which key and why.
3. `generate`: port `__call__` in order. Draw noise with the reference's exact shapes, layouts and order. Combine
   guidance in the same space as the reference.
4. `reshape_models`, `compile` (including the deferred per-stage GPU path), `clone`, `replace_defaults`, `decode`.
5. New config fields: field + `ov::Property` in `generation_config.hpp`, parsing in `generation_config_utils.cpp`
   (`update_generation_config`, and validation), `def_readwrite` in `py_video_generation_pipelines.cpp`, and in
   `py_utils.cpp` if Python must convert a list or custom type for it.

### Checkpoint

`.model_enabler/<model_type>/parity.md`: tiny-model video max abs diff (0–255) and audio max abs diff against
optimum-intel, for text-to-video with the default config and with the reference's recommended settings. Expect
video ≤ 1/255 (rounding) and audio ~1e-6.

---

## Step 4 — Tests, docs, samples

1. Tests in `tests/python_tests/test_video_generation.py`: add the tiny fixture to `VIDEO_GEN_MODELS`; cover the
   default config values, output shapes (video and audio), determinism, static reshape, every new config field, and
   error paths. Add a parity test against optimum-intel that skips when the installed optimum-intel cannot run it.
2. Docs: a row in `site/docs/supported-models/_components/video-generation-models-table/models.ts`; the export
   command and model notes in `samples/python/video_generation/README.md`.
3. Samples: make sure the existing samples do not hard-code settings that break the new model (e.g. a step count
   or guidance scale that overrides a distilled model's defaults); add a model-specific sample only for features the
   generic ones cannot show.

### Checkpoint

The video generation tests pass, the new tests included.

---

## Step 5 — Full-size validation

Run the full model through GenAI with its recommended settings on wwb's prompts and score it with wwb against the
diffusers ground truth, the same way the optimum-intel export was scored (see [video-validation.md](video-validation.md)).
The GenAI score should match optimum-intel's; GenAI-vs-optimum-intel similarity should be close to 1.0. Record peak
memory (`/usr/bin/time -v`).

### Checkpoint

`.model_enabler/<model_type>/full_size.md`: wwb scores (GenAI vs ground truth, optimum-intel vs ground truth, GenAI
vs optimum-intel), time per video, peak RSS.

---

## Pitfalls seen while enabling LTX-2.x

| Symptom | Cause |
|---|---|
| Video latents drift ~1e-2 on the tiny model, audio matches | Tiny fixtures map every word to `<unk>`; HF merges a run of unknowns into one token, the OpenVINO tokenizer does not. Use `prompt=""` for tiny parity. |
| Same seed, different video | Noise drawn in a different layout or order: diffusers pipelines of one family differ (5D `[B,C,F,H,W]` then packed vs drawn packed). |
| Conditioned outputs differ, text-to-video matches | A flag baked into the export at trace time differs from the reference's default for that diffusers version. GenAI cannot see it; check what the patcher traced. |
| Generated text (prompt enhancer, captions) diverges after some tokens | A generation setting that GenAI applies differently (e.g. `no_repeat_ngram_size`), or prompt ids missing from the sampler history. Compare against HF token by token. |
| Image-conditioned output differs completely | Resizing that does not match PIL bit for bit. Use GenAI's Pillow ports (`bicubic_resize`/`bilinear_resize` in `visual_language/clip.cpp`), not OpenVINO's `*_PILLOW` interpolation. |
| A frame count or duration is off by one grid step | Python's `round` is half-to-even; use `std::nearbyint`. |
| A default config value lands in the wrong field | A field inserted mid-struct into `VideoGenerationConfig`. Append only. |
| A full-size CPU run is killed | GenAI compiles every component at construction on CPU; text encoder + transformer can exceed RAM. Check peak RSS on a short run first. |
