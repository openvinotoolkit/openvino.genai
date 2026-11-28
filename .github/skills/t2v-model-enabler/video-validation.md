# Video Validation

How to run a video model through GenAI, check its frames and audio against the reference, and find where a
mismatch comes from. Keep the scripts in `.model_enabler/<model_type>/`.

## Environments

Use two Python environments and exchange results through `.npz` files:

- **Reference**: optimum-intel (a version that exports the model), diffusers, torch. Produces exports and reference
  outputs.
- **GenAI**: the editable GenAI build (see `genai-t2v-architecture.md`, "Build and Run Notes"). If its import fails
  on `libopenvino_*.so`, set `LD_LIBRARY_PATH=<venv>/lib/python3.*/site-packages/openvino/libs`.

GenAI's test environment pins its own optimum-intel/transformers/diffusers (`tests/python_tests/requirements.txt`),
which may be too old to export a new model; the reference environment does not have to match it.

## Export

```bash
# Tiny fixture: seconds per run, random weights
optimum-cli export openvino --model optimum-intel-internal-testing/tiny-random-<model> --task text-to-video tiny_ov
# Full model
optimum-cli export openvino --model <org>/<model> --task text-to-video full_ov
```

From Python, `main_export(model_id, Path(out), task="text-to-video", convert_tokenizer=True)`: without
`convert_tokenizer=True` there is no `openvino_tokenizer.xml`, and GenAI cannot encode prompts.

Check the export before anything else: component list in `model_index.json`, IR inputs/outputs, `rt_info` flags
(`ov.Core().read_model(xml).has_rt_info([...])`).

## Reference Outputs

Generate with optimum-intel's OpenVINO pipeline on the same export, at f32, with a fixed seed, and save:

```python
# ref.py <export_dir> <out.npz>
import sys
import numpy as np
import torch
from optimum.intel.openvino import OVLTX2Pipeline  # the model's OV pipeline class

model_dir, out = sys.argv[1], sys.argv[2]
pipe = OVLTX2Pipeline.from_pretrained(model_dir, ov_config={"INFERENCE_PRECISION_HINT": "f32"})
video, audio = pipe(
    prompt="",  # tiny fixtures: see "Tokenizer" below
    height=64, width=64, num_frames=17, frame_rate=24.0,
    # pass every setting explicitly, the same values as on the GenAI side
    generator=torch.Generator().manual_seed(42), output_type="np", return_dict=False,
)
np.savez(out, video=np.asarray(video), audio=audio.float().numpy())
```

Validate optimum-intel against diffusers on the same inputs separately (optimum-intel's own tests do), so a GenAI
comparison only has the host code left to explain.

## GenAI Run and Comparison

```python
# compare.py <export_dir> <ref.npz>
import sys
import numpy as np
import openvino_genai as ov_genai

model_dir, ref_path = sys.argv[1], sys.argv[2]
pipe = ov_genai.Text2VideoPipeline(model_dir, "CPU", INFERENCE_PRECISION_HINT="f32")
result = pipe.generate("", height=64, width=64, num_frames=17, frame_rate=24.0,
                       generator=ov_genai.TorchGenerator(42))  # same settings as the reference
ref = np.load(ref_path)
video = result.video.data.astype(np.float32)          # u8 [B, F, H, W, 3]
ref_video = np.round(ref["video"] * 255.0)            # diffusers "np" output is float in [0, 1]
diff = np.abs(video - ref_video)
print("video max", diff.max(), "mean", diff.mean(), "per-frame max", diff.reshape(diff.shape[1], -1).max(1))
print("audio max", np.abs(result.audio.data - ref["audio"]).max())
```

`TorchGenerator(seed)` reproduces `torch.Generator().manual_seed(seed)` draws, so both sides start from the same
noise only if they draw the same shapes in the same order. `CppStdGenerator` (GenAI's default) does not match torch.

Expected on the tiny model at f32:

| Signal | Expected |
|---|---|
| Video | max ≤ 1 (out of 255, rounding), mean ≪ 0.01 |
| Audio | ~1e-6 |
| Video max a few levels, mean ~0.4 | not rounding: bisect |

## Bisecting a Mismatch

Work from the inputs forward and stop at the first stage that differs.

1. **Tokens.** Compare GenAI's tokenizer (`ov_genai.Tokenizer(<dir>/tokenizer).encode(text, add_special_tokens=True)`)
   with HF's (`AutoTokenizer.from_pretrained(<dir>/tokenizer)`).
2. **Components with identical inputs.** Feed the same tensors to GenAI's Python-bound wrappers
   (`ov_genai.LTX2TextConnectors(...).infer`, `LTX2VideoTransformer3DModel` with `set_rope_coords`,
   `set_hidden_states`, `infer`) and to optimum-intel's wrappers (`pipe.connectors(...)`, `pipe.transformer(...)`).
   A match (~1e-6) clears the component and its host-side inputs (RoPE, masks).
3. **Per-step latents.** Record the packed latents after every step on both sides and compare them step by step:
   - GenAI: `generate(..., callback=lambda step, total, latent: steps.append(np.array(latent.data)) or False)`
   - optimum-intel: `pipe(..., callback_on_step_end=fn)` with
     `fn = lambda pipe, i, t, kw: steps.append(kw["latents"].clone().numpy()) or kw`
   A difference already at step 0 is in the noise, the conditioning, or the transformer's inputs. One that grows
   from a small step-0 value is in a model input. One that appears at a later step is in the schedule or the
   guidance.
4. **Decoding.** Same final latents but different frames: unpacking, denormalization, VAE post-processing.

## Known Causes

| Symptom | Cause | Check |
|---|---|---|
| Video drifts ~1e-2, audio matches, tiny model only | Tiny fixtures map every word to `<unk>`; HF merges a run of them into one token, the OpenVINO tokenizer emits one per character. | Step 1. Use `prompt=""` for tiny parity; the problem does not exist on real vocabularies. |
| Different from step 0, same seed | Noise drawn in another shape, layout or order (e.g. `[B,C,F,H,W]` then packed vs `[B,S,C]`; video before audio). | Read the reference's `prepare_latents` / `prepare_audio_latents`. |
| Matches until conditioning is involved | A trace-time flag in the export differs from the reference default of that diffusers version (LTX-2: `use_cross_timestep`). | Run the reference with the flag the export traced. |
| Small per-step growth everywhere | Host-side position or RoPE precision (fp32 vs fp64), a shifted schedule. | Step 2 with the coordinates the reference computes. |
| Off by one frame or one grid step | Python `round` is half-to-even (`std::nearbyint`), integer flooring of sizes. | Compare the derived sizes directly. |
| Image-conditioned outputs differ | Resizing that is not bit-exact with PIL. | Compare the resized image arrays. |
| Generated text diverges (prompt enhancer) | Decoding settings applied differently (`no_repeat_ngram_size`), prompt tokens missing from the sampler history, image preprocessing. | Compare token by token against HF `generate`. |

## Looking at the Frames

Numbers first, but look at the video when a number is borderline or a full-size output is suspicious:

```bash
# frames 0, middle and last side by side
ffmpeg -v error -y -i video.mp4 -vf "select=eq(n\,0)+eq(n\,60)+eq(n\,120),scale=320:-1,tile=3x1" -frames:v 1 frames.png
```

Then read `frames.png` with the image viewer tool. For the audio, check the level and duration (`soundfile`) before
listening: LTX-2 audio follows the prompt's description of the sound, so a quiet prompt gives quiet audio, which is
not an export error.

## Full-Size Validation with wwb

1. Ground truth: diffusers videos for wwb's default prompts with the recommended settings, and their `gt.csv`. Reuse
   an existing one when it was made with the settings you compare.
2. GenAI target: generate the same prompts, sizes, frame count, frame rate and seed with GenAI, write
   `target/video_<i>.mp4` and a `target.csv` with the GT's columns (`prompt`, `negative_prompt`, `width`, `height`,
   `guidance_scale`, `videos`). Use a script rather than wwb's GenAI runner when the model needs settings wwb does not
   pass: wwb's runner always passes `num_inference_steps`, which replaces a custom (e.g. distilled) schedule.
3. Score, from the directory the CSVs' relative paths start at:

   ```bash
   wwb --gt-data gt.csv --target-data target/target.csv --model-type text-to-video --output scores
   ```

4. Compare three numbers: GenAI vs ground truth, optimum-intel vs ground truth (should be equal), GenAI vs
   optimum-intel (should be close to 1.0; LTX-2.5 distilled: 0.973, 0.973 and 0.998).
5. Read the per-prompt scores, not only the average. Video generation is chaotic: a tiny numerical difference can
   send one prompt to a different video, so a few prompts can pull the average down while the pipeline is exact.
   GenAI-vs-optimum-intel near 1.0 rules out a GenAI defect; a low score against ground truth then belongs to the
   optimum-intel side, where it may still be chaos rather than export error.

Run long jobs under `nohup /usr/bin/time -v ... &` and check `Maximum resident set size` at the end; a killed run
shows in `dmesg -T | grep -i "killed process"`.
