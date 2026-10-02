---
name: tests-and-cache
description: "Run, write, and fix OpenVINO GenAI Python and WWB tests, and set up their Hugging Face / OpenVINO model and dataset caches. Use when: running tests/python_tests or tools/who_what_benchmark/tests locally, adding a test that downloads a model/dataset/adapter from the Hugging Face Hub, fixing tests that fail because a Hub resource can't be found in the local cache, warming or debugging CI caches, adding a test entry to the CI test matrices."
---

# Tests and Their Caches

## Layout

| What | Where |
|---|---|
| Python API tests | `tests/python_tests/` (helpers in `tests/python_tests/utils/`) |
| Sample tests | `tests/python_tests/samples/` |
| WWB tests | `tools/who_what_benchmark/tests/` (helpers in `conftest.py`, `ov_utils.py`) |
| WWB default datasets | `tools/who_what_benchmark/whowhatbench/*_evaluator.py`, `whowhatbench/utils.py` |
| CI test matrices | `.github/workflows/test_matrices/<platform>/{wheel_tests,samples_tests}.yml` |

## Environment

```sh
python3.11 -m venv .venv311 && source .venv311/bin/activate
python -m pip install -r tests/python_tests/requirements.txt
python -m pip install openvino-genai "openvino-tokenizers[transformers]"   # or a locally built wheel / PYTHONPATH=<build dir>
python -m pip install -e tools/who_what_benchmark                          # only for WWB tests
```

`.venv*` directories are git-ignored.

## Running

```sh
python -m pytest -v tests/python_tests/test_llm_pipeline.py -k "<expr>"
python -m pytest -v tests/python_tests/ --model_ids "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
python -m pytest -v tools/who_what_benchmark/tests/test_cli_vlm.py
```

- `pytest.ini` defaults to `-m "not real_models and not nightly"`. Pass `-m` explicitly to run those.
- Sample tests need `SAMPLES_PY_DIR`, `SAMPLES_CPP_DIR`, `SAMPLES_C_DIR`, `SAMPLES_JS_DIR`.
- Copy the exact CI command from the matching entry in `.github/workflows/test_matrices/<platform>/wheel_tests.yml`. Some entries install a different `transformers` version first.

## Caches

| Variable | Content | Notes |
|---|---|---|
| `HF_HOME` | Hugging Face Hub cache (`$HF_HOME/hub`): models, dataset files, adapters | Persistent shared mount in CI. This is the only cache tests may rely on for Hub resources. |
| `HF_DATASETS_CACHE` | `datasets` Arrow cache and dataset modules | In CI it points to `/tmp` and is **not** persistent. Never rely on it. |
| `OV_CACHE` | Converted OpenVINO models (`<OV_CACHE>/<YYYYMMDD>/optimum-intel-<ver>_transformers-<ver>/{downloaded_models,converted_models}`) | Unset means a temporary directory per session. The date and version subfolders make it rebuild daily and on dependency bumps. |
| `CLEANUP_CACHE` | Removes the `OV_CACHE` session directory after the run | Doesn't touch `HF_HOME`. |

Converted-model helpers: `utils.hugging_face.download_and_convert_model()`, `utils.constants.get_ov_cache_converted_models_dir()`, and the WWB `conftest.convert_model()`. They write through `AtomicDownloadManager` so parallel jobs never see partial directories.

## Rules for Hub Resources in Tests

Every Hub access must resolve to files stored in `$HF_HOME/hub` so that a warm cache is enough to run the test without network access.

### Models and single files

- Models: `snapshot_download(model_id)` wrapped in `retry_request(...)` (`utils.network` / WWB `ov_utils`). Pass the returned path to `from_pretrained`, `optimum-cli`, or GenAI pipelines.
- Single files (LoRA adapters, GGUF, JSON): `snapshot_download(repo_id, allow_patterns=[filename])`, then `Path(result) / filename`.
- Don't use `hf_hub_download(..., local_dir=tmp_path)`. It bypasses the shared cache and downloads again on every run.
- Pin a revision with the **full 40-char commit hash**. Short hashes and branch names can't be resolved from the cache alone.

### Datasets

Pick the helper by dataset size:

| Case | Helper |
|---|---|
| Small dataset, whole repo is fine | `utils.dataset_utils.load_dataset_via_snapshot(repo_id, config, split=...)` |
| Large or sharded dataset: fetch only the needed parquet files | `utils.dataset_utils.load_parquet_dataset_via_snapshot(repo_id, {"<split>": "<repo-relative glob>"}, revision=..., split=..., streaming=...)` |
| WWB default datasets | `whowhatbench.utils.load_hub_parquet_dataset(repo_id, {"<split>": "<glob>"}, ...)` |
| WWB CLI `--dataset` in a test | `snapshot_download(..., repo_type="dataset", allow_patterns=[...])` in the test, then pass the local split directory (e.g. `<snapshot>/unlabeled`) |

- Don't call `datasets.load_dataset("org/name", ...)` with a Hub id, with or without `streaming=True`. It resolves through the Hub API and `HF_DATASETS_CACHE`.
- Find the files a config/split maps to with `datasets.load_dataset_builder(repo_id, config).config.data_files`. Many script-less repos keep them under `<config>/<split>-*.parquet`, `data/<split>-*`, or `parquet-data/<config>/<split>-*`.
- A partial snapshot plus `load_dataset(<snapshot dir>)` fails with `DataFilesNotFoundError` when the README declares configs whose files weren't downloaded. Load through the `"parquet"`/`"json"` builder with explicit `data_files`, or point at the split subdirectory.
- When changing a WWB default dataset loader, keep the samples identical: compare fingerprints of `take(N)` from the old Hub streaming loader and the new local loader. `shuffle(seed)` over many shards reorders shards, so loading a subset of shards changes the samples.

### Diffusers LoRA

`pipe.load_lora_weights(<file>)` can't guess `weight_name` without Hub access. Pass the parent directory plus `weight_name=<file name>`, as `whowhatbench.model_loaders._apply_diffusers_lora_adapters` does.

## Warming and Verifying the Cache

1. Run the new or changed test once with network access and the same `HF_HOME` CI uses (`/mount/caches/huggingface/lin` on Linux, `C:/mount/caches/huggingface/win` on Windows). This populates `$HF_HOME/hub`.
2. Re-run it with network access disabled. It must pass using only cached files.
3. Check what was cached: `ls $HF_HOME/hub/{models,datasets}--<org>--<name>/snapshots/*/`.
4. macOS CI uses per-run, non-persistent `HF_HOME` directories, so nothing is warm there.

## Known Pitfalls

- `tqdm==4.70.0`: `thread_map` fails on generators. That makes `snapshot_download` raise `ValueError: min() arg is an empty sequence` on the first download from repos with >1000 files (e.g. `facebook/multilingual_librispeech`, `google/fleurs`, `lmms-lab/LLaVA-Video-178K`). Use `tqdm==4.70.1`.
- Errors that mean a resource is missing from `$HF_HOME/hub`: `LocalEntryNotFoundError`, `ConnectionError: Couldn't reach '<repo>' on the Hub (OfflineModeIsEnabled)`, `ValueError: When using the offline mode, you must specify a weight_name`. Fix the test to go through the cache as described above instead of retrying.
- WWB tests run `wwb` in a subprocess. The real error is in the captured output after `ERROR:conftest:'wwb ...' returned 1. Output:`, not in the `CalledProcessError` line.
- A module-scoped fixture failure is reported once with its output. Later tests that use the same fixture only show the cached `CalledProcessError`.

## Adding a Test to CI

Add or extend an entry in `.github/workflows/test_matrices/<platform>/wheel_tests.yml` (or `samples_tests.yml`) with `name`, `cmd`, `timeout`, and `components`. An entry runs only when one of its `components` is affected, per `.github/scripts/build_test_matrices.py`.
