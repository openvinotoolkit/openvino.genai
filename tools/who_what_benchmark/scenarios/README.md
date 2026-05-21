# WWB Scenarios

Scenarios let you describe a full benchmark comparison — base models, quantized targets, tasks, and datasets — in a single YAML file.

## Quick start

```bash
# Dry run — validate and print the planned execution matrix (no models loaded)
wwb run scenarios/llm_quantization.yaml --dry-run

# Run the scenario (requires model paths to exist)
wwb run scenarios/llm_quantization.yaml --output /tmp/results

# Run only a subset of tasks
wwb run scenarios/llm_quantization.yaml --only text_quality

# Override the output directory
wwb run scenarios/llm_quantization.yaml --output /my/results
```

## Shipped scenarios

| File | What it tests |
|------|---------------|
| `llm_quantization.yaml` | Text + chat quality: FP16 HF baseline vs int4/int8 OpenVINO models |
| `vlm_quality.yaml` | Visual question answering: HF VLM vs quantized OpenVINO variant |
| `rag_pipeline.yaml` | Embedding + reranking quality: FP16 vs int8 for two RAG components |

## Scenario file structure

```yaml
schema_version: 1
name: my-scenario           # slug: lowercase, hyphens/underscores only

defaults:                   # applied to every task unless overridden
  device: CPU
  num_samples: 32
  seed: 42

models:
  my_base:
    path: org/model-id      # HuggingFace ID or local path
    backend: hf             # hf | genai | llamacpp | onnx

datasets:
  my_data:
    type: builtin           # builtin | huggingface | csv | inline

tasks:
  - id: my_task
    type: text              # any of the 12 WWB task types
    base: my_base
    targets: [my_target]
    dataset: my_data

report:
  formats: [markdown, json]
  group_by: task            # task | target
```

## Output

After a run, the output directory contains:

```
<output_dir>/
  report.md               ← human-readable leaderboard
  report.json             ← machine-readable metrics (CI-friendly)
  run_manifest.json       ← versions, hashes, timing (reproducibility)
  tasks/
    <task_id>/<target_id>/
      metrics.csv
      metrics_per_question.csv
      target.csv
```

## Supported task types

`text`, `text-chat`, `text-to-image`, `text-to-video`, `speech-generation`,
`visual-text`, `visual-text-chat`, `visual-video-text`, `image-to-image`,
`image-inpainting`, `text-embedding`, `text-reranking`
