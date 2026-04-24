# Model Analysis

Analyze the target model architecture against the GenAI VLM pipeline to produce an enablement design.
See [genai-vlm-architecture.md](genai-vlm-architecture.md) for the current GenAI VLM pipeline architecture reference.

## Prerequisites

Check that `transformers` and `optimum-intel` are installed in the active Python virtual environment.
If not, install them. If user hints at a custom version or branch, install from that source.

## Step 1 — Export Model with Optimum-Intel

If `model_id` and `task` are provided, export the model to OpenVINO IR:

```bash
optimum-cli export openvino \
  --model <model_id> \
  --task <task> \
  <model_dir>
```

After export, read all `.json` configuration files in `<model_dir>`.
Detect `model_type` and `architecture` from `config.json`.

## Step 2 — Analyze Transformers Implementation

Locate the Transformers source for the model:

```
<python_env>/lib/python3.*/site-packages/transformers/models/<model_type>/
```

Identify:
- Model class hierarchy (e.g., `ForConditionalGeneration`, `VisionModel`, `LanguageModel`)
- Forward pass inputs/outputs for each sub-model
- Custom components: attention mechanisms, vision encoders, resamplers, projectors
- Special token handling for image/video placeholders
- Image preprocessing logic (resizing, normalization, tiling/slicing strategy)

### Useful code samples

Print model architecture and I/O:

```python
from transformers import AutoModel, AutoProcessor
import torch

model = AutoModel.from_pretrained("<model_id>", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("<model_id>", trust_remote_code=True)

# Print sub-modules tree
print(model)

# Trace a forward pass to see tensor shapes
from PIL import Image
import requests
image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(text="Describe this image.", images=[image], return_tensors="pt")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
```

## Step 3 — Analyze Optimum-Intel Export Artifacts

Locate the Optimum-Intel model implementation:

```
<python_env>/lib/python3.*/site-packages/optimum/intel/openvino/
```

Enumerate exported OpenVINO IR models in `<model_dir>` (`openvino_*.xml`).
For each model, document:
- File name and purpose (vision encoder, language model, projector, etc.)
- Input names, shapes, and dtypes
- Output names, shapes, and dtypes
- Whether the model is optional or required

### Useful code samples

Inspect exported IR inputs/outputs:

```python
from openvino import Core

core = Core()
for xml_path in sorted(Path("<model_dir>").glob("openvino_*.xml")):
    model = core.read_model(xml_path)
    print(f"\n=== {xml_path.name} ===")
    for inp in model.inputs:
        print(f"  INPUT  {inp.any_name}: {inp.partial_shape}, {inp.element_type}")
    for out in model.outputs:
        print(f"  OUTPUT {out.any_name}: {out.partial_shape}, {out.element_type}")
```

Run reference inference with Optimum-Intel (for later accuracy comparison):

```python
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image

model = OVModelForVisualCausalLM.from_pretrained("<model_dir>")
processor = AutoProcessor.from_pretrained("<model_dir>")

image = Image.open("test_image.jpg")
inputs = processor(text="Describe this image.", images=[image], return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Step 4 — Analyze GenAI Enablement

Compare the model architecture with the GenAI VLM pipeline (see [genai-vlm-architecture.md](genai-vlm-architecture.md)).

Key questions to answer:
1. Which existing GenAI model implementation is closest? Can it be reused or extended?
2. Does the vision encoder follow a standard pattern (single-pass, tiled/sliced, multi-scale)?
3. How are image tokens inserted into the text prompt? What special tokens are used?
4. Does the model need custom position IDs (e.g., 3D position embeddings like Qwen2-VL)?
5. Are there additional sub-models beyond vision encoder + language model (e.g., resampler, projector, merge model)?
6. Does the model support video? If so, how are frames encoded?
7. What image preprocessing is needed (normalization constants, resize strategy, tiling)?

Map each component to the GenAI interfaces:
- Vision encoding → `VisionEncoder` subclass
- Prompt normalization + embedding merge → `IInputsEmbedder` subclass
- Image preprocessing → `ProcessorConfig` or custom preprocessing in `VisionEncoder::encode()`
- Config fields → `VLMConfig` additions (if any)

## Output — Architecture Analysis Report

After analysis, create `<model_type>_architecture_analysis.md` in the working directory:

```markdown
## Model
- **Model ID**: `<model_id>`
- **Task**: `<task>`
- **Model Type**: `<model_type>`
- **Architecture**: `<architecture>`

## Transformers Analysis
### Sub-models
- List each sub-model (vision encoder, language model, projector, etc.)
- Forward pass signature and tensor shapes

### Image Preprocessing
- Resize strategy, normalization constants, tiling/slicing

### Special Tokens
- Image/video placeholder tokens and insertion rules

### Custom Components
- Non-standard attention, positional encodings, resamplers, etc.

## Optimum-Intel Export Analysis
### Exported Models
| File | Purpose | Inputs (name, shape, dtype) | Outputs (name, shape, dtype) |
|------|---------|----------------------------|------------------------------|
| openvino_vision_embeddings_model.xml | Vision encoder | ... | ... |
| openvino_language_model.xml | LLM | ... | ... |

## GenAI Enablement Design
### Closest Existing Implementation
- Which GenAI model class to base on and why

### Required Changes
1. New files to create (`<model_type>/classes.hpp`, `<model_type>/classes.cpp`)
2. VisionEncoder subclass design
3. IInputsEmbedder subclass design
4. VLMConfig / ProcessorConfig additions
5. Factory registration points (`vision_encoder.cpp`, `inputs_embedder.cpp`, `vlm_config.hpp`)

### Gaps and Risks
- Components not covered by existing GenAI infrastructure
- Potential performance concerns
```
