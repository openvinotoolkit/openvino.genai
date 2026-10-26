from pathlib import Path

import openvino as ov
import openvino_genai
from transformers import AutoProcessor


model_dir = Path("sandbox/lfm3-vl-1.9b-rl-Intel-int8")
core = ov.Core()
for xml in sorted(model_dir.glob("openvino_*.xml")):
    if "tokenizer" in xml.name:
        continue
    model = core.read_model(xml)
    print(xml.name)
    for port in model.inputs:
        print(" IN", port.any_name, port.partial_shape, port.element_type)
    for port in model.outputs:
        print(" OUT", port.any_name, port.partial_shape, port.element_type)
    print(" STATES", len(model.get_variables()))

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = openvino_genai.Tokenizer(model_dir)
for token in ("<|image|>", "<|video|>", "<|vision_start|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>"):
    print(token, processor.tokenizer.encode(token, add_special_tokens=False),
          tokenizer.encode(token, add_special_tokens=False).input_ids.data.tolist())
print("GenAI:", openvino_genai.__file__)
