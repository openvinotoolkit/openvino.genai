from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image
import requests
import cv2
import numpy as np
res = 448, 448
im = np.arange(res[0] * res[1] * 3, dtype=np.uint8) % 255
im = im.reshape([*res, 3])
cv2.imwrite("lines.png", im)
model_id = "openbmb/MiniCPM-V-2_6"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
prompt = processor.tokenizer.apply_chat_template([{"role": "user", "content": "(<image>./</image>)\nWhat is unusual on this image?"}], tokenize=False, add_generation_prompt=True)
image = Image.open("/home/vzlobin/r/g/lines.png").convert('RGB')
# image = Image.open(requests.get("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11", stream=True).raw).convert('RGB')
model = OVModelForVisualCausalLM.from_pretrained("MiniCPM-V-2_6", trust_remote_code=True)
inputs = processor([prompt], [image], return_tensors="pt")
result = model.generate(**inputs, max_new_tokens=200)
decoded = processor.tokenizer.batch_decode(result[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(decoded)
with open("ref.txt", "w") as f:
    f.write(decoded)
