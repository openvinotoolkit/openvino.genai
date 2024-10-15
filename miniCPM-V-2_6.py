from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import requests

model_id = "openbmb/MiniCPM-V-2_6"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "(<image>./</image>)\nWhat is unusual on this image?"}], tokenize=False, add_generation_prompt=True)
# image = Image.open(requests.get("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11", stream=True).raw).convert('RGB')
image = Image.open("/home/vzlobin/r/g/g.png").convert('RGB')

model = OVModelForVisualCausalLM.from_pretrained("MiniCPM-V-2_6", trust_remote_code=True)

inputs = processor([prompt], [image], return_tensors="pt")

result = model.generate(**inputs, max_new_tokens=200)

print(processor.tokenizer.batch_decode(result[:, inputs["input_ids"].shape[1]:]))
