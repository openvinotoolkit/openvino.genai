from pathlib import Path
from openvino.runtime import Core
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

#Define model parameters
distil_model_id = "distil-whisper/distil-large-v2"
core = Core()
device = "GPU"
processor = AutoProcessor.from_pretrained(distil_model_id)

#Specify model dir
model_dir = Path(distil_model_id.split("/")[-1])

#Prepare and load dataset
def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = dataset[0]
input_features = extract_input_features(sample)

#Load and compile the model
ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_dir, compile=False
)
ov_distil_model.to(device)
ov_distil_model.compile()

#Run inference
print("Running inference")

predicted_ids = ov_distil_model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(f"Reference: {sample['text']}")
print(f"Result: {transcription[0]}")