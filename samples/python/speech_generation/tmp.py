from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForTextToSpeechSeq2Seq
from transformers import AutoTokenizer

output_dir = "tts_model"

model = OVModelForTextToSpeechSeq2Seq.from_pretrained("microsoft/speecht5_tts", vocoder="microsoft/speecht5_hifigan", export=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("microsoft/speecht5_tts")
export_tokenizer(tokenizer, output_dir)