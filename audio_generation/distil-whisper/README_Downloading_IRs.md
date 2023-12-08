# Downloading OpenVINO IRs

Prior to running inference on the model, the OpenVINO IRs formats need to be downloaded as below:

```
from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

distil_model_id = "distil-whisper/distil-large-v2"
distil_model_path = Path(distil_model_id.split("/")[-1])

if not distil_model_path.exists():
    ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        distil_model_id, export=True, compile=False
    )
    ov_distil_model.half()
    ov_distil_model.save_pretrained(distil_model_path)
```

Run *main.py* to execute the inference pipeline of the model.