#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sounddevice as sd
import torch.nn.functional as F
import torchaudio
from scipy.io.wavfile import write
from speechbrain.pretrained import SpeakerRecognition

# Settings
duration = 5  # seconds
sample_rate = 16000  # Hz
output_file = "your_audio.wav"

print(f"Recording for {duration} seconds...")

# Record audio
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()

# Save to WAV file
write(output_file, sample_rate, recording)

print(f"Saved recording to {output_file}")

# Load your WAV file
signal, fs = torchaudio.load("your_audio.wav")
assert fs == 16000, "Frame rate must be 16 KHz"

# Load the pre-trained speaker embedding model (x-vector)
# based on https://huggingface.co/mechanicalsea/speecht5-vc/blob/main/manifest/utils/prep_cmu_arctic_spkemb.py
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
)

# Extract x-vector embedding
embedding = model.encode_batch(signal)
embedding = F.normalize(embedding, dim=2)
embedding = embedding.squeeze().cpu().numpy().astype("float32")

embedding.tofile("speaker_embedding.bin")
