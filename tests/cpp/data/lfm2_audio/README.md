# LiquidAI audio reference data

The binary files are little-endian F32 arrays, generated independently of OpenVINO.
`manifest.json` records the publisher revision, source hashes, preprocessing settings
and shapes. Preprocessing cases cover 160, 1281 and 16000 waveform samples.
ISTFT cases cover 1, 18 and 37 spectral frames with nonzero magnitude and phase.

Regenerate with `python generate_reference.py --liquid-audio <checkout> --config
<LFM2.5-Audio-1.5B/config.json>`, using NumPy and PyTorch. The checkout must be
`19e65845923a7f136442c95137884ec61eb386aa` from `Liquid4All/liquid-audio`.
The config is from `LiquidAI/LFM2.5-Audio-1.5B` revision
`c362a0625dfe45aa588dce5f0ada28a7e5707628`. No model weights are required.

`test_gguf_lfm2_audio.py` additionally runs the real speech bundle when
`OV_LFM2_AUDIO_BUNDLE` is set; it otherwise records an explicit skip.
