These fixtures execute the retained publisher classes from LiquidAI/LFM3-Omni-2B
revision `ce2458cbce4ad970f92032ed94e3d18d30b04f9c`. `manifest.json` records source
URLs, hashes, Torch version, shapes and sample counts. Regenerate them with:

```sh
python generate_reference.py
```

The generator compiles the original `Lfm3OmniFeatureExtractor` and
`LiquidCodecISTFT` class ASTs. It does not implement a second NumPy feature
extractor or inverse FFT. References cover sub-window inputs, right padding,
multiple encoder chunks, single-frame synthesis and unequal streaming splits.
