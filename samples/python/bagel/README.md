# BAGEL Python sample

`bagel.py` runs [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) with `openvino_genai.BagelPipeline`:
visual question answering (`vqa`), text-to-image generation (`t2i`) and image editing (`edit`) with one model.

## Download and convert the model

BAGEL is exported with optimum-intel from its transformers checkpoint. Install the export dependencies, then convert
the original checkpoint and export it:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT
python -m transformers.models.bagel.convert_bagel_weights_to_hf --input_dir BAGEL-7B-MoT --output_dir BAGEL-7B-MoT-hf
optimum-cli export openvino -m BAGEL-7B-MoT-hf --task image-text-to-text --weight-format int4 --sym --group-size 128 \
    --ratio 1.0 --backup-precision int8_sym --group-size-fallback adjust BAGEL-7B-MoT-ov-int4
```

`--weight-format int4` compresses the language model only; use `--weight-format fp16` for an uncompressed model.

> [!NOTE]
> BAGEL needs transformers and optimum-intel versions that include it (huggingface/optimum-intel#TBD).

## Run

Install the runtime dependencies:

```sh
pip install -r ../../deployment-requirements.txt
```

The tokenizer is exported next to the models, so the model directory is passed for both arguments:

```sh
# visual question answering
python bagel.py BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 CPU vqa "Describe this image in one sentence." --image cats.jpg

# text-to-image, 512x512
python bagel.py BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 GPU t2i "A red bicycle leaning against a blue wall, photo" \
    --steps 25 --out bicycle.png

# image editing, the output keeps the VAE resolution of the input image
python bagel.py BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 NPU edit "Make the bicycle green" --image bicycle.png \
    --steps 25 --out bicycle_green.png
```

Options: `--steps` denoising steps (default 50), `--size` text-to-image side (default 512), `--cfg-text` / `--cfg-img`
guidance scales, `--seed` noise seed, `--max-new-tokens` answer length, `--repeat` to time repeated runs, `--props
"K=V;..."` extra compile properties. The sample prints the compile and per-stage inference times.

On NPU the language, vision and VAE models run through NPUW (`NPUW_BAGEL`), which compiles a static variant of a model
the first time an input size is used, so the first run is slower than the next ones.
