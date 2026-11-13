# BAGEL C++ sample

`bagel` runs [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) with `ov::genai::BagelPipeline`:
visual question answering (`vqa`), text-to-image generation (`t2i`) and image editing (`edit`) with one model.

## Download and convert the model

BAGEL is exported with optimum-intel from its transformers checkpoint:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT
python -m transformers.models.bagel.convert_bagel_weights_to_hf --input_dir BAGEL-7B-MoT --output_dir BAGEL-7B-MoT-hf
optimum-cli export openvino -m BAGEL-7B-MoT-hf --task image-text-to-text --weight-format int4 --sym --group-size 128 \
    --ratio 1.0 --backup-precision int8_sym --group-size-fallback adjust BAGEL-7B-MoT-ov-int4
```

> [!NOTE]
> BAGEL needs transformers and optimum-intel versions that include it (huggingface/optimum-intel#TBD).

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html)
to build the samples. The tokenizer is exported next to the models, so the model directory is passed for both
arguments; input images can be JPEG / PNG / BMP, generated images are written as BMP:

```sh
./bagel BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 CPU vqa "Describe this image in one sentence." --image cats.jpg
./bagel BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 GPU t2i "A red bicycle leaning against a blue wall, photo" --steps 25 --out bicycle.bmp
./bagel BAGEL-7B-MoT-ov-int4 BAGEL-7B-MoT-ov-int4 NPU edit "Make the bicycle green" --image bicycle.bmp --steps 25 --out green.bmp
```

Run `./bagel` without arguments for the list of options. On NPU the language, vision and VAE models run through NPUW
(`NPUW_BAGEL`), which compiles a static variant of a model the first time an input size is used, so the first run is
slower than the next ones.
