# BAGEL C++ sample

`bagel` runs [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) with `ov::genai::BagelPipeline`:
visual question answering (`vqa`), text-to-image generation (`t2i`) and image editing (`edit`) with one model.

## Download and convert the model

BAGEL is exported with optimum-intel from its transformers checkpoint. Until BAGEL support is merged in
transformers and optimum-intel ([huggingface/optimum-intel#2126](https://github.com/huggingface/optimum-intel/pull/2126)),
install them from these commits:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
pip install "transformers @ git+https://github.com/padatta/transformers.git@38b3e74123a005abfd071d2709bbbb017a7919e1"
pip install --no-deps "optimum-intel @ git+https://github.com/padatta/optimum-intel.git@b13d14169be2f04a06a5251931cba624170849d4"
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT
python -m transformers.models.bagel.convert_bagel_weights_to_hf --input_dir BAGEL-7B-MoT --output_dir BAGEL-7B-MoT-hf
optimum-cli export openvino -m BAGEL-7B-MoT-hf --task image-text-to-text --weight-format int4 --sym --group-size 128 \
    --ratio 1.0 --backup-precision int8_sym --group-size-fallback adjust BAGEL-7B-MoT-ov-int4
```

`genai-cli` (`pip install openvino-genai[export]`) exports the converted checkpoint natively, without optimum-intel,
and defaults to the same INT4 settings:

```sh
genai-cli export -m BAGEL-7B-MoT-hf --weight-format int4 BAGEL-7B-MoT-ov-int4
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html)
to build the samples. Input images can be JPEG / PNG / BMP; generated images are written to `image_0.bmp`:

```sh
./bagel BAGEL-7B-MoT-ov-int4 CPU vqa "Describe this image in one sentence." cats.jpg
./bagel BAGEL-7B-MoT-ov-int4 CPU t2i "A red bicycle leaning against a blue wall, photo"
./bagel BAGEL-7B-MoT-ov-int4 CPU edit "Make the bicycle green" bicycle.png
```

On NPU the language, vision and VAE models run through NPUW (`NPUW_BAGEL`), which compiles a static variant of a model
the first time an input size is used, so the first run is slower than the next ones.
