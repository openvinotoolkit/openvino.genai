# BAGEL Python sample

`bagel.py` runs [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) with `openvino_genai.BagelPipeline`:
visual question answering (`vqa`), text-to-image generation (`t2i`) and image editing (`edit`) with one model.

## Download and convert the model

BAGEL is exported with optimum-intel from its transformers checkpoint. Until BAGEL support is merged in
transformers and optimum-intel ([huggingface/optimum-intel#2126](https://github.com/huggingface/optimum-intel/pull/2126)),
install them from these commits:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
pip install "transformers @ git+https://github.com/padatta/transformers.git@38b3e74123a005abfd071d2709bbbb017a7919e1"
pip install --no-deps "optimum-intel @ git+https://github.com/padatta/optimum-intel.git@2135bb984870b131a5403ad6afca1bf53fba990d"
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT
python -m transformers.models.bagel.convert_bagel_weights_to_hf --input_dir BAGEL-7B-MoT --output_dir BAGEL-7B-MoT-hf
optimum-cli export openvino -m BAGEL-7B-MoT-hf --task image-text-to-text --weight-format int4 --sym --group-size 128 \
    --ratio 1.0 --backup-precision int8_sym --group-size-fallback adjust --model-kwargs '{"num_frames": 3}' \
    BAGEL-7B-MoT-ov-int4
```

`--weight-format int4` compresses the language model only; use `--weight-format fp16` for an uncompressed model.
`--model-kwargs '{"num_frames": 3}'` adds the 3-frame vision and VAE models of `frames`.

`genai-cli` (`pip install openvino-genai[export]`) exports the converted checkpoint natively, without optimum-intel,
and defaults to the same INT4 settings:

```sh
genai-cli export -m BAGEL-7B-MoT-hf --weight-format int4 BAGEL-7B-MoT-ov-int4
```

## Run

Install the runtime dependencies:

```sh
pip install -r ../../deployment-requirements.txt
```

```sh
# visual question answering
python bagel.py BAGEL-7B-MoT-ov-int4 CPU vqa "Describe this image in one sentence." cats.jpg

# text-to-image, writes image.bmp
python bagel.py BAGEL-7B-MoT-ov-int4 CPU t2i "A red bicycle leaning against a blue wall, photo"

# image editing, writes image.bmp at the VAE resolution of the input image
python bagel.py BAGEL-7B-MoT-ov-int4 CPU edit "Make the bicycle green" bicycle.png

# several images, in the order the prompt refers to them; editing outputs the size of the last image
python bagel.py BAGEL-7B-MoT-ov-int4 CPU vqa "What do these images have in common?" cats.jpg dog.jpg bird.jpg
python bagel.py BAGEL-7B-MoT-ov-int4 CPU edit "Put the cat from the first image on the sofa of the second" cat.jpg sofa.jpg
```

The images can also be consecutive frames of a video, oldest first, for example from a robot camera. Say in the prompt
that they are consecutive frames, then ask for the next action (`vqa`) or for the next frame (`edit`, which outputs the
size of the last frame):

```sh
python bagel.py BAGEL-7B-MoT-ov-int4 CPU vqa "The images above are consecutive frames from a robot camera, ordered from oldest to newest. The task of the robot is: put the white mug on the left plate. Based on the motion across the frames, what should the robot arm do next? Answer with one short imperative sentence describing only the next action." frame_0.png frame_1.png frame_2.png
python bagel.py BAGEL-7B-MoT-ov-int4 CPU edit "Generate the next frame of this robot video. In the next frame the robot arm lifts the white mug. Keep the camera viewpoint, the scene and all other objects unchanged." frame_0.png frame_1.png frame_2.png
```

`frames` predicts the next 3 frames as a world model and writes `next_frame_0.bmp` .. `next_frame_2.bmp`. Each
generated frame replaces the oldest one of the window and joins the context as VAE latents only, without a decode and
re-encode; the 3 frames are decoded at the end, in one VAE decoder call (one call per frame on NPU). As the original
OpenVINO BAGEL world model demo, each frame of the window follows the label `Frame 1 (t-2):`, `Frame 2 (t-1):`,
`Frame 3 (t, current):` for 3 input frames; `--frame-labels` sets other ones:

```sh
python bagel.py BAGEL-7B-MoT-ov-int4 CPU frames "Generate the next frame of this robot video. The task of the robot is: put the white mug on the left plate. In the next frame the robot arm lifts the white mug. Keep the camera viewpoint, the scene and all other objects unchanged." frame_0.png frame_1.png frame_2.png
```

`--steps` sets the denoising steps (default 50). Classifier-free guidance runs extra language model passes for every
step: `--cfg-text` (default 4) is the text guidance scale and `--cfg-img` (default 2) the image guidance scale of
`edit` and `frames`, on top of the text guidance. A scale of 1 disables that guidance: `--cfg-text 1` runs one pass
per step and keeps one KV cache, the fastest and least memory-hungry setting, `--cfg-img 1` keeps the text guidance
only. `--num-frames` (default 3) sets the number of generated frames and `--seed` the initial noise.

Keep the guidance on for `frames`: each generated frame conditions the next one, so without guidance the first frame
is already ghosted and the following ones blur out. The prompt should state the task and the next action, as above:

```sh
# 20 steps with text and image guidance (recommended)
python bagel.py BAGEL-7B-MoT-ov-int4 NPU frames "Generate the next frame of this robot video. The task of the robot is: put the white mug on the left plate. In the next frame the robot arm lifts the white mug. Keep the camera viewpoint, the scene and all other objects unchanged." frame_0.png frame_1.png frame_2.png --steps 20 --cfg-text 4 --cfg-img 2
# 20 steps without guidance: about 3x faster and less memory, but blurry frames
python bagel.py BAGEL-7B-MoT-ov-int4 NPU frames "Generate the next frame of this robot video. The task of the robot is: put the white mug on the left plate. In the next frame the robot arm lifts the white mug. Keep the camera viewpoint, the scene and all other objects unchanged." frame_0.png frame_1.png frame_2.png --steps 20 --cfg-text 1 --cfg-img 1
```

An export with the 3-frame models of the demo's batch decode (optimum-intel `--model-kwargs '{"num_frames": 3}'`)
encodes the 3 input frames and decodes the 3 generated frames in one call each, also on NPU.

On NPU the language, vision and VAE models run through NPUW (`NPUW_BAGEL`), which compiles a static variant of a model
the first time an input size is used, so the first run is slower than the next ones. `--props` passes device
properties: `--props NPUW_BAGEL_MAX_LM_VARIANTS=2` caps the compiled language model variants, for hosts with 32 GB of
memory.
