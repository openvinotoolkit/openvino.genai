# PYTHON API SAMEPLE

#### How to run

```
./run.sh
```

#### How to run module pipeline with torch:

Download model, password: `intel123`
```bash
scp -r ziniu@lic-code-vm13:/home/ziniu/web_files/models/Qwen2.5-VL-3B-Instruct/torch ../../cpp/module_genai/ut_pipelines/Qwen2.5-VL-3B-Instruct/
```
Run test:

```bash
source ../../../../python-env/bin/activate
pip install -r requirements.txt
bash run_pipeline_with_torch.sh
```

#### How to run zimage transfomer test

Download the model, password: `intel123`
```bash
scp -r ziniu@lic-code-vm13:/home/ziniu/web_files/models/Z-Image-Turbo-fp16-ov ../../cpp/module_genai/ut_pipelines/
```

Run test:
```bash
source ../../../../python-env/bin/activate
pip install -r requirements.txt
bash run_module_pipeline_z_image.sh
```

The output image is `zimage_denoiser_loop_output.png`