SCRIPT_DIR_GENAI_MODULE_PY="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_GENAI_MODULE_PY}

source ../../../../python-env/bin/activate
source ../../../../source_ov.sh

GENAI_ROOT_DIR=${SCRIPT_DIR_GENAI_MODULE_PY}/../../../../openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

cd ${SCRIPT_DIR_GENAI_MODULE_PY}

model_dir=${SCRIPT_DIR_GENAI_MODULE_PY}/../../cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov/
prompt="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
# prompt="A chinese man with white T-shirt and blue jeans, standing in the forest, draw the light and shadow of the scene clearly, photo taken by Nikon D850, high resolution, detailed texture, draw full person"

python md_image_generation.py --model_path ${model_dir} --prompt "${prompt}" --device CPU --enable_tiling
# python module_pipeline_z_image.py ${model_dir} "${prompt}" CPU
