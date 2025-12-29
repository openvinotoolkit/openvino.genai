SCRIPT_DIR_EXAMPLE_OV_CPP_RUN="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

source ../../../../python-env/bin/activate
UBUNTU_VER=$(lsb_release -rs | cut -d. -f1)
source ../../../../openvino_toolkit_ubuntu${UBUNTU_VER}_2025.4.0.20398.8fdad55727d_x86_64/setupvars.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

./build/module_genai_ut_app "ut_pipelines" "./ut_pipelines/Qwen2.5-VL-3B-Instruct/config.yaml"
