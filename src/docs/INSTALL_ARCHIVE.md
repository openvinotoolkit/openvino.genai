### OpenVINO™ GenAI: Archive Installation

To install the GenAI flavor of OpenVINO from an archive file, follow the standard installation steps for your system
but instead of using the vanilla package file, download the one with OpenVINO GenAI:

- Ubuntu 24.04
    <!-- TODO Update link to GenAI archive -->
    ```sh
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.1/linux/l_openvino_genai_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_genai_2024.1.0.tgz
    tar -xf openvino_genai_2024.1.0.tgz
    sudo mv l_openvino_genai_toolkit_ubuntu24_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_genai_2024.1.0
    ```

- Ubuntu 22.04
    <!-- TODO Update link to GenAI archive -->
    ```sh
    curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.1/linux/l_openvino_genai_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_genai_2024.1.0.tgz
    tar -xf openvino_genai_2024.1.0.tgz
    sudo mv l_openvino_genai_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_genai_2024.1.0
    ```

For other operating systems, please refer to the guides in documentation:
- [Install OpenVINO GenAI Archive for Windows]() <!-- TODO Add link to docs -->
- [Install OpenVINO GenAI Archive for macOS]()<!-- TODO Add link to docs -->
