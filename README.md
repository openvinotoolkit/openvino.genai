GenAI contains pipelines that implement image and text generation tasks. The implementation exploits OpenVINO capabilities to optimize the pipelines. Each sample covers a family of models and suggests that its implementation can be modified to adapt for a specific need.

> [!NOTE]
> This project is not for production use.

Every pipeline requires https://github.com/openvinotoolkit/openvino for C++ to be installed.

Build the pipelines and `user_ov_extensions`

```sh
git submodule update --init
mkdir ./build/ && cd ./build/
source <OpenVINO dir>/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Release ../ && cmake --build ./ --config Release -j
```

To enable non ASCII characters for Windows cmd open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
