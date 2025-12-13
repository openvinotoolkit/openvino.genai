Please refer to the following blogs for the setup instructions.

https://medium.com/openvino-toolkit/how-to-build-openvino-genai-app-in-c-32dcbe42fa67

## Windows Troubleshooting

### "System Error" or "DLL not found" after compilation

If `chat_sample.exe` fails to start despite a successful build, it may be unable to locate the OpenVINO or TBB libraries. This is common when building with Visual Studio.

**Solution:**

Ensure the required DLLs are present in the executable directory (`Release` or `Debug`). You can copy them from your OpenVINO installation or Python environment:

```cmd
rem Example for copying TBB and OpenVINO libraries
copy <path_to_venv>\Lib\site-packages\openvino\libs\*.dll <path_to_build>\Release\
copy <path_to_venv>\Lib\site-packages\openvino\libs\tbb\*.dll <path_to_build>\Release\


