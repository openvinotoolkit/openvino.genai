# Samples

Small API examples for the C++ Misaki port.

## Files

- `basic_usage.cpp`
  - Creates an English engine
  - Calls `phonemize_with_tokens(...)`
  - Prints phoneme output and token fields

- `fallback_and_unknown.cpp`
  - Sets `set_unknown_token(...)`
  - Installs `set_fallback_hook(...)`
  - Demonstrates fallback resolution for unknown words

- `consumer/`
  - Tiny standalone CMake consumer using `find_package(MisakiCpp CONFIG REQUIRED)`
  - Useful reference for integrating the installed package into another project
  - Includes `CMakePresets.json` for one-command configure/build

## Build and run

From `cpp/`:

```powershell
cmake --build build --config Debug
```

Or build sample targets directly:

```powershell
cmake --build build --config Debug --target sample_basic_usage sample_fallback_and_unknown
```

Run from `cpp/build/Debug` on Windows:

```powershell
.\sample_basic_usage.exe
.\sample_fallback_and_unknown.exe
```

If IPA symbols still look garbled in `cmd.exe`, ensure your console uses UTF-8 and a Unicode-capable font:

```powershell
chcp 65001
```

Then run the sample again. (The samples also set UTF-8 code page programmatically on Windows.)

You can compile these sample files in your own app/tooling setup by linking against `misaki_cpp` and including `include/`.
