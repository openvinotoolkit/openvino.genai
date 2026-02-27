# Consumer Sample

Tiny standalone CMake consumer that uses `find_package(MisakiCpp CONFIG REQUIRED)`.

## Quick start with presets

From `cpp/samples/consumer`:

Windows (Visual Studio):

```powershell
cmake --preset windows-vs-debug
cmake --build --preset windows-vs-debug
```

Linux/macOS (Ninja):

```bash
cmake --preset ninja-debug
cmake --build --preset ninja-debug
```

These presets resolve the package via `MisakiCpp_DIR` pointing at:
`cpp/build/install-test/lib/cmake/MisakiCpp`.

## Configure

From `cpp/`:

Windows (PowerShell):

```powershell
cmake -S samples/consumer -B build/consumer -DCMAKE_PREFIX_PATH="${PWD}/build/install-test"
```

Linux/macOS (Bash/Zsh):

```bash
cmake -S samples/consumer -B build/consumer -DCMAKE_PREFIX_PATH="$PWD/build/install-test"
```

## Build

Multi-config generators (Visual Studio/Xcode):

```powershell
cmake --build build/consumer --config Debug
```

Single-config generators (Ninja/Unix Makefiles):

```bash
cmake --build build/consumer
```

## Run

Windows (Visual Studio generator):

```powershell
.\build\consumer\Debug\consumer_example.exe
```

Linux/macOS or single-config build trees:

```bash
./build/consumer/consumer_example
```
