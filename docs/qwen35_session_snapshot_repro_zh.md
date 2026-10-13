# 复现指南：Qwen3.5（KV + Linear-Attention 混合模型）会话快照 save/restore

> 目标：在一台**全新机器**上，从源码构建 `openvino.genai`（`wzx_dev_la` 分支），并端到端复现
> Qwen3.5 混合模型的"关机前保存会话、开机后恢复续写"功能，包括正确性验证和性能对比。
> 功能本身的 API 用法见 [session_snapshot_save_restore_zh.md](session_snapshot_save_restore_zh.md)，
> 本文只关心"怎么从零搭出一套能跑这个功能的环境"。

## 0. 前置条件

- Windows（本文所有步骤都在 Windows 上验证过；Linux/macOS 理论上也能跑，但未验证）
- Python 3.10+（本文以 3.12 为例）
- CMake（`pip install cmake` 即可，不需要单独装）
- C++ 编译器：Visual Studio（MSVC），需要支持 C++17
- Git

**不需要**从源码构建 OpenVINO Runtime 本身——直接用 pip 安装的 OpenVINO 即可，省掉几个小时的编译时间。

## 1. 获取源码

```powershell
git clone <你的 openvino.genai 仓库地址> openvino.genai
cd openvino.genai
git checkout wzx_dev_la
```

如果这个分支已经合并进主干，跳过 `git checkout` 即可。

## 2. 准备 Python 环境

```powershell
python -m pip install --upgrade pip
python -m pip install cmake openvino openvino-tokenizers packaging
python -m pip install -r src\bindings\python\wheel\requirements-dev.txt
python -m pip install -r tools\llm_bench\requirements.txt
```

说明：
- `openvino` 这个 pip 包提供的是**运行时**（推理引擎），我们接下来要从源码构建的是
  `openvino_genai`（这个功能所在的库）。两者版本不需要完全对应，但差太多可能有 ABI 不兼容的
  风险——如果后面编译/运行出现奇怪的崩溃，先检查这里的 OpenVINO 版本。
- `tools\llm_bench\requirements.txt` 里也会装一份 pip 版的 `openvino_genai`，这是正常的，后面
  跑 `benchmark.py` 时会被我们自己编译的版本覆盖（见第 4 步）。

## 3. 配置并编译 openvino_genai

```powershell
cmake -S . -B build_bench
cmake --build build_bench --config RelWithDebInfo --target openvino_genai
cmake --build build_bench --config RelWithDebInfo --target py_openvino_genai
```

- 第一次 `cmake -S . -B build_bench` 配置大概需要几分钟，会自动找到 pip 安装的 OpenVINO
  （`OpenVINO_DIR` 指向 `<python>/Lib/site-packages/openvino/cmake`），不需要手动指定。
- 两个 `--build` 命令分别编译 C++ 库本体和 Python 绑定，各自可能需要十几到几十分钟，取决于机器配置。
- 如果你之前已经跑过一次，之后只改了 `.cpp`/`.hpp` 文件（没有新增/删除源文件），直接重新跑
  `--build` 即可，不需要重新 `cmake -S`；如果新增/删除了源文件（比如这个功能用到的
  `session_snapshot.cpp`），必须重新跑一次 `cmake -S . -B build_bench` 让 CMake 重新扫描源文件
  列表，否则新文件不会被编译进去。

编译完成后，`build_bench\openvino_genai\` 目录下应该有 `openvino_genai.dll`、
`py_openvino_genai.cp3xx-win_amd64.pyd`、`__init__.py` 等文件——这就是一个完整的、可直接
`import openvino_genai` 的 Python 包目录。

## 4. 准备 Qwen3.5（KV + Linear-Attention 混合）模型

这个功能需要一个**原生支持 linear attention 的 Qwen3.5 混合模型**，目录下应该同时包含：

```
openvino_language_model.xml / .bin        <- 语言模型本体（含 linear_attention + full_attention 层）
openvino_text_embeddings_model.xml / .bin  <- 文本向量化模型（即使纯文本也会用到）
openvino_tokenizer.xml / .bin
openvino_detokenizer.xml / .bin
config.json   (architectures 里应为 Qwen3_5ForConditionalGeneration, model_type 为 qwen3_5)
```

如果你手上已经有这样一份模型目录，直接跳到第 5 步，把下面命令里的 `<MODEL_DIR>` 换成你的路径。

如果还没有，需要从 HuggingFace 上对应的 Qwen3.5 系列 checkpoint 自行导出（具体用哪个 checkpoint
取决于你能访问到的模型授权，本文不代为指定）：

```powershell
optimum-cli export openvino --model <你的 Qwen3.5 模型 ID 或本地 PyTorch 权重路径> \
    --trust-remote-code --weight-format int4 <MODEL_DIR>
```

导出完成后，用下面的命令确认一下这确实是一个"混合"模型（能看到 `linear_attention` 和
`full_attention` 两种层交替出现）：

```powershell
Select-String -Path "<MODEL_DIR>\config.json" -Pattern "linear_attention|full_attention" | Select-Object -First 8
```

## 5. 跑正确性 + 性能复现（用 `benchmark.py`，推荐）

这是最省事的复现方式：不需要额外编译测试套件，直接用仓库自带的 `benchmark.py`。

```powershell
cd tools\llm_bench

# 第一步：正常跑一次完整 prefill，作为"不走 restore"的性能基线
python benchmark.py -m <MODEL_DIR> -d CPU -pf session_snapshot_long_prompt.jsonl -n 0 -ic 1 `
    --disable_prompt_permutation --cb_config session_snapshot_cb_config.json

# 第二步：跑几步后"保存会话"（模拟关机前的那一刻，不是跑满整个生成）
python benchmark.py -m <MODEL_DIR> -d CPU -pf session_snapshot_long_prompt.jsonl -n 0 -ic 1 `
    --disable_prompt_permutation --cb_config session_snapshot_cb_config.json `
    --session_snapshot_save snapshot_q35.bin --session_snapshot_save_steps 3

# 第三步（模拟重新开机）：从快照恢复，继续生成，走标准 benchmark 指标输出
python benchmark.py -m <MODEL_DIR> -d CPU -p "ignored" -n 0 -ic 1 `
    --disable_prompt_permutation --cb_config session_snapshot_cb_config.json `
    --session_snapshot_restore snapshot_q35.bin
```

**预期结果**：第一步和第三步都会打印一行 `First token latency: ... ms`。第三步（走 restore）的数字
应该比第一步（完整 prefill）**小一个数量级**，这就是这个功能的核心收益。本次验证在 CPU 上的真实
数据是 `135602 ms → 5788 ms`（约 23 倍）；你的机器性能不同，具体数字会不一样，但"restore 明显更
快"这个结论应该是一致的。

第二步执行完之后，终端里应该能看到一行：

```
[ INFO ] [...] Saved session snapshot to '...\snapshot_q35.bin' after 3 steps in ... s
```

如果看到类似 `the request finished naturally within 3 steps` 的报错，说明 prompt 太短、3 步内就
生成完了——把 `session_snapshot_long_prompt.jsonl` 换成更长的 prompt，或者把
`--session_snapshot_save_steps` 调大一点。

## 6. 跑正确性验证（用 C++ 测试，可选/进阶）

如果你想要更严格的"续写内容和未打断基线完全一致"级别的验证（而不只是看性能数字），可以编译并跑
gtest 测试套件。这一步需要额外给 CMake 加上 `-DENABLE_TESTS=ON`，首次配置时会从 GitHub 拉取
googletest 源码（需要网络；如果在公司内网访问 GitHub 受限，需要配置代理环境变量
`HTTP_PROXY`/`HTTPS_PROXY`）：

```powershell
cmake -S . -B build_bench -DENABLE_TESTS=ON
cmake --build build_bench --config RelWithDebInfo --target tests_continuous_batching
```

运行时需要手动把编译产物和 OpenVINO 运行时 DLL 加到 `PATH`（`benchmark.py` 会自动处理这件事，但
直接跑 `.exe` 不会）：

```powershell
cd build_bench\bin\RelWithDebInfo
$env:PATH = "$PWD\..\..\openvino_genai;$(python -c 'import sys; print(sys.prefix)')\Lib\site-packages\openvino\libs;" + $env:PATH
$env:OFFLOAD_E2E_VLM_MODEL = "<MODEL_DIR>"
$env:OFFLOAD_E2E_DEVICE = "CPU"
.\tests_continuous_batching.exe --gtest_filter="TestSessionSnapshotEndToEnd.HybridModel*"
```

预期两个测试都通过：

```
[ RUN      ] TestSessionSnapshotEndToEnd.HybridModelRestoredRequestContinuesIdenticallyToUninterruptedRun
[       OK ] ...
[ RUN      ] TestSessionSnapshotEndToEnd.HybridModelRestoreAvoidsFullPrefillLatency
[ PERF     ] TTFT without session restore (full re-prefill): ... ms
[ PERF     ] TTFT with session restore (no re-prefill):      ... ms
[       OK ] ...
[  PASSED  ] 2 tests.
```

第一个测试会在内部自动完成"跑一遍完整基线 -> 打断并保存 -> 新建 pipeline 恢复 -> 续写"，并断言
续写出来的 token 序列和未打断的基线**逐字节完全一致**——这是比看性能数字更强的正确性保证。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 运行 `.exe` 直接退出，退出码 `-1073741515`，没有任何输出 | 缺少 OpenVINO 运行时 DLL 的 `PATH`，见第 6 步的 `$env:PATH` 设置 |
| `ModuleNotFoundError: No module named 'openvino_genai'` | 没有把 `build_bench` 加到 `sys.path`/`PYTHONPATH`；用 `benchmark.py` 跑则不需要手动处理（它内部的 `configure_local_runtime()` 会自动找 `<仓库根目录>/build_bench`） |
| 新增/删除了源文件后编译没生效 | 需要重新跑 `cmake -S . -B build_bench` 让 CMake 重新扫描源文件列表（`tests/cpp/CMakeLists.txt` 用的是 `file(GLOB *.cpp)`） |
| `[ERROR] BlockManager leaked sequence block tables` / `BlockAllocator leaked blocks` 警告 | 这是测试里**故意**中途打断请求（模拟进程被杀掉）、不走正常收尾流程导致的预期日志，不是真正的内存泄漏，可以忽略 |
| `the request finished naturally within N steps, test setup is invalid` | 测试用的 prompt 太短或 `--session_snapshot_save_steps`/`infer_count` 配置不当，在还没来得及"中途打断"之前生成就自然结束了；换更长的 prompt 或调整步数 |
| 续写内容和基线不一致 / 抛出 layout mismatch 异常 | 确认 save 和 restore 用的是**同一个模型目录、同一个推理设备**；这个功能的快照文件里存了模型版式指纹，换模型/换设备会被明确拒绝而不是产生错误结果 |

如果以上步骤都跑通，说明你的环境已经完整复现了 Qwen3.5 的会话保存/恢复功能；更详细的 API 说明、
使用限制、异常列表见 [session_snapshot_save_restore_zh.md](session_snapshot_save_restore_zh.md)。
