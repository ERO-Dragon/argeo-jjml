# JJML b9717 / MTP 增量更新说明

本文档记录本次底层升级相对旧版 JJML 的增量变化，供真正调用 JJML 的上层使用。重点是三个互相独立的方向：文本推理的 MTP、视觉/mmproj 的显式 opt-in、以及 Vulkan 时间片调度。

调用粒度上，除视觉/mmproj 是加载前或进程级 opt-in 外，MTP 和时间片优先级都按一次生成请求的运行配置使用；其中时间片调度底层是 Vulkan backend 全局状态，上层应在请求开始前设置需要的优先级。

## 1. 更新范围

- `llama.cpp` 升级到 b9717 对应提交，当前保持官方源码干净状态。
- `ggml` 使用我们的 `ggml-game` 分支，包含 Vulkan 时间片调度改动。
- `whisper.cpp` 同步到新提交，当前没有额外本地修改。
- JJML 的 Java/JNI 层已适配新版 llama.cpp / ggml 的构建、加载和推理路径。
- Windows 一键构建脚本继续面向 MSVC v142、Java 21、Vulkan。

当前子模块状态以本地 `git submodule status` 为准：

- `native/tp/llama.cpp`: `8141e730f1598780c19b153e0e212ed70a672c53`
- `native/tp/ggml`: `a6b52e09c1ee0fc1cbe8347d37a80f4b41675936`
- `native/tp/whisper.cpp`: `86c40c3bd6fc86f1187fb751d111b49e0fc18e84`

## 2. 文本推理与 MTP

MTP 已作为显式 speculative decoding 路径接入 JJML。它不是模型加载时永久开启的开关，而是一次生成请求使用 `LlamaCppSpeculativeProcessor` 时才启用的推理路径。普通文本生成仍可继续走原有 processor。

对外 API：

- `LlamaCppModel.supportsMtp()`: 判断当前 GGUF 是否声明了 MTP/NextN 能力。
- `LlamaCppModel.getMtpLayerCount()`: 返回模型元数据里声明的 MTP 层数，只表示模型能力。
- `SpeculativeParams.none()`: 显式表示不开 speculative decoding。
- `SpeculativeParams.draftMtp()`: 使用默认 `draftMax = 3` 的入门配置。
- `SpeculativeParams.draftMtp(int draftMax)`: 推荐上层生产环境使用的入口，由上层传入本次请求要测试或使用的 draft 数。
- `SpeculativeParams.draftMtp(int draftMax, float draftSplitProbability, float draftMinProbability)`: 给 benchmark 或专家调参使用。
- `SpeculativeParams.adjustTargetContextParams(ContextParams params, int maxDraftMax)`: 创建 target context 前预留最大 MTP draft 窗口。
- `SpeculativeParams.adjustTargetContextParams(ContextParams params)`: 按当前 `SpeculativeParams` 自动调整 target context。
- `LlamaCppSpeculativeProcessor.begin(...)` / `read(...)` / `readArray(...)`: 执行一次 MTP 生成。
- `LlamaCppSpeculativeProcessor.getStats()`: 获取本次 speculative decoding 统计。

`getMtpLayerCount()` 和 `draftMax` 不等价：

- `getMtpLayerCount()` 是模型文件声明的静态能力，用来判断模型是否支持 MTP。
- `draftMax` 是本次生成请求实际尝试预测的 draft token 数，是上层需要 benchmark 和选择的运行参数。
- 上层可以先用 `getMtpLayerCount()` 做能力检测，再逐个测试 `draftMax = 1..N`，记录速度、接受率、显存和输出质量。

与 Qwen3.5 官方推荐方式的对应关系：

- Qwen3.5 官方说明里，MTP 是基于 NextN / qwen3_next_mtp 的 speculative decoding。JJML 这里走的是 llama.cpp 的 MTP/NextN draft path，语义上对齐。
- SGLang 的 `--speculative-num-draft-tokens`、vLLM 的 `num_speculative_tokens` 对应 JJML 里的主要运行参数 `draftMax`。
- SGLang 的 `--speculative-num-steps`、`--speculative-eagle-topk` 是 SGLang 自身 serving 实现的调度/树展开参数；JJML 当前没有直接暴露等价参数，只暴露 llama.cpp 当前 MTP 路径真实使用的 draft window 和概率阈值。

推荐上层流程：

1. 加载模型后调用 `model.supportsMtp()`。
2. 若不支持，继续使用普通推理路径。
3. 若支持，选择计划测试的最大 `maxDraftMax`。
4. 创建 context 前调用 `SpeculativeParams.adjustTargetContextParams(baseContextParams, maxDraftMax)`。
5. 每次请求按需要创建 `SpeculativeParams.draftMtp(draftMax)`。
6. 请求结束后读取 `SpeculativeStats`，用于自动调参和回退。

统计字段：

- `promptTokens`: prompt token 数。
- `generatedTokens`: 实际生成 token 数。
- `draftedTokens`: draft 侧尝试生成的 token 数。
- `acceptedDraftTokens`: 被 target 接受的 draft token 数。
- `acceptanceRate()`: 接受率。
- `tokensPerSecond()`: 按 decode 阶段统计的生成速度。

## 3. MTP 调参边界

上层只需要把 `draftMax` 作为主要可调参数暴露给自动测试逻辑。`draftSplitProbability` 和 `draftMinProbability` 已保留给 benchmark 或实验，但不建议普通业务 UI 直接暴露。

建议上层缓存最优配置时至少区分：

- 模型文件和量化类型。
- backend 类型，当前主推 Vulkan。
- 显卡型号、驱动版本、可用显存。
- context 长度、batch/ubatch、是否启用时间片调度。

MTP 在不同硬件和 backend 上的收益并不固定。Vulkan 路径下目前已经可用，但不应假设所有模型都能稳定获得 50% 提速。自动选择最优 `draftMax` 是必要的，尤其是游戏内运行环境会受功耗模式、后台负载、驱动和显存压力影响。

## 4. 视觉 / mmproj

视觉能力已经改为显式 opt-in。纯文本推理默认不加载 mtmd，也不要求 mmproj。

对外 API：

- `MtmdNative.isEnabled()`: 判断当前进程是否启用了 mtmd/vision。
- `MtmdNative.enable()`: 在 mtmd native library 加载前启用。
- `MtmdNative.disable()`: 在 mtmd native library 加载前显式禁用。
- 系统属性 `jjml.mtmd.enabled=true`: 启用 mtmd。
- 系统属性 `jjml.vision.enabled=true`: 兼容视觉语义的启用入口。
- `new MtmdContext(model, mmprojPath, useGpu, threads)`: 加载指定 mmproj。
- `new MtmdProcessor(context, samplerChain, mtmdContext).transcribe(prompt, bitmaps)`: 执行图文输入。

注意事项：

- 只跑文本时不需要设置任何视觉相关属性。
- 这对应 Qwen3.5 官方 serving 里的 text-only / language-model-only 用法：跳过视觉编码器和多模态 profiling，把资源留给文本 KV cache。
- 一旦 mtmd native library 已经加载，不能再通过 `enable()` / `disable()` 切换。
- 视觉模型的显存占用来自 mmproj 和图像处理路径，和 MTP 的 draft 窗口是两套独立机制。

## 5. Vulkan 时间片调度

Vulkan 时间片调度是 ggml 后端能力，和 MTP 没有参数关系。它对外只暴露简单的 `0.0..1.0` 推理优先级。

对外 API：

- `GgmlVulkanScheduler.isSupported()`: 当前加载的 Vulkan 后端是否支持我们的调度扩展。
- `GgmlVulkanScheduler.setInferencePriority(float priority)`: 设置推理优先级。
- `GgmlVulkanScheduler.getStats()`: 读取调度统计。
- `GgmlVulkanScheduler.resetStats()`: 重置统计。

语义：

- `priority = 0.0`: 最大程度让位给游戏。
- `priority = 1.0`: 不进行时间分片，回到原始 Vulkan 执行路径。
- 内部映射为 11 个 level，并包含自适应 chunk 参数；上层不需要也不应该直接控制 raw chunk、pause 或 abort。

## 6. 打包与运行目录

`build-v142-vulkan.bat` 现在会在构建成功后刷新目标运行目录：

- `LlamaDependencies\jar`
- `LlamaDependencies\native`

刷新策略：

- 复制前先清空 `jar` 和 `native` 子目录。
- 复制 JJML jar 和 multimedia jar。
- 复制 JJML JNI DLL。
- 复制 ggml / llama / whisper 相关 native DLL。
- 复制 `build\bin` 下新版 llama.cpp common runtime 生成的辅助 DLL。

新版运行包必须包含 `llama-common.dll`、`parakeet.dll` 等辅助依赖。缺这些 DLL 时，单独的 `llama.dll` 或 JJML JNI DLL 可能能编译出来，但运行时加载会失败。

## 7. 已验证结果

已执行：

```bat
.\build-v142-vulkan.bat --quiet-warnings
```

构建目标：

- MSVC v142
- Java 21
- Vulkan enabled
- Release

目标目录烟测已经从 `LlamaDependencies` 运行通过，不依赖 build 临时目录：

- 日志：`build\target-folder-mtp-smoke.out.txt`
- 错误输出：`build\target-folder-mtp-smoke.err.txt`
- `JAVA_EXIT=0`
- 成功加载 `LlamaDependencies\native\ggml-vulkan.dll`
- 成功加载 `D:\AIPROJECT\Qwen3.5-2B-Q4_K_M.gguf`
- `model.supportsMtp=true`
- `model.mtpLayerCount=1`
- MTP 生成 64 tokens 成功
- `draftedTokens=65`
- `acceptedDraftTokens=29`
- `acceptance=0.4462`

## 8. 设计取舍与约束

- 官方 llama.cpp 已支持 MTP，但 JJML 不是直接运行官方 server。JJML 仍然需要自己对接 Java API、JNI 生命周期、context 参数、sampler、batch 输出、统计和 DLL 打包。
- 当前不修改官方 llama.cpp 源码来补 MTP；必要的 graph reserve 和上下文校验放在 JJML JNI 层。
- JJML 的 MTP draft context 当前保留 `n_rs_seq = draftMax`。这是为了在没有完整复刻官方 server checkpoint fallback 的情况下，保证 recurrent-state 回滚路径更稳健。
- 旧的 backend load probe 曾出现崩溃，不作为本版可用性的判断依据；真实 JJML 目标目录加载和 MTP 生成烟测已通过。
- MTP 可能改变逐 token 采样路径，输出不需要和普通 greedy/default 逐字一致。上层质量评估应关注是否遵循提示词、回答是否通顺正确、是否出现明显退化。
- 如果上层要做一键最优配置，建议以普通推理为 baseline，再分别测 `draftMax`、时间片 priority、context/batch 参数，最终按速度、接受率、显存和质量综合选择。
