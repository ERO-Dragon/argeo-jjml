# GPU Scheduler Game Test Plan

This document defines how to validate the Vulkan GPU coexistence scheduler before enabling it by default in a game runtime.

## Scope

The scheduler is a cooperative limiter for JJML/ggml Vulkan inference. It maps the application priority value to eleven levels:

- `0.0` / `level0`: game priority, minimum non-zero inference budget.
- `1.0` / `level10`: inference priority, scheduler disabled, upstream Vulkan behavior.

It is not a hardware SM percentage guarantee. A passing result means the LLM backend avoids sustained queue pressure and improves game frame-time behavior on tested systems.

## Product Rules

- Do not enable `DISABLE_VULKAN_OBS_CAPTURE` by default.
- Keep `level10` as the exact fallback path for full-speed inference and regression isolation.
- Keep `level0` as a non-zero inference budget. Use explicit pause/abort controls for a fully stopped model.
- Treat the feature as experimental until frame-time data is collected across different GPUs and overlay setups.

## Test Matrix

Run every test with the same model, prompt, graphics settings, world/location, camera path, and inference workload.

| Case | Game graphics | LLM level | OBS/overlay | Required |
| --- | --- | --- | --- | --- |
| Baseline game only | Normal | Off | Off | Yes |
| Baseline game only with shader | Heavy shader | Off | Off | Yes |
| Full inference | Normal | 10 | Off | Yes |
| Full inference with shader | Heavy shader | 10 | Off | Yes |
| Game priority | Normal | 0 | Off | Yes |
| Game priority with shader | Heavy shader | 0 | Off | Yes |
| Mid priority | Heavy shader | 5 | Off | Yes |
| High inference priority | Heavy shader | 9 | Off | Yes |
| Overlay compatibility | Heavy shader | 0, 5, 10 | OBS on | Recommended |
| Driver coverage | Heavy shader | 0, 5, 10 | Off | NVIDIA / AMD / Intel when available |

## Metrics

Collect at least 3 minutes per case after warm-up.

- Average FPS.
- 1% low FPS.
- 0.1% low FPS.
- Frame time average, p95, p99, max.
- Count of frames over 33.3 ms and 50 ms.
- GPU 3D utilization.
- GPU compute utilization, if the tool exposes it.
- VRAM used.
- CPU package utilization.
- LLM tokens per second.
- JJML scheduler stats:
  - chunk count
  - slot wait count and total slot wait time
  - fence wait count and total fence wait time

## Suggested Tools

- CapFrameX or PresentMon for frame-time capture.
- MSI Afterburner / RTSS, HWiNFO, or vendor tools for GPU utilization and VRAM.
- JJML scheduler stats from `GgmlVulkanScheduler.getStats()`.
- Application logs containing model, level, context size, GPU layer count, and generated token count.

## Acceptance Criteria

For the same graphics workload:

- `level10` must match upstream/full-speed inference behavior within normal run-to-run variance.
- `level0` must reduce game frame-time spikes compared with `level10` while inference is running.
- `level0` should not introduce recurring CPU-side stutter from excessive waiting or synchronization.
- Frame-time p99 is more important than average FPS.
- GPU memory usage must stay stable across repeated load/unload and inference runs.
- OBS/overlay tests must not require disabling capture globally for the game process.

Suggested initial thresholds:

- Compared with game-only baseline, `level0` p99 frame time should stay within 15-25% on the primary target machine.
- Compared with `level10`, `level0` should improve p99 frame time by at least 20% in GPU-bound scenes.
- If `level0` LLM tokens per second is too low for product use, adjust the level mapping rather than weakening `level10`.

## Failure Handling

If a case fails:

- If only `level10` fails, investigate upstream Vulkan/JJML integration first.
- If `level10` passes but `level0-9` fails, disable scheduler by default and keep it behind a config flag.
- If OBS/overlay causes crashes, record the layer/tool version and do not add `DISABLE_VULKAN_OBS_CAPTURE` as a default runtime setting.
- If frame-time spikes correlate with high `fenceWaitCount`, increase `maxChunksInFlight` or active window for that level.
- If frame-time spikes correlate with high GPU queue pressure, reduce chunk size or active window for that level.

## Report Template

Record one row per test case:

| Field | Value |
| --- | --- |
| Date | |
| GPU / driver | |
| CPU | |
| RAM / VRAM | |
| OS | |
| Game version / modpack | |
| Shader / graphics preset | |
| Model / quant | |
| Context size / GPU layers | |
| LLM level | |
| Prompt and generated tokens | |
| Avg FPS / 1% / 0.1% | |
| Frame time avg / p95 / p99 / max | |
| Frames > 33.3 ms / > 50 ms | |
| GPU utilization / VRAM | |
| LLM tokens per second | |
| Scheduler stats | |
| Notes | |
