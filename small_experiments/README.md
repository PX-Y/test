# Global 与 STE：固定参数代码及原始训练日志

本包对应这两组已完成实验：

| 项目 | Global | STE（使用 Global 的 FP mask） |
|---|---|---|
| 模型 | GPT-2 small：12 层、12 heads、宽度 768 | 相同 |
| 总步数 | 5000 | 5000 |
| LR / 最低 LR | 7e-4 / 7e-5 | 1e-3 / 0 |
| Weight decay | 0.4 | 0.1 |
| 最佳 val loss | 3.525939，4800 步 | 3.539313，4800 步 |
| 最终 val loss | 3.5606 | 3.574867 |
| FP 位置 | global periodic，周期 20，保留 1 组 | 读取左侧最佳 checkpoint 的完全相同位置 |
| 量化 | 4-bit，step=0.01，网格 -0.07～0.07（15 个值） | 相同 |
| Batch / 梯度累积 / block | 16 / 32 / 256 | 相同 |
| Eval 间隔 / eval batches | 100 / 50 | 相同 |
| 日志间隔 / compile | 每步 / False | 相同 |
| AdamW betas / grad clip | 0.9, 0.95 / 1 | 相同 |
| LR warmup | 150 | 300 |
| Fisher warmup | 150 | 无 |
| primal_first | True | 不适用，直接 STE |
| Budget start/middle/final | 3.70 / 3.45 / 3.45 | 不适用 |
| Wave budget / force commit | 3.62 / True | 不适用 |
| 三轮起点 / 截止 | 4000,4500,4700 / 4500,4700,4900 | 不适用 |
| Phase 冻结 / 刷新 | 600 / 83 | 整个训练固定 mask |
| Pull / L1 系数 | tran_grid / 0.1,0.1 | 不适用 |

两者均从保留的重建 FineWeb 浮点起点初始化。STE 只读取 Global 的 mask，不使用 Global 训练后的权重初始化。这里的 STE 是后来 mask 对齐的新实验，不是之前随机 mask、loss=3.548665 的那组。

## 直接运行

在本目录运行以下命令即可，不需要 `experiment.py`、`configurator.py` 或命令行附加参数。保留各脚本旁的 `model.py`、量化模块等配件：

```bash
python global/train_global.py
```

或者：

```bash
python ste_global_mask/train_ste.py
```

它们分别写入 `global/out/` 和 `ste_global_mask/out/`，互不覆盖。参数直接在代码顶部对应赋值处修改，`exec(configurator.py)` 已移除。固定为单 GPU，不受调度器 RANK 环境变量影响。

需要使用具有 PyTorch CUDA、numpy 等依赖的 Python 环境；现有 EFabric `/home/xpan92/nanoGPT/.venv/bin/python` 已有依赖。目录内附有自定义 model/量化配件，不包含 PyTorch 等第三方库。

## 输入文件

默认使用 EFabric 上仍保留的数据和权重路径，未把大文件复制进压缩包：

- 两者的 `init_ckpt_path`：`/home/xpan92/nanoGPT/global-4bit-lr7e4-budget345-wave362/rebuild-fp/out/final_weights.pt`。
- 浮点初始化旁的 `model_config.json` 也需要保留。
- `data_root`：`/home/xpan92/nanoGPT/data`；读取 `fineweb/train.bin` 和 `fineweb/val.bin`。当前缓存为 200M train token、1M val token。
- STE 的 `fp_mask_path`：`/home/xpan92/global4bit-pf-lr7e4-wd04-job-Rx3qgFEP/out/ckpt_periodic_rowdual_best_valmix.pt`。

迁移到其他机器时，需要复制上述输入文件并修改这些路径。模型和训练数据不是凭一个脚本就能自动包含的。

## 可读源码和日志

- `global/train_global.py`：普通源码形式入口，配套模块在同目录下。
- `global/train-original.log`：Global 原实验完整日志。
- `ste_global_mask/train_ste.py`：普通源码形式入口，配套模块在同目录下。
- `ste_global_mask/train-original.log`：新 STE 原实验完整日志。
- `verification.json`：参数核对、训练函数/循环未改动、语法检查及日志 SHA256。

普通源码入口也不读取外部配置，直接 `python global/train_global.py` 或 `python ste_global_mask/train_ste.py` 即可；它们分别写入自己目录的 `out/`。

所有源码均取自实际运行版本。适配仅固化当时有效参数、固定单 GPU、集中数据路径及分离输出目录；两份日志是原实验产物，没有为打包重新训练。
