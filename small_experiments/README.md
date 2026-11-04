# GPT-2 small：Global G6 与 STE 对照


## 配置

| 参数 | Global G6 | STE 随机生成初始phase，然后建立同款mask |
|---|---|---|
| 模型 | GPT-2 small，12层/12 heads/宽度768，约124M | 相同 |
| 初始化 | 重建 FineWeb 浮点5000步权重 | 同一起点，不使用 Global 量化后权重 |
| 训练步数 | 5000 | 5000 |
| LR / min_lr | **6e-4 / 0** | 1e-3 / 0 |
| WD | 0.5 | 0.3 |
| Fisher / LR warmup | **50 / 50** | 无 / 300 |
| L1 s1 / s2 | **0.03 / 0.03** | 不适用 |
| primal_first / pull_shape | True / tran_grid | 不适用 |
| Budget 起/中/末 | 3.70 / 3.45 / 3.45 | 不适用 |
| Wave budget / force commit | 3.62 / True | 不适用 |
| 三轮开始 / 截止 | 4000,4500,4700 / 4500,4700,4900 | 不适用 |
| Phase | 全模型共享，按敏感度和量化误差选择；600步冻结，刷新间隔83 | seed=1337，随机抽到15（0–19编号），全模型共享且全程固定 |
| FP 规则 | column % 20 == phase | 第15、35、55……列 |
| 目标矩阵内实际 FP 比例 | 5.05642361% | 4.95876736% |
| 量化 | 4-bit，step=0.01，-0.07至0.07共15个值 | 相同 |
| Batch / 梯度累积 / block | 16 / 32 / 256 | 相同 |
| AdamW betas / grad clip | 0.9,0.95 / 1.0 | 相同 |
| 精度 / compile | bfloat16 / False | 相同 |
| 验证间隔 / eval batches | 100 / 50 | 相同 |
| Best val loss / iteration | **3.514864 / 4800** | **3.519908 / 4800** |
| Final val loss / iteration | 3.5509 / 5000 | 3.552879 / 5000 |

## 评估结果

| 数据集 | Global G6 | STE 随机 phase |
|---|---:|---:|
| WikiText 验证 | **4.101398** | 4.130829 |
| WikiText 测试 | **4.030696** | 4.061563 |
| CNN/DailyMail | **3.538491** | 3.544739 |
| arXiv | **4.078955** | 4.107883 |
| LAMBADA | **3.739464** | 3.742901 |
| PG19 | **3.822147** | 3.830295 |
| FineWeb 留出集 | **3.601566** | 3.604145 |


## 本次实验：Global B 与 STE（step=0.03）

以下为本次新增实验，保留上方 step=0.01 的历史记录。仅提供 Global 的代码和日志，STE 仅记录配置与结果。

### 配置

| 参数 | Global B | STE 随机共同 phase |
|---|---|---|
| 模型 | GPT-2 small，12层 / 12 heads / 宽度768，约124M | 相同 |
| 初始化 | 原 FineWeb 浮点5000步权重 | 同一个浮点起点，不使用 Global 量化权重 |
| 训练步数 | 5000（前50步为冻结权重的 Fisher 累积） | 5000 次优化器更新 |
| LR / min_lr | **6e-4 / 0** | **1e-3 / 0** |
| WD | **0.5** | **0.3** |
| Fisher / LR warmup | **50 / 50** | 无 / 300 |
| L1 s1 / s2 | **0.1 / 0.1** | 不适用 |
| primal_first / pull_shape | True / tran_grid | 不适用 |
| Budget 起 / 中 / 末 | 3.70 / 3.45 / 3.45 | 不适用 |
| Wave budget / force commit | 3.62 / True | 不适用 |
| 三轮开始 / 截止 | **3000,3500,4000 / 3500,4000,4500** | 不适用 |
| 三轮目标 | 每行锁定16 / 18 / 19个余数组 | 不适用 |
| Phase | 全模型共享，按敏感度和量化误差选择；600步冻结，刷新间隔83 | seed=1337，随机抽到15（0–19编号），全模型共享且全程固定 |
| FP 规则 | column % 20 == phase，其他目标坐标量化 | column % 20 == 15，其他目标坐标量化 |
| 目标矩阵内实际 FP 比例 | 5.05642361% | 4.95876736% |
| 量化 | **4-bit，step=0.03，-0.21至0.21共15个值** | 相同 |
| Batch / 梯度累积 / block | 16 / 32 / 256 | 相同 |
| 每次更新 tokens | 131,072 | 相同 |
| AdamW betas / grad clip | 0.9,0.95 / 1.0 | 相同 |
| 精度 / compile | bfloat16 / False | 相同 |
| 验证间隔 / eval batches | 100 / 50 | 相同 |
| Best val loss / iteration | **3.490158 / 4800**（periodic-mix） | **3.498702 / 4800** |
| Final val loss / iteration | **3.5209 / 5000**（periodic-mix） | **3.530985 / 5000** |

### 结构与训练方式

GPT-2 Small 共12个 Transformer block。每层量化 `attn.c_attn`、`attn.c_proj`、`mlp.c_fc`、`mlp.c_proj` 的权重，共48个目标矩阵、84,934,656个目标参数；其余参数保持浮点。目标矩阵按列模20划分，每行一个余数组保留浮点，其余19组使用4bit格点；不同 phase 对应不同列数，因此两者实际 FP 比例略有差异。

Global B 使用连续权重训练，配合 primal-first、tran_grid 拉力与逐步收缩，在三轮 wave 中提交硬量化，最终完成所有非 FP 余数组。STE 使用固定随机共同 phase=15 的周期 mask，前向对非 FP 坐标量化，反向更新浮点影子权重。本次评估使用两者各自的原始 FP mask。

### 评估结果

两个模型均使用第4800步的 best。沿用上方评估的数据缓存并核验 SHA256；每项取相同的50批、batch=8、context=256、seed=1337，以 bfloat16 运行。表内为 next-token 交叉熵 loss，越低越好；这是固定采样评估，LAMBADA 也是全文 loss，而非末词准确率。

| 数据集 | Global B，step=0.03 | STE 随机 phase，step=0.03 |
|---|---:|---:|
| WikiText 验证 | **4.051934** | 4.080221 |
| WikiText 测试 | **3.983206** | 4.008142 |
| CNN/DailyMail | **3.474016** | 3.512547 |
| arXiv | **4.002466** | 4.075277 |
| LAMBADA | **3.703354** | 3.729701 |
| PG19 | **3.761803** | 3.802337 |
| FineWeb 留出集 | **3.578797** | 3.583558 |

本次 Global B 在7项评估中 loss 均低于 STE。

### Global 文件与运行

- [Global B 固定参数代码](global/train_global_step003_B.py)
- [Global B 完整训练日志](global/train-step003-B.log)
- [续跑状态工具](global/resume_utils.py)

参数已写入脚本，无需外加训练参数；保留同目录现有的 `model.py`、`periodic_dynamic_ops_v3.py`、`periodic_rowdual_ops.py` 和 `qat_prox2/` 配件。

```bash
cd small_experiments/global
python -u train_global_step003_B.py
```

训练数据目录由 `data_root` 指定（默认 `/home/xpan92/nanoGPT/data`）；初始化由 `init_ckpt_path` 指向原 FineWeb 浮点5000步权重。换机器时只需修改这两个路径，并保留起点旁的 `model_config.json`。数据集和 checkpoint 不包含在仓库内。

本次脚本输出到 `global/out/step003-B/`，与旧 G6 输出分开。每100步覆盖保存 `training_resume.pt`，包含连续训练权重、优化器、随机数、数据位置、Fisher/phase/dual/wave 状态；相同代码和配置重启会自动续跑。best/final 单独保存且不含优化器；突然终止时可能需要重跑最后一次存档之后的步骤。
