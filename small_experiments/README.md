# GPT-2 small：Global G2 与 STE 对照

本目录提供当前最好 Global **G2** 的固定参数代码与完整训练日志。STE 仅记录参数和结果，不包含 STE 源码。

## 配置

| 参数 | Global G2 | STE 随机生成初始phase，然后建立同款mask |
|---|---|---|
| 模型 | GPT-2 small，12层/12 heads/宽度768，约124M | 相同 |
| 初始化 | 重建 FineWeb 浮点5000步权重 | 同一起点，不使用 Global 量化后权重 |
| 训练步数 | 5000 | 5000 |
| LR / min_lr | **6e-4 / 0** | 1e-3 / 0 |
| WD | 0.4 | 0.3 |
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
| Best val loss / iteration | **3.516509 / 4800** | **3.519908 / 4800** |
| Final val loss / iteration | 3.5514 / 5000 | 3.552879 / 5000 |

Global 比较指标是 **periodic-mix val loss**：保留 FP 位置，其他目标权重投影到量化网格。48个 attention/MLP 矩阵参与量化，embedding、LayerNorm 等不在这些 mask 内。STE 前向使用混合权重，反向更新潜在浮点权重。

Fisher warmup 冻结权重、仅收集梯度平方 EMA；LR warmup 按扣除 Fisher warmup 的 task iteration 计算。min_lr=0 是调度下限，不代表总5000步的最后一步必然恰好到0。

## 其他 STE 参考

以下均为 LR=1e-3、min_lr=0、LR warmup=300、5000步、相同浮点起点和4-bit网格；best 均在4800步。

| FP 布局 | WD | Best val loss |
|---|---:|---:|
| 读取旧 Global 的周期 phase | 0.1 | 3.539313 |
| 读取旧 Global 的周期 phase | 0.3 | 3.517770 |
| 读取旧 Global 的周期 phase | 0.5 | **3.511834** |
| 随机选择约5%的独立 FP 点，seed=1337 | 0.3 | 3.519940 |
| 随机共同 phase=15，周期20 | 0.3 | 3.519908 |

“读取旧 Global”使用早期 WD=0.4 Global best 的布局，**并非 G2 的 phase**。独立随机 FP 点与随机共同 phase 是不同的 mask。

## G2 与随机共同 phase STE 评估

两组均使用4800步 best；每项相同50批、batch=8、block=256、seed=1337、bfloat16，约10万 token。指标为 token 加权 next-token cross entropy，越低越好。

| 数据集 | Global G2 | STE 随机 phase |
|---|---:|---:|
| WikiText-103 validation | **4.096863** | 4.130829 |
| WikiText-103 test | **4.027134** | 4.061563 |
| CNN/DailyMail test | **3.534492** | 3.544739 |
| arXiv test（部分快照） | **4.073901** | 4.107883 |
| LAMBADA test（全文 loss） | 3.743793 | **3.742901** |
| PG19 test | **3.825910** | 3.830295 |
| FineWeb 留出集 | **3.602857** | 3.604145 |

G2 六项数值更低。WikiText、CNN/DailyMail、arXiv 的差异相对批间波动较清楚；其余三项差异较小。FP比例、LR、WD和训练方法不同，不能将结果单独归因于 phase 选择。这是采样评估；arXiv不是完整test，LAMBADA不是最后一个词准确率。

## 直接运行

```bash
python -u global/train_global.py
```

无需额外命令行参数、experiment.py 或 configurator.py。参数已固化，固定单GPU；保留 global/ 内的 model.py、量化模块和 qat_prox2/ 配件。

默认输入：
- init_ckpt_path：/home/xpan92/nanoGPT/global-4bit-lr7e4-budget345-wave362/rebuild-fp/out/final_weights.pt；旁边需有 model_config.json。
- data_root：/home/xpan92/nanoGPT/data，读取 fineweb/train.bin、fineweb/val.bin；训练缓存200M token，验证缓存1M token。
- 输出：脚本旁 global/out/，保存 best、final，不保存 optimizer/resume 状态。

迁移机器时修改路径并准备 Python、PyTorch CUDA、numpy 等依赖。仓库不含模型权重、数据集或第三方库。

## 文件

- [Global G2 代码](global/train_global.py)：取自实际运行源码，固化配置后训练函数和循环保持一致。
- [Global G2 完整原始日志](global/train-original.log)：含5000步、best和final结果。

日志 SHA256：`192c91e1f1c23d32da9487fcfda80676208e6b221b6d980876cd2c75325a357c`。
