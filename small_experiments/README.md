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
