## GPT-2 small: fixed-parameter Global and STE experiments

See [small_experiments/README.md](small_experiments/README.md) for the exact settings, required input paths, and complete training logs.

- [Global single-file script](small_experiments/standalone/train_global.py): primal-first, LR `7e-4`, weight decay `0.4`; best validation loss `3.525939`.
- [STE single-file script](small_experiments/standalone/train_ste.py): same FP mask as Global, LR `1e-3`, weight decay `0.1`; best validation loss `3.539313`.
- Original training logs: [Global](small_experiments/global/train-original.log), [STE](small_experiments/ste_global_mask/train-original.log).

Run either script with Python, without command-line configuration flags. Custom modules are embedded in the single-file editions; the Python/CUDA dependencies, dataset token files and initialization/mask checkpoints are described in the linked README.

## Original XL commands


`python -u train_xl_periodic_4bit_15k_lr1e4_v8.py \
  --pull_shape=const --hybrid_p=0.0 \
  --v3_l1_c_s1=0.1 --v3_l1_c_s2=0.1 \
  --loss_pause_enable=False --loss_dual_lr=0.05 \
  --periodic_val_budget_start=3.03 --periodic_val_budget_middle=3.00 --periodic_val_budget_final=2.99 `

  `python -u train_xl_periodic_4bit_15k_lr1e4_v8.py \
  --pull_shape=tran_grid --hybrid_p=0.0 \
  --v3_l1_c_s1=0.05 --v3_l1_c_s2=0.05 \
  --loss_pause_enable=False --loss_dual_lr=0.05 \
  --periodic_val_budget_start=3.03 --periodic_val_budget_middle=3.00 --periodic_val_budget_final=2.99`
