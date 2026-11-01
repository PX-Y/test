## GPT-2 small: Global G2 and STE comparison

See [small_experiments/README.md](small_experiments/README.md) for fixed parameters, STE reference results, paired evaluation, input paths, and the complete Global training log.

- [Global G2 training script](small_experiments/global/train_global.py): primal-first, LR `6e-4`, WD `0.4`, L1 `0.03/0.03`, min_lr `0`, Fisher/LR warmup `50/50`; best periodic-mix validation loss **3.516509** at step 4800.
- [Global G2 original log](small_experiments/global/train-original.log).
- STE reference: random shared periodic phase 15, LR `1e-3`, WD `0.3`; best validation loss **3.519908**. Settings and evaluation are documented in the linked README; STE source is not included.

Run `python -u small_experiments/global/train_global.py` without additional configuration flags. Keep adjacent custom modules and prepare the external dataset and floating-point initialization checkpoint described in the README.

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
