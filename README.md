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
