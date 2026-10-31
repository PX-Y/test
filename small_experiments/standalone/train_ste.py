# Single-file edition. Custom Python modules are embedded below and unpacked
# into a temporary directory; experiment parameters remain readable below.
# Requires the same external torch/numpy environment, data and initialization weights.
import base64 as _embedded_base64
import io as _embedded_io
import sys as _embedded_sys
import tempfile as _embedded_tempfile
import zipfile as _embedded_zipfile
_embedded_runtime = _embedded_tempfile.TemporaryDirectory(prefix="nanogpt_fixed_")
_embedded_modules = """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"""
with _embedded_zipfile.ZipFile(_embedded_io.BytesIO(_embedded_base64.b64decode(_embedded_modules))) as _embedded_zip:
    _embedded_zip.extractall(_embedded_runtime.name)
_embedded_sys.path.insert(0, _embedded_runtime.name)

# Reproduction of STE using the best Global's exact FP mask: LR=1e-3, WD=0.1.
# Best validation loss 3.539313 at iteration 4800. Run: python train_ste.py
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from ste_quant2 import ste_quantize_weight_with_mask
import torch.nn.functional as F
import types
import hashlib
from pathlib import Path
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_ste_global_mask')
eval_interval = 100
log_interval = 1
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
best_checkpoint_filename = 'ckpt_best.pt'
init_from = 'ckpt_model_only'
init_ckpt_path = '/home/xpan92/nanoGPT/global-4bit-lr7e4-budget345-wave362/rebuild-fp/out/final_weights.pt'
data_root = '/home/xpan92/nanoGPT/data'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
# New FineWeb split with the held-out test portion removed from train/val.
dataset = 'fineweb'
# model
gradient_accumulation_steps = 32
batch_size = 16
block_size = 256
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # same as the best existing random-mask STE run
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 300 # how many steps to warm up for
lr_decay_iters = 5000
min_lr = 0 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False
ste_ratio = 0.95
ste_mask_seed = 1337
quant_bits = 4
quant_step = 0.01
fp_mask_mode = 'periodic'
fp_mask_path = '/home/xpan92/global4bit-pf-lr7e4-wd04-job-Rx3qgFEP/out/ckpt_periodic_rowdual_best_valmix.pt'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# All experiment parameters are fixed above; no CLI or external config is loaded.
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = False  # Fixed reproduction recipe: one GPU, regardless of scheduler RANK variables.
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

#data_dir = os.path.join('data', dataset)
#data_dir = os.path.join('/home/sfong7/QAT/data', dataset)
#data_dir = os.path.join('/home/sfong7/muon/nanoGPT/data/', dataset)
data_dir = os.path.join(data_root, dataset)

class EpochDataLoader:
    def __init__(self, split, batch_size, block_size, device, device_type, ddp_rank, ddp_world_size):
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        
        if split == 'train':
            self.data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
            
        self.num_chunks = (len(self.data) - block_size) // block_size
        self.epoch = 0
        self.reset()

    def reset(self):
        g = torch.Generator()
        g.manual_seed(1337 + self.epoch) 
        all_chunk_indices = torch.randperm(self.num_chunks, generator=g) * self.block_size
        
        self.indices = all_chunk_indices[self.ddp_rank :: self.ddp_world_size]
        self.current_pos = 0
        self.epoch += 1

    def get_batch(self):
        if self.current_pos + self.batch_size > len(self.indices):
            self.reset()
            
        ix = self.indices[self.current_pos : self.current_pos + self.batch_size]
        self.current_pos += self.batch_size
        
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

current_rank = ddp_rank if ddp else 0
train_loader = EpochDataLoader('train', batch_size, block_size, device, device_type, current_rank, ddp_world_size)
val_loader = EpochDataLoader('val', batch_size, block_size, device, device_type, current_rank, ddp_world_size)

def get_batch(split):
    if split == 'train':
        return train_loader.get_batch()
    else:
        return val_loader.get_batch()

def is_target_linear_name(name: str) -> bool:
    return (
        name.endswith("attn.c_attn")
        or name.endswith("attn.c_proj")
        or name.endswith("mlp.c_fc")
        or name.endswith("mlp.c_proj")
    )

def print_target_layers(model):
    print("==== target quant layers ====")
    for name, module in model.named_modules():
        if is_target_linear_name(name):
            print(name, type(module))
    print("=============================")


def build_fixed_mask_for_weight(weight: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    numel = weight.numel()
    n_quant = int(round(ratio * numel))
    flat_mask = torch.zeros(numel, dtype=torch.float32)
    if n_quant > 0:
        ix = torch.randperm(numel, generator=g)[:n_quant]
        flat_mask[ix] = 1.0
    return flat_mask.view_as(weight)


def patch_linear_module_for_ste(module, fixed_mask, n_bits=4, step=0.1/7):
    if hasattr(module, "_ste_patched") and module._ste_patched:
        return

    module.register_buffer("_ste_mask", fixed_mask.to(dtype=torch.float32), persistent=False)

    def ste_forward(self, x):
        w_ste = ste_quantize_weight_with_mask(self.weight, self._ste_mask, n_bits=n_bits, step=step)
        return F.linear(x, w_ste, self.bias)

    module.forward = types.MethodType(ste_forward, module)
    module._ste_patched = True


def clean_weight_name(name):
    while name.startswith(('_orig_mod.', 'module.')):
        name = name.split('.', 1)[1]
    return name.replace('._orig_mod.', '.')


def load_periodic_phases(path):
    if not path:
        raise ValueError('periodic FP requires fp_mask_path')
    obj = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if 'phase' in obj:
        period, raw = int(obj['period']), obj['phase']
    elif 'periodic_phase' in obj:
        period, raw = int(obj.get('periodic_period', 20)), obj['periodic_phase']
        if int(obj.get('periodic_fp_groups_per_row', 1)) != 1:
            raise ValueError('expected one FP residue group per row')
    else:
        raise KeyError('mask source has no phase or periodic_phase')
    if period < 1:
        raise ValueError('invalid periodic FP period')
    phases = {}
    for name, phase in raw.items():
        if phase.ndim != 1 or phase.is_floating_point():
            raise ValueError(f'invalid phase tensor: {name}')
        if bool(((phase < 0) | (phase >= period)).any()):
            raise ValueError(f'phase outside [0, period): {name}')
        phases[clean_weight_name(name)] = phase.to(torch.int64).clone()
    return period, phases


def build_periodic_quant_mask(weight, phase, period):
    if weight.ndim != 2 or phase.numel() != weight.shape[0]:
        raise ValueError('phase/weight shape mismatch')
    fp = (torch.arange(weight.shape[1]) % period)[None, :] == phase.long()[:, None]
    return (~fp).to(torch.float32)


def apply_ste_to_target_layers(model, n_bits=4, step=0.01, ratio=0.95,
                               mask_seed=1337, mode='periodic', mask_path=''):
    global periodic_mask_metadata
    if mode not in ('periodic', 'random'):
        raise ValueError('fp_mask_mode must be periodic or random')
    period, phases = load_periodic_phases(mask_path) if mode == 'periodic' else (None, {})
    targets = {name + '.weight' for name, module in model.named_modules()
               if is_target_linear_name(name) and hasattr(module, 'weight')}
    if mode == 'periodic' and targets != set(phases):
        raise ValueError(f'mask target mismatch: missing={targets-set(phases)}, extra={set(phases)-targets}')
    patched = []
    total = quantized = 0
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if is_target_linear_name(name):
            if not hasattr(module, "weight"):
                continue
            if mode == 'periodic':
                fixed_mask = build_periodic_quant_mask(module.weight, phases[name + '.weight'], period)
            else:
                fixed_mask = build_fixed_mask_for_weight(module.weight, ratio=ratio, seed=mask_seed + len(patched))
            total += fixed_mask.numel()
            quantized += int(fixed_mask.sum())
            digest.update(name.encode())
            digest.update(fixed_mask.to(torch.uint8).numpy().tobytes())
            patch_linear_module_for_ste(module, fixed_mask=fixed_mask, n_bits=n_bits, step=step)
            patched.append(name)

    if len(patched) != 48:
        raise ValueError(f'expected 48 GPT-2 small target matrices, got {len(patched)}')
    periodic_mask_metadata = {'period': period, 'phase': phases,
        'source': mask_path, 'mask_sha256': digest.hexdigest(),
        'target_parameters': total, 'quantized_parameters': quantized,
        'fp_fraction': (total - quantized) / total}
    print(f"[STE] patched {len(patched)} modules, mode={mode}, {n_bits}-bit step={step}, "
          f"quantized={quantized}/{total}, FP={(total-quantized)/total:.8%}, mask_sha256={digest.hexdigest()}")
    for n in patched:
        print("  ", n)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from == 'ckpt_model_only':
    print(f"Initializing from model-only checkpoint: {init_ckpt_path}")
    checkpoint = torch.load(init_ckpt_path, map_location='cpu', weights_only=True)
    if 'model' in checkpoint and 'model_args' in checkpoint:
        checkpoint_model_args = checkpoint['model_args']
    else:
        import json
        from pathlib import Path
        checkpoint_model_args = json.loads(Path(init_ckpt_path).with_name('model_config.json').read_text())
        checkpoint = {'model': checkpoint, 'model_args': checkpoint_model_args}
    assert (checkpoint_model_args['n_layer'],checkpoint_model_args['n_head'],checkpoint_model_args['n_embd']) == (12,12,768)

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    iter_num = 0
    best_val_loss = 1e9

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

print_target_layers(model)
apply_ste_to_target_layers(model, n_bits=quant_bits, step=quant_step, ratio=ste_ratio,
    mask_seed=ste_mask_seed, mode=fp_mask_mode, mask_path=fp_mask_path)
if master_process and fp_mask_mode == 'periodic':
    exported_mask_path = os.path.abspath(os.path.join(out_dir, 'fp_mask.pt'))
    torch.save(periodic_mask_metadata, exported_mask_path + '.partial')
    os.replace(exported_mask_path + '.partial', exported_mask_path)
else:
    exported_mask_path = ''

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or iter_num == max_iters) and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        val_loss = float(losses['val'].item())
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            # Save only one model-only best checkpoint. Each improvement
            # overwrites the previous file, avoiding a full Adam optimizer
            # checkpoint for this GPT-2 XL control experiment.
            checkpoint = {
                'format': 'small-ste4bit-globalfp-best-v1',
                'model': raw_model.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'ste_ratio': float(ste_ratio),
                'ste_mask_seed': int(ste_mask_seed),
                'quant_bits': quant_bits,
                'quant_step': quant_step,
                'fp_mask_mode': fp_mask_mode,
                'fp_mask_path': exported_mask_path,
                'fp_mask_source': fp_mask_path,
                'periodic_period': periodic_mask_metadata['period'],
                'periodic_phase': periodic_mask_metadata['phase'],
                'mask_metadata': periodic_mask_metadata,
                'config': config,
            }
            checkpoint_path = os.path.join(
                out_dir, best_checkpoint_filename
            )
            print(
                f"saving new best checkpoint to {checkpoint_path} "
                f"(iter={iter_num}, val_loss={best_val_loss:.6f})"
            )
            torch.save(checkpoint, checkpoint_path + '.partial')
            os.replace(checkpoint_path + '.partial', checkpoint_path)
    if iter_num == 0 and eval_only:
        break

    # The evaluation above is also run at step=max_iters. Stop before another
    # optimizer update so max_iters is the exact number of completed updates.
    if iter_num >= max_iters:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

if master_process and not eval_only:
    import json
    from ste_quant2 import quantize_to_fixed_grid
    base = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
    deployment = {k: v.detach().cpu().clone() for k,v in base.state_dict().items()}
    for name,module in base.named_modules():
        if getattr(module, '_ste_patched', False):
            w=module.weight.detach()
            q=quantize_to_fixed_grid(w,n_bits=quant_bits,step=quant_step)
            mask=module._ste_mask.to(device=w.device,dtype=torch.bool)
            deployment[name+'.weight']=torch.where(mask,q,w).cpu()
    path=os.path.join(out_dir,'final_weights.pt')
    torch.save(deployment,path+'.partial'); os.replace(path+'.partial',path)
    with open(os.path.join(out_dir,'model_config.json'),'w') as f: json.dump(model_args,f,indent=2)
    with open(os.path.join(out_dir,'training_summary.json'),'w') as f:
        json.dump({'completed_updates':iter_num,'final_val_loss':val_loss,'best_val_loss':best_val_loss,
                   'n_bits':quant_bits,'step':quant_step,'ste_ratio_random_mode':ste_ratio,'mask_seed':ste_mask_seed,
                   'fp_mask_mode':fp_mask_mode,'fp_mask_source':fp_mask_path,
                   'fp_mask_path':exported_mask_path,'actual_fp_fraction':periodic_mask_metadata['fp_fraction'],
                   'mask_sha256':periodic_mask_metadata['mask_sha256'],
                   'weights_kind':'mixed-deployment-state-dict-float-storage','config':config},f,indent=2)
    print(f'FINAL_WEIGHTS_SAVED updates={iter_num} val_loss={val_loss:.6f} path={path}',flush=True)
if ddp:
    destroy_process_group()
