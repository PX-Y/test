"""Single-process, update-boundary training recovery; deployment weights stay separate."""
import hashlib
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch

FORMAT = 'full-training-resume-v1'
stop_requested = False

def install_signals():
    def request_stop(signum, frame):
        global stop_requested
        stop_requested = True
        print(f'[resume] signal={signum}; saving at next complete iteration', flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

def pack(value):
    if isinstance(value, torch.Tensor):
        return {'__resume_tensor__': value.detach().cpu().clone(), 'device': value.device.type}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(pack(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype, np.ndarray, np.generic)):
        return value.copy() if isinstance(value, np.ndarray) else value
    raise TypeError(f'Unsupported recovery state: {type(value)}')

def unpack(value, device):
    if isinstance(value, dict):
        if '__resume_tensor__' in value:
            return value['__resume_tensor__'].to(device if value['device'] == 'cuda' else 'cpu')
        return {k: unpack(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(unpack(v, device) for v in value)
    return value

def signature(config):
    settings = {k: v for k, v in config.items() if k not in ('init_from', 'out_dir')}
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path('.').rglob('*.py')) if 'out' not in p.parts and '__pycache__' not in p.parts}
    return hashlib.sha256(json.dumps({'config': settings, 'sources': sources}, sort_keys=True).encode()).hexdigest()

def load(path, expected_signature):
    # Only load our own full training state, never fall back to a deployment best.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != FORMAT or state.get('signature') != expected_signature:
        raise ValueError('Resume format/code/settings mismatch; refusing partial recovery')
    for key in ('model', 'optimizer', 'scaler', 'rng', 'loaders', 'pending', 'runtime', 'model_args'):
        if key not in state:
            raise ValueError(f'Incomplete resume checkpoint: missing {key}')
    if state['torch_version'] != str(torch.__version__):
        raise ValueError('Resume requires the same PyTorch version')
    return state

def save(path, namespace, model, optimizer, scaler, extra, run_signature):
    loaders = {}
    for name in ('train_loader', 'val_loader'):
        loader = namespace[name]
        data_path = str(loader.data.filename)
        stat = os.stat(data_path)
        loaders[name] = {'epoch': loader.epoch, 'current_pos': loader.current_pos,
                         'indices': loader.indices.detach().cpu().clone(),
                         'data_path': data_path, 'data_size': stat.st_size,
                         'data_mtime_ns': stat.st_mtime_ns}
    state = {'format': FORMAT, 'signature': run_signature,
             'torch_version': str(torch.__version__),
             'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             'model_args': namespace['model_args'], 'optimizer': optimizer.state_dict(),
             'scaler': scaler.state_dict(), 'iter_num': namespace['iter_num'],
             'best_val_loss': namespace['best_val_loss'], 'loaders': loaders,
             'pending': (namespace['X'].detach().cpu(), namespace['Y'].detach().cpu()),
             'runtime': pack(extra),
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    print(f'[resume-saved] next_iter={namespace["iter_num"]} path={path}', flush=True)

def restore_tail(state, namespace):
    for name, saved in state['loaders'].items():
        loader = namespace[name]
        stat = os.stat(str(loader.data.filename))
        if (str(loader.data.filename), stat.st_size, stat.st_mtime_ns) != (
                saved['data_path'], saved['data_size'], saved['data_mtime_ns']):
            raise ValueError('Dataset changed since checkpoint')
        loader.epoch, loader.current_pos = saved['epoch'], saved['current_pos']
        loader.indices = saved['indices']
    namespace['X'], namespace['Y'] = [v.to(namespace['device']) for v in state['pending']]
    namespace['scaler'].load_state_dict(state['scaler'])
    rng = state['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        if len(rng['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed')
        torch.cuda.set_rng_state_all(rng['cuda'])
    print(f'[resume-restored] next_iter={namespace["iter_num"]}; optimizer/RNG/data restored', flush=True)
