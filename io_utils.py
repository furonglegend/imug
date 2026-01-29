# utils/io_utils.py
"""
I/O helpers: config load/save (json/yaml), checkpoint save/load for PyTorch models, and simple logging helpers.
"""
import json
import os
import torch

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model_state': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optim_state': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch
    }
    torch.save(state, path)

def load_checkpoint(path, model=None, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt and ckpt['optim_state'] is not None:
        optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt
