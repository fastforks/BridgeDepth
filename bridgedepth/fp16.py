"""Upstream-friendly fp16 inference helpers (plain PyTorch, ONNX/TensorRT-exportable).

prepare_fp16(model): run the model under torch.autocast(fp16) and
  - LayerNorm in the activation dtype (fp16 in/out, statistics in fp32 inside the kernel / ONNX LayerNormalization)
    instead of autocast's fp32 output, which forces an fp32 copy of every token stream and a cast back to fp16 at the
    next Linear;
  - window-attention masks cached per (shape, device) instead of rebuilt in every layer / call.
The mask cache must not be filled while tracing: for ONNX / TensorRT export the unmodified model (upstream already
exports); prepare_fp16 is for eager inference.
Disparity arithmetic (proposals, updates, sorting / gathering) stays fp32 as under plain autocast.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LN16(nn.Module):
    def __init__(self, ln, chw=False):
        super().__init__()
        self.normalized_shape, self.eps = ln.normalized_shape, ln.eps
        self.weight, self.bias = ln.weight, ln.bias
        self.chw = chw  # torchvision LayerNorm2d (ConvNeXt stem / downsampling): NCHW in, normalizes over C

    def forward(self, x):
        if self.chw:
            x = x.permute(0, 2, 3, 1)
        if x.dtype == torch.float32:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            with torch.autocast("cuda", enabled=False):
                y = F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
        return y.permute(0, 3, 1, 2) if self.chw else y


def _swap_ln(mod):
    for n, c in mod.named_children():
        if isinstance(c, nn.LayerNorm) and c.elementwise_affine:
            setattr(mod, n, _LN16(c, chw=type(c).__name__ == "LayerNorm2d"))
        else:
            _swap_ln(c)


def prepare_fp16(model, ln16=True, cache_masks=True):
    if ln16:
        _swap_ln(model)
    if cache_masks:
        from .stereo.NMP import WindowAttention
        from . import blocks
        if not hasattr(WindowAttention, "_gen_mask_orig"):
            WindowAttention._gen_mask_orig = WindowAttention.gen_window_attn_mask
            WindowAttention._gen_shift_orig = WindowAttention.gen_shift_window_attn_mask
            blocks._gen_shift_orig = blocks.gen_shift_window_attn_mask
            m1 = functools.lru_cache(64)(lambda ws, dev: WindowAttention._gen_mask_orig(ws, dev))
            m2 = functools.lru_cache(64)(lambda r, ws, s, dev: WindowAttention._gen_shift_orig(r, ws, s, dev))
            m3 = functools.lru_cache(64)(lambda r, ws, s, dev: blocks._gen_shift_orig(r, ws, s, dev))
            WindowAttention.gen_window_attn_mask = staticmethod(
                lambda window_size, device=torch.device("cuda"): m1(tuple(window_size), device))
            WindowAttention.gen_shift_window_attn_mask = staticmethod(
                lambda input_resolution, window_size, shift_size, device=torch.device("cuda"):
                m2(tuple(input_resolution), tuple(window_size), shift_size, device))
            blocks.gen_shift_window_attn_mask = lambda input_resolution, window_size, shift_size, device=torch.device("cuda"): \
                m3(tuple(input_resolution), tuple(window_size), shift_size, device)
    model.fp16 = True
    return model
