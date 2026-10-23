"""CUDA-kernel inference path: prepare_fp16 (fp16.py) plus WindowAttention (head dim 32) through a fused window-attention
kernel with q / k / v relative-position embeddings (nmrf_kernels.cu, win_rpe_attn). Same math as upstream (fp32
accumulation inside the kernel), any input size, no per-size compile.
The kernel is loaded from prebuilt/nmrf_kernels_<hash>.cubin (one per sm_80/86/89/90/120); on other GPUs or after editing
the .cu it is compiled once with nvcc ($CUDA_HOME, default /usr/local/cuda) into ~/.cache/nmrf-kernels.
"""
import ctypes
import hashlib
import os
import subprocess

import torch

from .stereo.NMP import WindowAttention
from .fp16 import prepare_fp16

_HERE = os.path.dirname(os.path.abspath(__file__))
_K = [None]


class _Kern:
    """cubin loader (driver API)."""
    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = os.path.join(_HERE, "nmrf_kernels.cu")
        tag = hashlib.sha1(open(src, "rb").read() + f"sm_{major}{minor}".encode()).hexdigest()[:16]
        cubin = os.path.join(_HERE, "prebuilt", f"nmrf_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            cache = os.path.join(os.path.expanduser("~"), ".cache", "nmrf-kernels")
            os.makedirs(cache, exist_ok=True)
            cubin = os.path.join(cache, f"nmrf_kernels_{tag}.cubin")
            if not os.path.exists(cubin):
                ch = os.environ.get("CUDA_HOME", "/usr/local/cuda")
                subprocess.run([os.path.join(ch, "bin", "nvcc"), "-cubin", "-O3", "-std=c++17", f"-arch=sm_{major}{minor}",
                                "-o", cubin + ".tmp", src], check=True)
                os.replace(cubin + ".tmp", cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        torch.zeros(1, device="cuda")
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        self._img = open(cubin, "rb").read()
        assert self.lib.cuModuleLoadData(ctypes.byref(self.mod), self._img) == 0
        assert self.lib.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, b"win_rpe_attn") == 0
        self.lib.cuFuncSetAttribute(self.fn, 8, 96 * 1024)  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

    def attn(self, q, k, v, table, mask, heads, T, ws, n, scale):
        out = torch.empty_like(q)
        nW = 0 if mask is None else mask.shape[0]
        nwin = q.shape[0]
        wpc = max(1, 128 // T)
        args = [ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
                ctypes.c_void_p(table.data_ptr()), ctypes.c_int32(table.stride(0)),
                ctypes.c_void_p(None if mask is None else mask.data_ptr()), ctypes.c_int32(max(nW, 1)), ctypes.c_int32(heads),
                ctypes.c_int32(T), ctypes.c_int32(ws), ctypes.c_int32(n), ctypes.c_float(scale), ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(nwin), ctypes.c_int32(wpc)]
        arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        R = (2 * ws - 1) ** 2
        shm = (R * 96 + 2 * wpc * T * 32) * 2
        nb = (nwin + wpc - 1) // wpc * heads
        rc = self.lib.cuLaunchKernel(self.fn, nb, 1, 1, (wpc * T + 31) // 32 * 32, 1, 1, shm,
                                     torch.cuda.current_stream().cuda_stream, arr, None)
        assert rc == 0, f"win_rpe_attn launch error {rc}"
        return out


def _win_forward(self, x, attn_mask):
    B_, L, _ = x.shape
    ws = self.window_size[0]
    N = L // (ws * self.window_size[1])
    C = self.dim
    h = self.num_heads
    assert self.window_size[0] == self.window_size[1] and C // h == 32
    qkv = self.qkv(x).half().reshape(B_, L, 3, h, C // h).permute(2, 0, 3, 1, 4)
    q, k, v = (t.contiguous() for t in qkv)
    if _K[0] is None:
        _K[0] = _Kern()
    if getattr(self, "_tab16", None) is None:
        self._tab16 = self.relative_position_enc_table.detach().half().contiguous()
    mask = None if attn_mask is None else attn_mask.reshape(-1, L, L).float().contiguous()
    o = _K[0].attn(q, k, v, self._tab16, mask, h, L, ws, N, self.scale)
    return self.proj(o.transpose(1, 2).reshape(B_, L, C))


def prepare_fast(model, fused_window=True, **kw):
    prepare_fp16(model, **kw)
    if fused_window:
        WindowAttention.forward = _win_forward
    return model.eval()
