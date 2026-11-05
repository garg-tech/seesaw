"""Decoder layer stored as two TP halves.

A layer's output is f_h0(x) + f_h1(x). Under TP each GPU holds one half and
all-reduce does the sum; under PP one GPU holds both and adds them locally.
Same code, so a re-shard only has to move half tensors.

Like vLLM, Q/K/V and gate/up are one GEMM each, and torch.compile fuses the small ops
between the GEMMs, the attention kernels and the collectives. SEESAW_COMPILE=0 runs eager.
"""
import functools
import os

import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

from ..config import BLOCK_SIZE

SEGMENTS = 16       # the decode kernel's parallel softmax segments, as in vLLM
SEGMENT_SEQS = 8    # above this the 2D path wins: 66.8 vs 64.3 ms at B=32, 38 vs 42 ms at B=1
COMPILE = os.environ.get("SEESAW_COMPILE", "1") == "1"


def compiled(fn):
    return torch.compile(fn) if COMPILE else fn


def rms_norm(x, weight, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def rope_table(head_dim, params, max_len, device):
    """cos/sin per position, [max_len, head_dim]; half cos then half sin."""
    theta = params["rope_theta"]
    if params.get("rope_type") == "proportional":
        angles = int(head_dim * params["partial_rotary_factor"]) // 2
        inv = theta ** (-torch.arange(0, 2 * angles, 2, dtype=torch.float) / head_dim)
        inv = torch.cat([inv, torch.zeros(head_dim // 2 - angles)])
    else:
        inv = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    freqs = torch.outer(torch.arange(max_len, dtype=torch.float), inv)
    return torch.cat([freqs.cos(), freqs.sin()], -1).to(device, dtype=torch.bfloat16)


def apply_rope(x, cos, sin):
    """x: [T, heads, head_dim], cos/sin: [T, 1, head_dim // 2]"""
    a, b = x.chunk(2, -1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


# One compiled function per shape family, shared by every layer of that family. They take
# tensors only, so dynamo guards on shapes and never recompiles per layer.
@functools.cache
def attn_in(q_heads, kv_heads, head_dim, own_v, eps):
    """Input norm, the QKV GEMM, per-head norms and RoPE, for one half."""
    q_n, kv_n = q_heads * head_dim, kv_heads * head_dim

    def f(x, norm, w_qkv, q_norm, k_norm, cos, sin):
        qkv = F.linear(rms_norm(x, norm, eps), w_qkv)
        q = qkv[:, :q_n].unflatten(-1, (q_heads, head_dim))
        k = qkv[:, q_n:q_n + kv_n].unflatten(-1, (kv_heads, head_dim))
        # Full layers ship no v_proj: V is K before its norm and RoPE.
        v = qkv[:, q_n + kv_n:].unflatten(-1, (kv_heads, head_dim)) if own_v else k
        q = apply_rope(rms_norm(q, q_norm, eps), cos, sin)
        return q, apply_rope(rms_norm(k, k_norm, eps), cos, sin), rms_norm(v, None, eps)
    return compiled(f)


@functools.cache
def attn_out(eps):
    """Norm the attention output, add the residual, then the pre-MLP norm."""
    def f(y, x, post, pre):
        x = rms_norm(y, post, eps) + x
        return x, rms_norm(x, pre, eps)
    return compiled(f)


@functools.cache
def mlp_half():
    def f(h, w_gate_up, w_down):
        gate, up = F.linear(h, w_gate_up).chunk(2, -1)
        return F.linear(F.gelu(gate, approximate="tanh") * up, w_down)
    return compiled(f)


@functools.cache
def layer_out(eps):
    def f(y, x, post, scalar):
        return (rms_norm(y, post, eps) + x) * scalar
    return compiled(f)


class Layer:
    def __init__(self, spec, cfg, ckpt, device, halves=(0, 1)):
        self.spec, self.cfg, self.device = spec, cfg, device
        self.scale = 1.0
        self.window = (-1, -1) if spec.full else (cfg.window - 1, 0)
        self.rope_key = "full_attention" if spec.full else "sliding_attention"
        p = f"layers.{spec.idx}."
        self.norms = {n: ckpt.full(p + n + ".weight", device) for n in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm")}
        self.q_norm = ckpt.full(p + "self_attn.q_norm.weight", device)
        self.k_norm = ckpt.full(p + "self_attn.k_norm.weight", device)
        self.layer_scalar = ckpt.full(p + "layer_scalar", device)
        self.one = torch.ones((), device=device)
        self.h = {h: self._load_half(ckpt, p, h) for h in halves}
        self._attn_in = attn_in(spec.q_half, spec.kv_half, spec.head_dim, not spec.full, cfg.eps)
        self._attn_out, self._mlp, self._out = attn_out(cfg.eps), mlp_half(), layer_out(cfg.eps)

    def _load_half(self, ckpt, p, h):
        s, d = self.spec, self.device
        q_n = s.q_half * s.head_dim
        m = self.cfg.mlp // 2
        # Merged on the CPU and moved once: concatenating on the GPU leaves holes that the
        # KV cache, allocated next, can't use.
        cpu = "cpu"
        q = ckpt.rows(p + "self_attn.q_proj.weight", h * q_n, (h + 1) * q_n, cpu)
        if s.full:
            # One KV head can't split, so every half carries the whole k_proj.
            kv = [ckpt.full(p + "self_attn.k_proj.weight", cpu)]
        else:
            kv_n = s.kv_half * s.head_dim
            kv = [ckpt.rows(p + f"self_attn.{n}_proj.weight", h * kv_n, (h + 1) * kv_n, cpu) for n in "kv"]
        gate_up = [ckpt.rows(p + f"mlp.{n}_proj.weight", h * m, (h + 1) * m, cpu) for n in ("gate", "up")]
        return {
            "qkv": torch.cat([q, *kv]).to(d),
            "o": ckpt.cols(p + "self_attn.o_proj.weight", h * q_n, (h + 1) * q_n, d),
            "gate_up": torch.cat(gate_up).to(d),
            "down": ckpt.cols(p + "mlp.down_proj.weight", h * m, (h + 1) * m, d),
        }

    def _attend(self, q, key_cache, value_cache, b, kv, table):
        out = torch.empty_like(q)
        n = key_cache.shape[2]
        if not hasattr(b, "descale"):
            b.descale = {}
        if n not in b.descale:
            b.descale[n] = torch.ones((b.cu_seqlens.shape[0] - 1, n), device=q.device)
        ones = b.descale[n]
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=b.cu_seqlens, max_seqlen_q=b.max_q,
            seqused_k=b.seq_lens, max_seqlen_k=b.max_k,
            softmax_scale=self.scale, causal=True, window_size=self.window,
            block_table=table,
            softcap=0, q_descale=None, k_descale=ones, v_descale=ones,
            seq_threshold_3D=SEGMENT_SEQS, num_par_softmax_segments=SEGMENTS,
            **kv.segments[self.spec.head_dim],
        )
        return out

    def attention(self, x, b, kv, comm, rope):
        s, (cos, sin) = self.spec, rope
        y = None
        for h, w in self.h.items():
            q, k, v = self._attn_in(x, self.norms["input_layernorm"], w["qkv"], self.q_norm, self.k_norm, cos, sin)
            if s.full:
                kc, vc, table = kv.k[s.idx], kv.v[s.idx], b.table_full
                if y is None:       # both halves make the same K/V
                    triton_reshape_and_cache_flash(k, v, kc, vc, b.map_full, "auto", self.one, self.one)
            else:
                kc, vc = kv.k[s.idx][h], kv.v[s.idx][h]
                staged = (s.idx, h) in kv.staged
                table = b.table_stage if staged else b.table_sliding
                triton_reshape_and_cache_flash(k, v, kc, vc, b.map_stage if staged else b.map_sliding,
                                               "auto", self.one, self.one)
            out = F.linear(self._attend(q, kc, vc, b, kv, table).flatten(-2), w["o"])
            y = out if y is None else y + out
        if comm is not None:
            comm.all_reduce(y, y)
        return y

    def mlp(self, h, comm):
        y = None
        for w in self.h.values():
            out = self._mlp(h, w["gate_up"], w["down"])
            y = out if y is None else y + out
        if comm is not None:
            comm.all_reduce(y, y)
        return y

    def forward(self, x, b, kv, comm, rope):
        n = self.norms
        y = self.attention(x, b, kv, comm, rope)
        x, h = self._attn_out(y, x, n["post_attention_layernorm"], n["pre_feedforward_layernorm"])
        y = self.mlp(h, comm)
        return self._out(y, x, n["post_feedforward_layernorm"], self.layer_scalar)


class KVStore:
    """K and V tensors, starting in the TP layout: one half of every layer.

    k[l] is a dict {half: tensor} for sliding layers, one tensor for full ones.
    Full layers exist on both GPUs, since TP needs them there, even though PP uses
    only the copy on the stage that runs the layer.
    Shape [blocks, BLOCK_SIZE, heads, head_dim], what the kernels index.
    """

    def __init__(self, specs, tables, device, halves):
        self.k, self.v, self.segments = {}, {}, {}
        self.staged = set()     # (layer, half) whose KV is in the staging pool, see kv_cache.py
        for s in specs:
            # Scratch for the kernels' decode path. Layers run one at a time, so they share it.
            shape = (SEGMENT_SEQS, s.q_half, SEGMENTS)
            self.segments.setdefault(s.head_dim, dict(
                softmax_segm_output=torch.empty((*shape, s.head_dim), dtype=torch.float32, device=device),
                softmax_segm_max=torch.empty(shape, dtype=torch.float32, device=device),
                softmax_segm_expsum=torch.empty(shape, dtype=torch.float32, device=device)))
            if s.full:
                shape = (tables.num_full, BLOCK_SIZE, 1, s.head_dim)
                self.k[s.idx] = torch.empty(shape, dtype=torch.bfloat16, device=device)
                self.v[s.idx] = torch.empty(shape, dtype=torch.bfloat16, device=device)
            else:
                shape = (tables.num_sliding, BLOCK_SIZE, s.kv_half, s.head_dim)
                self.k[s.idx] = {h: torch.empty(shape, dtype=torch.bfloat16, device=device) for h in halves}
                self.v[s.idx] = {h: torch.empty(shape, dtype=torch.bfloat16, device=device) for h in halves}
