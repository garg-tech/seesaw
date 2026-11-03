"""Decoder layer stored as two TP halves.

A layer's output is f_h0(x) + f_h1(x). Under TP each GPU holds one half and
all-reduce does the sum; under PP one GPU holds both and adds them locally.
Same code, so a re-shard only has to move half tensors.
"""
import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

from ..config import BLOCK_SIZE

SEGMENTS = 16       # the decode kernel's parallel softmax segments, as in vLLM
SEGMENT_SEQS = 8    # above this the 2D path wins: 66.8 vs 64.3 ms at B=32, 38 vs 42 ms at B=1


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


class Layer:
    def __init__(self, spec, cfg, ckpt, device, rope, halves=(0, 1)):
        self.spec, self.cfg, self.device, self.rope = spec, cfg, device, rope
        self.scale = 1.0
        self.window = (-1, -1) if spec.full else (cfg.window - 1, 0)
        p = f"layers.{spec.idx}."
        self.norms = {n: ckpt.full(p + n + ".weight", device) for n in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm")}
        self.q_norm = ckpt.full(p + "self_attn.q_norm.weight", device)
        self.k_norm = ckpt.full(p + "self_attn.k_norm.weight", device)
        self.layer_scalar = ckpt.full(p + "layer_scalar", device)
        # Full layers ship k_proj only, and it is replicated instead of split.
        self.k_shared = ckpt.full(p + "self_attn.k_proj.weight", device) if spec.full else None
        self.h = {h: self._load_half(ckpt, p, h) for h in halves}

    def _load_half(self, ckpt, p, h):
        s, d = self.spec, self.device
        q_n = s.q_half * s.head_dim
        m = self.cfg.mlp // 2
        w = {
            "q": ckpt.rows(p + "self_attn.q_proj.weight", h * q_n, (h + 1) * q_n, d),
            "o": ckpt.cols(p + "self_attn.o_proj.weight", h * q_n, (h + 1) * q_n, d),
            "gate": ckpt.rows(p + "mlp.gate_proj.weight", h * m, (h + 1) * m, d),
            "up": ckpt.rows(p + "mlp.up_proj.weight", h * m, (h + 1) * m, d),
            "down": ckpt.cols(p + "mlp.down_proj.weight", h * m, (h + 1) * m, d),
        }
        if not s.full:
            kv_n = s.kv_half * s.head_dim
            w["k"] = ckpt.rows(p + "self_attn.k_proj.weight", h * kv_n, (h + 1) * kv_n, d)
            w["v"] = ckpt.rows(p + "self_attn.v_proj.weight", h * kv_n, (h + 1) * kv_n, d)
        return w

    def _attend(self, q, key_cache, value_cache, b, kv):
        out = torch.empty_like(q)
        n = key_cache.shape[2]
        ones = torch.ones((b.cu_seqlens.shape[0] - 1, n), device=q.device)
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=b.cu_seqlens, max_seqlen_q=b.max_q,
            seqused_k=b.seq_lens, max_seqlen_k=b.max_k,
            softmax_scale=self.scale, causal=True, window_size=self.window,
            block_table=b.table_full if self.spec.full else b.table_sliding,
            softcap=0, q_descale=None, k_descale=ones, v_descale=ones,
            seq_threshold_3D=SEGMENT_SEQS, num_par_softmax_segments=SEGMENTS,
            **kv.segments[self.spec.head_dim],
        )
        return out

    def _write_kv(self, k, v, kc, vc, slot_map):
        one = torch.ones((), device=k.device)
        triton_reshape_and_cache_flash(k, v, kc, vc, slot_map, "auto", one, one)

    def attention(self, x, b, kv, comm):
        s, cfg = self.spec, self.cfg
        cs = self.rope.index_select(0, b.positions).unsqueeze(1)
        cos, sin = cs.chunk(2, -1)
        if s.full:
            raw = F.linear(x, self.k_shared).unflatten(-1, (1, s.head_dim))
            k = apply_rope(rms_norm(raw, self.k_norm, cfg.eps), cos, sin)
            v = rms_norm(raw, None, cfg.eps)
            self._write_kv(k, v, kv.k[s.idx], kv.v[s.idx], b.map_full)
        y = None
        for h, w in self.h.items():
            q = F.linear(x, w["q"]).unflatten(-1, (s.q_half, s.head_dim))
            q = apply_rope(rms_norm(q, self.q_norm, cfg.eps), cos, sin)
            if s.full:
                kc, vc = kv.k[s.idx], kv.v[s.idx]
            else:
                k = F.linear(x, w["k"]).unflatten(-1, (s.kv_half, s.head_dim))
                v = F.linear(x, w["v"]).unflatten(-1, (s.kv_half, s.head_dim))
                k = apply_rope(rms_norm(k, self.k_norm, cfg.eps), cos, sin)
                v = rms_norm(v, None, cfg.eps)
                kc, vc = kv.k[s.idx][h], kv.v[s.idx][h]
                self._write_kv(k, v, kc, vc, b.map_sliding)
            out = F.linear(self._attend(q, kc, vc, b, kv).flatten(-2), w["o"])
            y = out if y is None else y + out
        if comm is not None:
            comm.all_reduce(y, y)
        return y

    def mlp(self, x, comm):
        y = None
        for w in self.h.values():
            act = F.gelu(F.linear(x, w["gate"]), approximate="tanh") * F.linear(x, w["up"])
            out = F.linear(act, w["down"])
            y = out if y is None else y + out
        if comm is not None:
            comm.all_reduce(y, y)
        return y

    def forward(self, x, b, kv, comm=None):
        eps, n = self.cfg.eps, self.norms
        y = self.attention(rms_norm(x, n["input_layernorm"], eps), b, kv, comm)
        x = rms_norm(y, n["post_attention_layernorm"], eps) + x
        y = self.mlp(rms_norm(x, n["pre_feedforward_layernorm"], eps), comm)
        x = rms_norm(y, n["post_feedforward_layernorm"], eps) + x
        return x * self.layer_scalar


class KVStore:
    """K and V tensors, starting in the TP layout: one half of every layer.

    k[l] is a dict {half: tensor} for sliding layers, one tensor for full ones.
    Full layers exist on both GPUs, since TP needs them there, even though PP uses
    only the copy on the stage that runs the layer.
    Shape [blocks, BLOCK_SIZE, heads, head_dim], what the kernels index.
    """

    def __init__(self, specs, tables, device, halves):
        self.k, self.v, self.segments = {}, {}, {}
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
