"""Gemma 4 text config, read from the checkpoint's config.json."""
import glob
import json
import os
from dataclasses import dataclass

REPO = "models--google--gemma-4-12B-it"
BLOCK_SIZE = 16


def snapshot_dir(hub=None):
    hub = hub or os.path.expanduser("~/.cache/huggingface/hub")
    root = os.path.join(hub, REPO)
    with open(os.path.join(root, "refs", "main")) as f:
        return os.path.join(root, "snapshots", f.read().strip())


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    full: bool          # full attention (1 KV head, replicated under TP)
    num_q: int          # query heads, both halves together
    num_kv: int         # KV heads, both halves together
    head_dim: int

    @property
    def q_half(self):
        return self.num_q // 2

    @property
    def kv_half(self):      # KV heads per half tensor; a full layer's one head can't split
        return 1 if self.full else self.num_kv // 2


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    mlp: int
    vocab: int
    window: int
    eps: float
    softcap: float
    layers: tuple
    rope: dict

    @classmethod
    def load(cls, path=None, num_layers=None):
        path = path or snapshot_dir()
        with open(os.path.join(path, "config.json")) as f:
            t = json.load(f)["text_config"]
        layers = []
        for i, kind in enumerate(t["layer_types"][:num_layers]):
            full = kind == "full_attention"
            layers.append(LayerSpec(
                idx=i, full=full, num_q=t["num_attention_heads"],
                num_kv=t["num_global_key_value_heads"] if full else t["num_key_value_heads"],
                head_dim=t["global_head_dim"] if full else t["head_dim"]))
        return cls(hidden=t["hidden_size"], mlp=t["intermediate_size"], vocab=t["vocab_size"],
                   window=t["sliding_window"], eps=t["rms_norm_eps"],
                   softcap=t["final_logit_softcapping"], layers=tuple(layers),
                   rope=t["rope_parameters"])
