"""First N layers on one GPU against the same layers in transformers.

The full model can't load in 15 GB of host RAM, so this truncates both sides.
Six layers cover the sliding and full paths. Prompts are longer than the window
so block recycling is exercised.

  python tests/test_layers_vs_hf.py [gpu] [layers]
"""
import dataclasses
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4_unified import modeling_gemma4_unified as hf

sys.path.insert(0, ".")
from seesaw.config import ModelConfig, snapshot_dir
from seesaw.kv_cache import Batch, BlockTables
from seesaw.model.gemma4 import Decoder

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LEN = 1600
DEV = torch.device("cuda", GPU)
torch.cuda.set_device(GPU)    # Triton compiles for the current device


def reference(tokens, dtype=torch.float32):
    snap = snapshot_dir()
    cfg = AutoConfig.from_pretrained(snap).text_config
    cfg.num_hidden_layers = N
    cfg.layer_types = cfg.layer_types[:N]
    cfg._attn_implementation = "eager"
    with torch.device(DEV):
        model = hf.Gemma4UnifiedTextModel(cfg)
    sd = {}
    import glob, os
    with safe_open(glob.glob(os.path.join(snap, "*.safetensors"))[0], "pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                short = k[len("model.language_model."):]
                if not short.startswith("layers.") or int(short.split(".")[1]) < N:
                    sd[short] = f.get_tensor(k).to(DEV, dtype)
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype != torch.float32):
            out = model(input_ids=tokens[None], use_cache=False).last_hidden_state[0].float().clone()
    del model, sd
    torch.cuda.empty_cache()
    return out


def run(dec, tables, tokens, cuts):
    """Prefill tokens in chunks ending at each cut; return hidden states."""
    slot, done, outs = 0, 0, []
    for end in cuts:
        tables.extend(slot, end)
        b = Batch(tables, [slot], [done], [end - done], DEV)
        with torch.inference_mode():
            y = dec.forward(dec.embed(tokens[done:end]), b)
            outs.append(dec.normed(y))
        tables.trim(slot, end)
        done = end
    tables.release(slot)
    return torch.cat(outs).float()


def main():
    torch.manual_seed(0)
    tokens = torch.randint(1000, 200000, (LEN,), device=DEV)
    ref = reference(tokens)
    noise = (reference(tokens, torch.bfloat16) - ref).abs().mean() / ref.abs().mean()
    print(f"rounding floor (HF bf16 vs HF fp32): {noise:.2e}")
    cfg = ModelConfig.load(num_layers=N)
    tables = BlockTables(slots=2, max_len=2048, window=cfg.window, scratch_blocks=256)
    dec = Decoder(cfg, snapshot_dir(), DEV, range(N), tables, max_len=2048)
    cases = {
        "one chunk": [LEN],
        "chunks": [700, 1400, LEN],
        "decode tail": [1500] + list(range(1501, LEN + 1)),
    }
    print(f"ref: mean |y| {ref.abs().mean():.4f}")
    bad = 0
    for name, cuts in cases.items():
        y = run(dec, tables, tokens, cuts)
        err = (y - ref).abs()
        rel = (err.mean() / ref.abs().mean()).item()
        print(f"{name:12} mean rel err {rel:.2e}  max abs {err.max():.3f}  free blocks {len(tables.free)}")
        per = err.mean(-1) / ref.abs().mean()
        w = per.topk(5)
        print("   worst positions", w.indices.tolist(), [round(v, 3) for v in w.values.tolist()],
              " by 200s", [round(per[i:i + 200].mean().item(), 4) for i in range(0, LEN, 200)])
        bad += rel > 0.02
    assert len(tables.free) == tables.num_sliding, "leaked blocks"
    print("FAIL" if bad else "PASS")
    sys.exit(bad != 0)


main()
