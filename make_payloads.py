#!/usr/bin/env python3
"""Turn ShareGPT into a JSONL payload file genai-perf can replay.

    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    python make_payloads.py

Writes payloads.jsonl with one {"text": ...} object per line.
Swap in your own prompts here later — same format.
"""
import json

SRC = "ShareGPT_V3_unfiltered_cleaned_split.json"
DST = "payloads.jsonl"
N = 1000
MIN_CHARS = 200          # drop trivial one-liners

rows = json.load(open(SRC))
out = []
for row in rows:
    turns = row.get("conversations", [])
    if not turns:
        continue
    first = turns[0].get("value", "")
    if len(first) >= MIN_CHARS:
        out.append(first)
    if len(out) >= N:
        break

with open(DST, "w") as f:
    for text in out:
        f.write(json.dumps({"text": text}) + "\n")

print(f"wrote {len(out)} prompts to {DST}")
