#!/usr/bin/env python3
"""Record greedy vLLM outputs as the correctness reference for the full model.

Start vLLM (CONFIG=tp2 ./serve.sh), then on the box:
  python bench/capture_golden.py          # writes tests/golden/golden.json

Real text keeps the model confident, so bf16 rounding can't flip a greedy pick.
The long prompts go past the sliding window and match the benchmark's input.
"""
import json
import os
import random
import urllib.request

from transformers import AutoTokenizer

URL = "http://localhost:8000/v1/completions"
MODEL = "gemma4-12b"
OUT_TOKENS = 64

PASSAGE = (
    "The history of computing is a story of abstraction layered on abstraction. Early "
    "machines were programmed by rewiring plugboards, then by punching cards, then by "
    "typing assembly language. Each step let people think less about the machine and more "
    "about the problem. Compilers made programs portable, operating systems shared the "
    "hardware between users, and networks joined separate computers into one system. "
)
SHORT = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Water is made of hydrogen and oxygen. The chemical formula of water is",
]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    prompts = [tok.encode(p) for p in SHORT]
    unit = tok.encode(PASSAGE, add_special_tokens=False)
    for n in (1500, 6144):
        ids = (unit * (n // len(unit) + 1))[:n - 1]
        prompts.append([tok.bos_token_id] + ids)
    rng = random.Random(0)
    prompts.append([rng.randrange(1000, 200000) for _ in range(6144)])

    cases = []
    for ids in prompts:
        r = post({"model": MODEL, "prompt": ids, "max_tokens": OUT_TOKENS, "temperature": 0,
                  "ignore_eos": True, "return_token_ids": True, "logprobs": 5})
        c = r["choices"][0]
        lp = c["logprobs"]
        cases.append({"prompt": ids, "tokens": c["token_ids"],
                      "top": lp["top_logprobs"], "logprobs": lp["token_logprobs"]})
        print(len(ids), repr(c["text"][:70]))
    os.makedirs("tests/golden", exist_ok=True)
    with open("tests/golden/golden.json", "w") as f:
        json.dump(cases, f)
    print("wrote", len(cases), "cases")


if __name__ == "__main__":
    main()
