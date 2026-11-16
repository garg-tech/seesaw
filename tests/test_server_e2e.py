"""End to end against a running server: HTTP in, tokens out. Standard library only.

  scripts/serve-engine.sh seesaw --split 28                     # on the GPU machine
  python tests/test_server_e2e.py [base_url]                    # default $BASE_URL or localhost:8000

Cases:
  1. /health, /version and /v1/models answer, and the model is listed
  2. a question the tests never used before, in Gemma 4's chat format, greedy: the
     answer is right and the usage counts are exact
  3. the same prompt streamed gives the same text as the non-streamed reply
  4. max_tokens=1 returns exactly one token
  5. malformed requests are refused with HTTP 400 and the server keeps serving: a
     prompt longer than max_len, a list of strings, an out-of-range token id, an
     empty prompt, max_tokens=0
  6. 40 concurrent 1500-token requests, more than the 32 slots: all finish with all
     their tokens. In --mode seesaw this batch switches to PP and back; the engine
     log shows "round B=32 layout=pp" then "round B=8 layout=pp"
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BASE_URL", "http://localhost:8000")).rstrip("/")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, json.loads(r.read() or b"{}")


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def complete(model, prompt, max_tokens, stream=False):
    status, raw = post("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                           "temperature": 0, "stream": stream})
    assert status == 200, (status, raw[:300])
    if not stream:
        return json.loads(raw)
    text, usage = "", None
    for line in raw.decode().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            text += "".join(c["text"] for c in chunk["choices"])
            usage = chunk.get("usage") or usage
    return {"choices": [{"text": text}], "usage": usage}


def main():
    ok = lambda msg: print(f"  ok  {msg}", flush=True)

    # 1. health and model list
    status, _ = get("/health")
    assert status == 200
    _, ver = get("/version")
    _, models = get("/v1/models")
    model = models["data"][0]["id"]
    max_len = models["data"][0].get("max_model_len", 7200)
    ok(f"health 200, version {ver.get('version')}, model {model}, max_len {max_len}")

    # 2. an unseen question, greedy, in the chat format the instruction-tuned model expects
    prompt = ("<bos><|turn>user\nWhich planet in our solar system is known as the Red Planet?"
              "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    t0 = time.time()
    r = complete(model, prompt, 12)
    text = r["choices"][0]["text"]
    assert r["usage"]["completion_tokens"] == 12, r["usage"]
    assert len(r["choices"][0]["token_ids"]) == 12
    assert r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + 12
    assert "Mars" in text, f"expected 'Mars' in {text!r}"
    ok(f"greedy reply {text!r} in {time.time() - t0:.1f} s, usage {r['usage']}")

    # 3. streaming returns the same text
    s = complete(model, prompt, 12, stream=True)
    assert s["choices"][0]["text"] == text, (s["choices"][0]["text"], text)
    assert s["usage"]["completion_tokens"] == 12
    ok("streamed text equals the non-streamed text")

    # 4. max_tokens=1
    r = complete(model, prompt, 1)
    assert r["usage"]["completion_tokens"] == 1 and len(r["choices"][0]["token_ids"]) == 1
    ok("max_tokens=1 gives one token")

    # 5. malformed requests: refused before they reach the GPU workers, server keeps serving
    bad = {
        f"{max_len}-token prompt": {"prompt": [1000] * max_len, "max_tokens": 16},
        "list of strings": {"prompt": ["input_ids", "attention_mask"], "max_tokens": 4},
        "token id past the vocabulary": {"prompt": [1, 10 ** 7], "max_tokens": 4},
        "empty prompt": {"prompt": "", "max_tokens": 4},
        "max_tokens=0": {"prompt": "Hello", "max_tokens": 0},
    }
    for name, body in bad.items():
        status, raw = post("/v1/completions", {"model": model, **body})
        assert status == 400, (name, status, raw[:200])
        assert get("/health")[0] == 200, name
    assert complete(model, prompt, 2)["usage"]["completion_tokens"] == 2
    ok(f"{len(bad)} malformed requests refused with 400; server still serves")

    # 6. more concurrent requests than slots
    rng = random.Random(0)
    prompts = [[rng.randrange(1000, 200000) for _ in range(1500)] for _ in range(40)]
    t0 = time.time()
    with ThreadPoolExecutor(40) as pool:
        outs = list(pool.map(lambda p: complete(model, p, 16), prompts))
    counts = [o["usage"]["completion_tokens"] for o in outs]
    assert counts == [16] * 40, counts
    ok(f"40 concurrent requests, 16 tokens each, in {time.time() - t0:.1f} s")
    print("PASS")


if __name__ == "__main__":
    main()
