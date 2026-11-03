"""OpenAI-compatible server, enough for `vllm bench serve` and vllm-bench.sh.

  python -m seesaw.server --mode seesaw       # or --mode tp, the engine's own TP2 baseline
"""
import argparse
import asyncio
import json
import logging
import threading
import time
import uuid

import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from transformers import AutoTokenizer

from .config import BLOCK_SIZE, snapshot_dir
from .scheduler import Request, Scheduler
from .worker import worker_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("seesaw")


def build_app(args, sched_box, results, procs):
    app = FastAPI()
    tok = AutoTokenizer.from_pretrained(snapshot_dir())
    model_id = args.served_name

    @app.on_event("startup")
    async def start():
        loop = asyncio.get_running_loop()
        sched = sched_box["sched"]

        def pump():
            while True:
                r = results.get()
                loop.call_soon_threadsafe(sched.results.put_nowait, r)

        threading.Thread(target=pump, daemon=True).start()
        sched_box["task"] = asyncio.create_task(sched.run())

    @app.get("/health")
    def health():
        task = sched_box.get("task")
        if task is None or task.done():
            raise HTTPException(503, "scheduler stopped")
        return {}

    @app.get("/version")
    def version():
        return {"version": "seesaw-" + args.mode}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "seesaw",
                                            "max_model_len": args.max_len}]}

    @app.get("/metrics")
    def metrics():
        blocks = args.slots * args.max_len // BLOCK_SIZE
        return PlainTextResponse(f'seesaw:cache_config_info{{num_gpu_blocks="{blocks}"}} 1.0\n')

    @app.post("/reset_prefix_cache")
    def reset():
        return {"success": True}        # no prefix cache in this engine

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt = body["prompt"]
        ids = prompt if isinstance(prompt, list) else tok.encode(prompt)
        max_tokens = body.get("max_tokens") or 16
        if len(ids) + max_tokens > args.max_len:
            raise HTTPException(400, f"{len(ids)} + {max_tokens} tokens exceeds {args.max_len}")
        req = Request(ids, max_tokens, body.get("temperature") == 0)
        sched_box["sched"].submit(req)
        rid = "cmpl-" + uuid.uuid4().hex
        made = int(time.time())

        def chunk(text=None, reason=None, usage=None):
            choices = [] if text is None else [{"index": 0, "text": text, "finish_reason": reason}]
            return "data: " + json.dumps({"id": rid, "object": "text_completion", "created": made,
                                          "model": model_id, "choices": choices, "usage": usage}) + "\n\n"

        async def tokens():
            n = 0
            while (got := await req.out.get()) is not None:
                for t in got:
                    n += 1
                    yield n, t

        if not body.get("stream"):
            out = [t async for _, t in tokens()]
            return JSONResponse({
                "id": rid, "object": "text_completion", "created": made, "model": model_id,
                "choices": [{"index": 0, "text": tok.decode(out), "finish_reason": "length",
                             "token_ids": out}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                          "total_tokens": len(ids) + len(out)}})

        async def stream():
            n = 0
            async for n, t in tokens():
                yield chunk(tok.decode([t]), "length" if n == max_tokens else None)
            usage = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}
            yield chunk(usage=usage)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seesaw", "tp"], default="seesaw")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--max-len", type=int, default=7200)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--split", type=int, default=24)
    p.add_argument("--min-switch-batch", type=int, default=4)
    p.add_argument("--admit-wait", type=float, default=0.2)
    p.add_argument("--served-name", default="gemma4-12b")
    args = p.parse_args()

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    cmds = [ctx.Queue(), ctx.Queue()]
    worker_args = {"slots": args.slots, "max_len": args.max_len, "chunk": args.chunk, "split": args.split}
    procs = [ctx.Process(target=worker_main, args=(r, 29411, cmds[r], results, worker_args), daemon=True)
             for r in range(2)]
    for proc in procs:
        proc.start()
    for _ in range(2):
        r = results.get()
        if r[0] != "ready":
            raise SystemExit(r[2])
    log.info("workers ready")
    sched = Scheduler(cmds, args.slots, args.mode, args.min_switch_batch, args.admit_wait)
    box = {"sched": sched}
    uvicorn.run(build_app(args, box, results, procs), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
