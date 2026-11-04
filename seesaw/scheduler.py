"""Scheduling: fill the GPUs in PP, switch once, decode to completion in TP.

--mode tp and --mode pp never switch: they are the same-engine baselines.

Prefill keeps taking new requests until the queue is empty or the slots run out,
so one switch serves as many sequences as possible. Decode then runs the batch to
the end; `--ignore-eos` with fixed lengths ends a whole batch on the same step.
"""
import asyncio
import collections
import logging
import time

log = logging.getLogger("seesaw")


class Request:
    def __init__(self, tokens, max_tokens, greedy):
        self.tokens, self.max_tokens, self.greedy = tokens, max_tokens, greedy
        self.out = asyncio.Queue()
        self.first = None
        self.slot = None


class Scheduler:
    def __init__(self, cmds, slots, mode, min_switch_batch, admit_wait):
        self.cmds, self.mode = cmds, mode
        self.free = list(range(slots - 1, -1, -1))
        self.min_switch = min_switch_batch
        self.admit_wait = admit_wait
        self.waiting = collections.deque()
        self.wake = asyncio.Event()
        self.active = {}             # slot -> Request
        self.layout = "pp" if mode == "pp" else "tp"
        self.results = asyncio.Queue()

    def submit(self, req):
        self.waiting.append(req)
        self.wake.set()

    def send(self, *cmd):
        for q in self.cmds:
            q.put(cmd)

    async def expect(self, kind):
        """Handle results until one of `kind` arrives, and return it."""
        while True:
            r = await self.results.get()
            if r[0] == kind:
                return r
            if r[0] == "first":
                _, slot, tok = r
                req = self.active[slot]
                req.first = tok
                req.out.put_nowait([tok])
            elif r[0] == "tokens":
                _, slots, ids = r
                for s, t in zip(slots, ids):
                    self.active[s].out.put_nowait([t])
            elif r[0] == "error":
                raise RuntimeError(f"worker {r[1]} failed:\n{r[2]}")

    async def switch(self, to):
        if self.layout == to:
            return 0.0
        self.send("to_" + to)
        info = (await self.expect("switched"))[1]
        self.layout = to
        return info[1]

    def admit(self):
        """Move waiting requests into slots."""
        batch = []
        while self.waiting and self.free:
            req = self.waiting.popleft()
            req.slot = self.free.pop()
            self.active[req.slot] = req
            batch.append(req)
        return batch

    async def run(self):
        while True:
            while not self.waiting:
                self.wake.clear()
                await self.wake.wait()
            await self.round()

    async def collect(self):
        """Closed-loop clients send their next requests together; let them land."""
        seen = -1
        while len(self.waiting) != seen and len(self.waiting) < len(self.free):
            seen = len(self.waiting)
            await asyncio.sleep(self.admit_wait)

    async def round(self):
        t0 = time.perf_counter()
        await self.collect()
        batch = self.admit()
        use_pp = self.mode == "pp" or (self.mode == "seesaw" and len(batch) >= self.min_switch)
        sw = await self.switch("pp" if use_pp else "tp")
        total = 0
        while batch:
            total += len(batch)
            self.send("prefill", [(r.slot, r.tokens) for r in batch], batch[0].greedy)
            await self.expect("prefill_done")
            await self.expect("prefill_done")
            batch = self.admit()       # keep prefilling while there is work and room
        t1 = time.perf_counter()
        seqs = []
        for req in self.active.values():
            if req.max_tokens > 1:
                seqs.append((req.slot, len(req.tokens), req.first, req.max_tokens - 1))
            else:
                req.out.put_nowait(None)
        if seqs:
            if self.mode != "pp":
                sw += await self.switch("tp")
            self.send("decode", seqs, next(iter(self.active.values())).greedy)
            await self.expect("decode_done")
        t3 = time.perf_counter()
        slots = list(self.active)
        self.send("release", slots)
        for s, req in list(self.active.items()):
            if req.max_tokens > 1:
                req.out.put_nowait(None)
            self.free.append(s)
        self.active.clear()
        log.info("round B=%d layout=%s prefill %.2fs switch %.2fs decode %.2fs",
                 total, "pp" if use_pp else "tp", t1 - t0, sw, t3 - t1 - sw)
