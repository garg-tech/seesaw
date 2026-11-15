"""The scheduler's switch policy and request handling, with fake workers. No GPU needed.

  python tests/test_scheduler.py        # or: pytest tests/test_scheduler.py

Checks, for --mode seesaw unless named:
  - one request, or fewer than --min-switch-batch, is prefilled in TP: no re-shard
  - a batch of at least --min-switch-batch switches to PP for prefill, back to TP to decode
  - more requests than slots run in several rounds, and every request gets all its tokens
  - max_tokens=1 needs no decode, so no switch back to TP
  - --mode tp never re-shards, --mode pp starts in PP and never re-shards
"""
import asyncio
import sys

sys.path.insert(0, ".")
from seesaw.scheduler import Request, Scheduler


class FakeWorker:
    """Records commands; rank 0 answers the way the real workers do together."""

    def __init__(self, answers):
        self.answers, self.log = answers, []

    def put(self, cmd):
        self.log.append(cmd[0])
        if self.answers is None:
            return
        q = self.answers.results
        if cmd[0] == "prefill":
            for slot, tokens in cmd[1]:
                q.put_nowait(("first", slot, len(tokens) % 1000))
            q.put_nowait(("prefill_done", 0))
            q.put_nowait(("prefill_done", 1))
        elif cmd[0] in ("to_tp", "to_pp"):
            q.put_nowait(("switched", (cmd[0], 0.0)))
        elif cmd[0] == "decode":
            seqs = cmd[1]
            left = {s[0]: s[3] for s in seqs}
            while left:
                q.put_nowait(("tokens", list(left), [7] * len(left)))
                left = {s: n - 1 for s, n in left.items() if n > 1}
            q.put_nowait(("decode_done",))


def serve(mode, n_requests, max_tokens=4, slots=32, min_switch=4):
    """Submit n requests together; return (commands seen by rank 0, tokens per request)."""

    async def main():
        w0, w1 = FakeWorker(None), FakeWorker(None)
        sched = Scheduler([w0, w1], slots, mode, min_switch, admit_wait=0.01)
        w0.answers = sched
        task = asyncio.create_task(sched.run())
        reqs = [Request(list(range(10 + i)), max_tokens, True) for i in range(n_requests)]
        for r in reqs:
            sched.submit(r)

        async def drain(r):
            out = []
            while (got := await r.out.get()) is not None:
                out += got
            return out

        outs = await asyncio.wait_for(asyncio.gather(*(drain(r) for r in reqs)), 10)
        task.cancel()
        assert w0.log == w1.log, "both workers must see the same commands"
        return w0.log, [len(o) for o in outs]

    return asyncio.run(main())


def test_single_request_stays_tp():
    log, toks = serve("seesaw", 1)
    assert "to_pp" not in log and "to_tp" not in log, log
    assert toks == [4]


def test_small_batch_stays_tp():
    log, toks = serve("seesaw", 3)
    assert "to_pp" not in log, log
    assert toks == [4] * 3


def test_batch_switches_pp_then_tp():
    log, toks = serve("seesaw", 8)
    assert log[:4] == ["to_pp", "prefill", "to_tp", "decode"], log
    assert toks == [4] * 8


def test_more_requests_than_slots():
    log, toks = serve("seesaw", 40, slots=32)
    # Round 1: 32 in PP. Round 2: the other 8, also >= 4, so PP again.
    assert log.count("to_pp") == 2 and log.count("decode") == 2, log
    assert toks == [4] * 40


def test_max_tokens_one():
    log, toks = serve("seesaw", 8, max_tokens=1)
    assert "decode" not in log and "to_tp" not in log, log
    assert toks == [1] * 8


def test_tp_mode_never_switches():
    log, toks = serve("tp", 8)
    assert "to_pp" not in log and "to_tp" not in log, log
    assert toks == [4] * 8


def test_pp_mode_never_switches():
    log, toks = serve("pp", 8)
    assert "to_pp" not in log and "to_tp" not in log, log
    assert toks == [4] * 8


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"PASS ({len(tests)} tests)")
