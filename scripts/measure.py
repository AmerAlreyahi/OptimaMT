"""Memory and speed of the original (v1) and the rebuilt (v2) decoders.

Every configuration runs in a fresh process, so one model never inherits
another's memory. Four configurations per pair:

    v1-default   original decoder, ONNX Runtime default session options
    v1-qdq_off   original decoder, session.disable_quant_qdq = 1
    v2-qdq_off   rebuilt decoder,  session.disable_quant_qdq = 1
    v2-default   rebuilt decoder,  default session options

(4 intra-op threads in every case, so timings compare.)

    python scripts/measure.py reference   # 5 pairs: RSS, peaks, ms/step on a 20-line batch
    python scripts/measure.py all         # all pairs: RSS after load

Memory figures are the resident set the encoder and decoder sessions add to
the process (RSS after load minus RSS just before creating them), in MiB.
Results: results/reference.csv, results/memory_all.csv, results/environment.json.
"""

import csv
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (DEC, ENC, REFERENCE, RESULTS, Tok, environment, greedy_timed, sentences, session, source,
                    sources, v1_dir, v2_decoder, write_json)

CONFIGS = ["v1-default", "v1-qdq_off", "v2-qdq_off", "v2-default"]
RUNS = 5


class Memory:
    """Resident set (MiB) and its peak since the last reset."""

    def __init__(self):
        self.linux = sys.platform == "linux"
        if not self.linux:
            import psutil
            self.proc = psutil.Process()
            self.top = 0
            threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while True:
            self.top = max(self.top, self.proc.memory_info().rss)
            time.sleep(0.002)

    def _status(self, field):
        for line in open("/proc/self/status"):
            if line.startswith(field):
                return int(line.split()[1]) / 1024

    def rss(self) -> float:
        return self._status("VmRSS") if self.linux else self.proc.memory_info().rss / 2**20

    def peak(self) -> float:
        return self._status("VmHWM") if self.linux else max(self.top, self.proc.memory_info().rss) / 2**20

    def reset(self):
        if self.linux:
            Path("/proc/self/clear_refs").write_text("5", encoding="utf-8")
        else:
            self.top = 0


def child(pair: str, config: str, full: bool) -> dict:
    import numpy  # noqa: F401  (loaded before the baseline)
    import onnxruntime  # noqa: F401
    e = source(pair)
    d = v1_dir(pair)
    dec = d / "onnx" / DEC if config.startswith("v1") else v2_decoder(pair)
    qdq_off = config.endswith("qdq_off")
    tok = Tok(d, e["lang_token"])
    rows = [tok.encode(s) for s in sentences(e["source"])]
    mem = Memory()
    mem.reset()
    base = mem.rss()
    enc_s, dec_s = session(d / "onnx" / ENC, qdq_off), session(dec, qdq_off)
    out = {"pair": pair, "config": config, "load_mib": round(mem.rss() - base, 1),
           "load_peak_mib": round(mem.peak() - base, 1)}
    if not full:
        return out
    mem.reset()
    greedy_timed(enc_s, dec_s, rows, e["pad"])  # warm-up
    per_step, ids = [], None
    for _ in range(RUNS):
        ids, times = greedy_timed(enc_s, dec_s, rows, e["pad"])
        per_step.append(1000 * sum(times) / len(times))
    out.update({"batch_peak_mib": round(mem.peak() - base, 1), "after_batch_mib": round(mem.rss() - base, 1),
                "lines": len(rows), "steps": len(times), "ms_per_step": round(statistics.median(per_step), 2),
                "ms_per_step_runs": [round(x, 2) for x in per_step],
                "ids_md5": hashlib.md5(json.dumps(ids).encode()).hexdigest()})
    return out


def spawn(pair: str, config: str, full: bool) -> dict:
    args = [sys.executable, __file__, "--child", pair, config, "full" if full else "load"]
    return json.loads(subprocess.check_output(args, text=True).strip().splitlines()[-1])


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2], sys.argv[3], sys.argv[4] == "full")))
        return
    RESULTS.mkdir(exist_ok=True)
    env = environment()
    env["threads"] = 4
    tag = "" if len(sys.argv) < 3 else "_" + sys.argv[2]  # e.g. "windows"
    write_json(RESULTS / f"environment{tag}.json", env)
    if sys.argv[1] == "reference":
        rows = []
        for pair in REFERENCE:  # one at a time: nothing else runs while timing
            got = [spawn(pair, c, True) for c in CONFIGS]
            by = {r["config"]: r["ids_md5"] for r in got}
            for r in got:  # v2 against v1 under the same session options
                r["ids_identical_to_v1"] = r["ids_md5"] == by["v1-" + r["config"].split("-")[1]]
                print(json.dumps({k: v for k, v in r.items() if k != "ms_per_step_runs"}), flush=True)
            rows += got
        write_json(RESULTS / f"reference{tag}.json", rows)
        write_csv(RESULTS / f"reference{tag}.csv", rows)
    elif sys.argv[1] == "all":
        jobs = [(e["pair"], c) for e in sources() for c in CONFIGS]
        with ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(lambda j: spawn(j[0], j[1], False), jobs))
        for r in rows:
            print(json.dumps(r), flush=True)
        write_csv(RESULTS / f"memory_all{tag}.csv", rows)


if __name__ == "__main__":
    main()
