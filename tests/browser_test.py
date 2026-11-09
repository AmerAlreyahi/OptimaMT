"""transformers.js in a browser: headless Chromium, onnxruntime-web (WASM, 4 threads), the five reference pairs,
original vs rebuilt decoder, defaults vs disable_quant_qdq = 1. One fresh browser per configuration.

    (cd tests/transformersjs && npm ci)
    CHROMIUM=/path/to/chrome-headless-shell python tests/browser_test.py

Memory: the RSS of all browser processes after loading the model and after the batch, minus the RSS with a blank
page. Speed: model.generate() on the 20-line batch, ms per generated position
(median of 3 runs after a warm-up). Result: results/browser.csv.
"""

import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import CACHE, HF, REFERENCE, RESULTS, repo_name, sentences, source  # noqa: E402

JS = Path(__file__).resolve().parent / "transformersjs"


def main() -> None:
    exe = os.environ["CHROMIUM"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copy(JS / "bench.html", root / "bench.html")
        for pair in REFERENCE:
            (root / "models" / "original").mkdir(parents=True, exist_ok=True)
            (root / "models" / "optimamt").mkdir(parents=True, exist_ok=True)
            (root / "models" / "original" / pair).symlink_to(CACHE / pair)
            (root / "models" / "optimamt" / pair).symlink_to(HF / repo_name(pair))
            e = source(pair)
            lines = root / f"{pair}.json"
            token = e["lang_token"]
            lines.write_text(json.dumps([(token + " " if token else "") + s for s in sentences(e["source"])],
                                        ensure_ascii=False), encoding="utf-8")
            got = {}
            for qdq in ("0", "1"):
                for variant in ("original", "optimamt"):
                    out = subprocess.run(["node", str(JS / "browser_bench.mjs"), str(root), exe, pair, variant, qdq,
                                          str(lines)], capture_output=True, text=True, cwd=JS, check=True).stdout
                    r = json.loads(out.strip().splitlines()[-1])
                    if "error" in r:
                        raise SystemExit(f"{pair} {variant} qdq={qdq}: {r['error']}")
                    got[(variant, qdq)] = r
                    print(json.dumps({k: v for k, v in r.items() if k != "ids"}), flush=True)
            for qdq in ("0", "1"):
                a, b = got[("original", qdq)], got[("optimamt", qdq)]
                rows.append({"pair": pair, "setting": "qdq_off" if qdq == "1" else "default",
                             "loaded_mib_original": round(a["rss_loaded_mib"] - a["rss_before_mib"], 1),
                             "loaded_mib_rebuilt": round(b["rss_loaded_mib"] - b["rss_before_mib"], 1),
                             "after_batch_mib_original": round(a["rss_after_mib"] - a["rss_before_mib"], 1),
                             "after_batch_mib_rebuilt": round(b["rss_after_mib"] - b["rss_before_mib"], 1),
                             "ms_per_step_original": round(a["ms_per_step"], 2),
                             "ms_per_step_rebuilt": round(b["ms_per_step"], 2),
                             "identical": a["ids"] == b["ids"], "chromium": a["chromium"]})
    with (RESULTS / "browser.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for s in ("default", "qdq_off"):
        sel = [r for r in rows if r["setting"] == s]
        print(s, "median memory after load", statistics.median(r["loaded_mib_original"] for r in sel), "->",
              statistics.median(r["loaded_mib_rebuilt"] for r in sel))


if __name__ == "__main__":
    main()
