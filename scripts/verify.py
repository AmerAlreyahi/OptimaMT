"""Checks that the rebuilt decoder produces exactly the original's tokens.

For every pair and both session settings (ONNX Runtime defaults, and
session.disable_quant_qdq = 1) it greedy-decodes the fixed sentences of
data/sentences.json with the original and the rebuilt decoder and compares the
token ids one by one.

    python scripts/verify.py              # all pairs
    python scripts/verify.py en-ar ja-en

Results: results/identical.csv and results/translations.json.
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor

from common import (DEC, ENC, RESULTS, Tok, greedy, merge_rows, sentences, session, source, sources, v1_dir, v2_decoder,
                    write_json)


def check(pair: str) -> list[dict]:
    e = source(pair)
    d = v1_dir(pair)
    tok = Tok(d, e["lang_token"])
    lines = sentences(e["source"])
    rows = [tok.encode(s) for s in lines]
    out = []
    for qdq_off in (False, True):
        enc = session(d / "onnx" / ENC, qdq_off, threads=2)
        old = greedy(enc, session(d / "onnx" / DEC, qdq_off, threads=2), rows, e["pad"])
        new = greedy(enc, session(v2_decoder(pair), qdq_off, threads=2), rows, e["pad"])
        out.append({"pair": pair, "setting": "qdq_off" if qdq_off else "default", "lines": len(rows),
                    "tokens": sum(map(len, old)), "identical": old == new,
                    "translations": [{"source": s, "v2": tok.decode(n)} for s, n in zip(lines, new)]})
    return out


def main() -> None:
    pairs = sys.argv[1:] or [e["pair"] for e in sources()]
    with ProcessPoolExecutor(5) as pool:
        rows = [r for rs in pool.map(check, pairs) for r in rs]
    RESULTS.mkdir(exist_ok=True)
    new = [{"pair": r["pair"], "setting": r["setting"], "lines": r["lines"], "tokens": r["tokens"],
            "identical": "yes" if r["identical"] else "no"} for r in rows]
    merged = merge_rows(RESULTS / "identical.csv", new)  # a run over some pairs keeps the rest
    with (RESULTS / "identical.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["pair", "setting", "lines", "tokens", "identical"], lineterminator="\n")
        w.writeheader()
        w.writerows(merged)
    tr = json.loads((RESULTS / "translations.json").read_text(encoding="utf-8")) if pairs != [e["pair"] for e in sources()] else {}
    tr.update({r["pair"]: r["translations"] for r in rows if r["setting"] == "default"})
    write_json(RESULTS / "translations.json", {e["pair"]: tr[e["pair"]] for e in sources() if e["pair"] in tr})
    same = sum(r["identical"] for r in rows)
    print(f"{same} of {len(rows)} pair/setting runs give identical tokens")
    if same != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
