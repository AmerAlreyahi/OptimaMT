"""Proves the Hugging Face folders are drop-in replacements.

For three pairs exported by Xenova it translates the same sentences with
  - transformers.js from the local folder, from the original files and from the
    original Xenova repo on the Hub (pinned revision), and
  - ONNX Runtime Python (scripts/translate.py) from the local folder and from
    the original files,
and requires identical output. One pair exported from Helsinki-NLP's weights
is checked the same way against its original files. It also records the
Node.js process memory after loading each model with transformers.js, with
the default session options and with session.disable_quant_qdq = 1.

    (cd tests/transformersjs && npm ci)
    python tests/dropin_test.py

Result: results/dropin_transformersjs-<version>.json.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import CACHE, HF, RESULTS, repo_name, sentences, source, write_json  # noqa: E402
from translate import translate  # noqa: E402

PAIRS = ["en-ar", "de-en", "ja-en", "ar-de"]
JS = Path(__file__).resolve().parent / "transformersjs"


def node(*args, qdq_off=False) -> dict:
    env = dict(os.environ, QDQ_OFF="1" if qdq_off else "0")
    out = subprocess.run(["node", str(JS / "translate.mjs"), *args], capture_output=True, text=True, cwd=JS,
                         check=True, env=env).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for pair in PAIRS:
            e = source(pair)
            token = e["lang_token"]
            texts = sentences(e["source"])[:3]
            lines = Path(tmp) / f"{pair}.json"
            inputs = [(token + " " if token else "") + t for t in texts]
            lines.write_text(json.dumps(inputs, ensure_ascii=False), encoding="utf-8")
            ours_js = node("local", str(HF), repo_name(pair), str(lines))
            orig_js = node("local", str(CACHE), pair, str(lines))
            hub_js = node("hub", e["onnx_export"], e["onnx_revision"], str(lines)) if e["onnx_export"] else orig_js
            ours_q = node("local", str(HF), repo_name(pair), str(lines), qdq_off=True)
            orig_q = node("local", str(CACHE), pair, str(lines), qdq_off=True)
            ours_py = translate(HF / repo_name(pair), texts, token)
            orig_py = translate(CACHE / pair, texts, token)
            row = {"pair": pair,
                   "original": f"{e['onnx_export']}@{e['onnx_revision'][:12]}" if e["onnx_export"]
                   else f"{e['upstream']}@{e['upstream_revision'][:12]} (ONNX export)",
                   "sentences": texts,
                   "transformers_js": ours_js["translations"],
                   "transformers_js_identical": ours_js["translations"] == orig_js["translations"]
                   == hub_js["translations"],
                   "onnxruntime_python": ours_py,
                   "onnxruntime_python_identical": ours_py == orig_py,
                   "transformers_js_qdq_off_identical": ours_q["translations"] == orig_q["translations"]
                   == orig_js["translations"],
                   "node_rss_loaded_mib": {"original": orig_js["rss_loaded_mib"], "rebuilt": ours_js["rss_loaded_mib"],
                                           "original_qdq_off": orig_q["rss_loaded_mib"],
                                           "rebuilt_qdq_off": ours_q["rss_loaded_mib"]}}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
    pkg = json.loads((JS / "node_modules/@huggingface/transformers/package.json").read_text(encoding="utf-8"))
    ort = json.loads((JS / "node_modules/onnxruntime-node/package.json").read_text(encoding="utf-8"))
    write_json(RESULTS / f"dropin_transformersjs-{pkg['version']}.json",
               {"transformers_js": pkg["version"], "onnxruntime_node": ort["version"], "rows": rows})
    if not all(r["transformers_js_identical"] and r["transformers_js_qdq_off_identical"]
               and r["onnxruntime_python_identical"] for r in rows):
        raise SystemExit("output differs")


if __name__ == "__main__":
    main()
