"""Shared helpers: project settings, the pinned sources, downloads, ONNX Runtime
sessions and the greedy decoder every script uses."""

import hashlib
import json
import os
import platform
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from translate import Tokenizer as Tok, greedy  # noqa: F401  (one tokenizer and decoder loop)

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "project.toml").read_text(encoding="utf-8"))
WORK = Path(os.environ.get("WORK_DIR") or ROOT / CFG["work_dir"]).resolve()
CACHE = WORK / "cache"      # pinned original files, in the drop-in layout
BUILD = WORK / "build"      # the rebuilt decoders
HF = WORK / "hf"            # one folder per Hugging Face repo
RESULTS = ROOT / "results"
CHARTS = ROOT / "assets" / "charts"

ENC = "encoder_model_quantized.onnx"
DEC = "decoder_model_merged_quantized.onnx"
REFERENCE = ["en-ar", "ar-en", "en-de", "de-en", "ja-en"]


def repo_name(pair: str) -> str:
    src, tgt = pair.split("-")
    pattern = CFG.get("hf_repo_pattern") or (
        "{src}-{tgt}" if CFG["hf_namespace"].lower() == CFG["name"].lower() else "{slug}-{src}-{tgt}")
    return pattern.format(slug=CFG["slug"], src=src, tgt=tgt)


def repo_id(pair: str) -> str:
    return f"{CFG['hf_namespace']}/{repo_name(pair)}"


OSES = {"Linux": "", "Windows": "_windows"}  # results file suffix per operating system


def _rows(name: str) -> list[dict]:
    import csv
    p = RESULTS / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def headline() -> dict:
    """Ranges over every operating system measured: speedup of disable_quant_qdq = 1 (rebuilt) over the
    defaults (original), median of the reference pairs; memory saved in that mode, per pair."""
    import statistics
    speed, saving = [], []
    for suffix in OSES.values():
        ref = {(r["pair"], r["config"]): float(r["ms_per_step"]) for r in _rows(f"reference{suffix}.csv")}
        if ref:
            pairs = sorted({p for p, _ in ref})
            speed.append(statistics.median(ref[(p, "v1-default")] for p in pairs)
                         / statistics.median(ref[(p, "v2-qdq_off")] for p in pairs))
        mem = {(r["pair"], r["config"]): float(r["load_mib"]) for r in _rows(f"memory_all{suffix}.csv")}
        saving += [100 * (1 - mem[(p, "v2-qdq_off")] / mem[(p, "v1-qdq_off")]) for p, c in mem if c == "v1-qdq_off"]

    def span(xs, d, unit=""):
        lo, hi = f"{min(xs):.{d}f}", f"{max(xs):.{d}f}"
        return lo + unit if lo == hi else f"{lo}\u2013{hi}{unit}"
    return {"speedup": span(speed, 1) if speed else "?", "saving": span(saving, 0, " %") if saving else "?"}


def tagline() -> str:
    return CFG["tagline"].format(**headline())


def used_in() -> str:
    """One markdown line per application of project.toml's [[used_in]]."""
    return "\n".join(f"- [{u['name']}]({u['url']}), {u['what']}, ships these models since {u['since']}."
                      for u in CFG.get("used_in", []))


def collection_url() -> str:
    return CFG.get("hf_collection") or f"https://huggingface.co/{CFG['hf_namespace']}"


def results_url() -> str:
    return f"https://huggingface.co/spaces/{CFG['hf_namespace']}/{CFG['results_space']}"


def sources() -> list[dict]:
    return json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))


def source(pair: str) -> dict:
    return next(e for e in sources() if e["pair"] == pair)


def sentences(lang: str) -> list[str]:
    return json.loads((ROOT / "data" / "sentences.json").read_text(encoding="utf-8"))[lang]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dst: Path, size: int | None = None, sha: str | None = None) -> Path:
    """Downloads `url` to `dst` once; checks the pinned size and sha256 when given."""
    def ok(p: Path) -> bool:
        return p.exists() and (size is None or p.stat().st_size == size) and (sha is None or sha256(p) == sha)
    if ok(dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, tmp)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if not ok(tmp):
        raise SystemExit(f"{url}: size or sha256 differs from the pin")
    tmp.replace(dst)
    return dst


def v1_dir(pair: str) -> Path:
    """The original files of a pair (pinned encoder, decoder, tokenizer)."""
    e = source(pair)
    d = CACHE / pair
    for name, f in e["v1"].items():
        download(f["url"], d / ("onnx/" + name if name.endswith(".onnx") else name), f["size"], f["sha256"])
    return d


def v2_decoder(pair: str) -> Path:
    return BUILD / f"opus-mt-{pair}-{DEC}"


def session(path: Path, qdq_off: bool, threads: int = 4):
    """An ONNX Runtime CPU session with default options, except a fixed thread
    count and, when `qdq_off`, `session.disable_quant_qdq = 1`."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    if qdq_off:
        so.add_session_config_entry("session.disable_quant_qdq", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def greedy_timed(enc, dec, rows, pad, max_steps=128):
    """`translate.greedy`, also returning the seconds of every decoder step."""
    times = []
    return greedy(enc, dec, rows, pad, max_steps, times), times


def environment() -> dict:
    import onnxruntime
    cpu = platform.processor()
    if sys.platform == "linux":
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    elif sys.platform == "win32":
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0] or cpu  # one per socket
    import psutil
    return {"os": platform.platform(), "python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
            "cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def merge_rows(path: Path, new: list[dict]) -> list[dict]:
    """Rows of `path` with `new` replacing the rows of the same pairs (a run over some pairs keeps the others)."""
    import csv
    done = {r["pair"] for r in new}
    old = [r for r in csv.DictReader(path.open(encoding="utf-8"))] if path.exists() else []
    rows = [r for r in old if r["pair"] not in done] + new
    order = {e["pair"]: i for i, e in enumerate(sources())}
    return sorted(rows, key=lambda r: order[r["pair"]])
