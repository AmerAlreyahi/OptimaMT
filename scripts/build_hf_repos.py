"""Builds one complete, drop-in Hugging Face folder per pair, and models.json.

Each folder holds the original encoder, tokenizer and config files unchanged
(from the pinned ONNX export) and the rebuilt decoder
(onnx/decoder_model_merged_quantized.onnx), plus a model card filled with that
pair's measured numbers. Run after convert.py, verify.py and measure.py.

    python scripts/build_hf_repos.py

Writes work/hf/<repo name>/ and models.json. Repo names and the namespace come
from project.toml.
"""

import csv
import json
import os
import shutil
from string import Template

from common import (CACHE, CFG, DEC, HF, OSES, RESULTS, ROOT, download, repo_id, repo_name, results_url, sentences,
                    sha256, sources, tagline, v1_dir, v2_decoder, write_json)

LANGS = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
         "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
         "ru": "Russian", "th": "Thai", "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese"}
EXPORT_FILES = ["config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "quantize_config.json", "target.spm"]
UPSTREAM_FILES = ["config.json", "generation_config.json", "tokenizer_config.json", "target.spm"]


def spm_scores(path) -> dict:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(path))
    return {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}


def tokenizer_json(d) -> dict:
    """tokenizer.json in the layout of the transformers.js exports: one
    Unigram model over vocab.json (by id), each piece scored with the higher of
    its source.spm / target.spm scores (-100 when in neither)."""
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    src, tgt = spm_scores(d / "source.spm"), spm_scores(d / "target.spm")
    pieces = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in pieces] == list(range(len(pieces)))
    special = {"</s>", "<unk>", "<pad>"}
    added = [{"id": vocab[t], "special": True, "content": t, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False} for t in ("</s>", "<unk>", "<pad>")]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": None},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"}, {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}]},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"Sequence": {"id": "A", "type_id": 0}},
                                      {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                                    {"SpecialToken": {"id": "</s>", "type_id": 0}},
                                    {"Sequence": {"id": "B", "type_id": 0}},
                                    {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                           "special_tokens": {"</s>": {"id": "</s>", "ids": [0], "tokens": ["</s>"]}}},
        "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "model": {"type": "Unigram", "unk_id": 2,
                  "vocab": [[p, 0.0 if p in special else max(src.get(p, -100), tgt.get(p, -100))]
                            for p, _ in pieces]},
    }


def original_folder(e: dict):
    """The pinned original as a complete folder in work/cache/<pair>/. A pair
    exported by Xenova gets its config and tokenizer files from that export; a
    pair exported from Helsinki-NLP's weights gets them from the Helsinki-NLP
    revision, plus tokenizer.json, special_tokens_map.json and the
    tokenizer_config.json fields the transformers.js exports carry."""
    pair = e["pair"]
    d = v1_dir(pair)
    if e["onnx_export"]:
        for f in EXPORT_FILES:
            download(f"https://huggingface.co/{e['onnx_export']}/resolve/{e['onnx_revision']}/{f}", d / f)
        return d
    for f in UPSTREAM_FILES:
        download(f"https://huggingface.co/{e['upstream']}/resolve/{e['upstream_revision']}/{f}", d / f)
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    if "tokenizer_class" not in tc:
        tc = {"clean_up_tokenization_spaces": True, "eos_token": "</s>", "model_max_length": 512,
              "pad_token": "<pad>", "separate_vocabs": False, "source_lang": tc["source_lang"],
              "sp_model_kwargs": {}, "target_lang": tc["target_lang"], "tokenizer_class": "MarianTokenizer",
              "unk_token": "<unk>"}
        (d / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    special = {"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}
    (d / "special_tokens_map.json").write_text(json.dumps(special, indent=2) + "\n", encoding="utf-8")
    text = json.dumps(tokenizer_json(d), indent=2, ensure_ascii=False) + "\n"
    (d / "tokenizer.json").write_text(text, encoding="utf-8")
    return d


def link(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def rows(name: str) -> list[dict]:
    p = RESULTS / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def scheme() -> str:
    return f"{CFG['hf_namespace']}/{repo_name('src-tgt')}".replace("src", "<src>").replace("tgt", "<tgt>")


def card(e: dict, files: list[dict]) -> str:
    pair, s, t = e["pair"], e["source"], e["target"]
    mems = {os_: {r["config"]: float(r["load_mib"]) for r in rows(f"memory_all{suffix}.csv") if r["pair"] == pair}
            for os_, suffix in OSES.items()}
    mems = {k: m for k, m in mems.items() if m}
    same = {r["setting"]: r["identical"] for r in rows("identical.csv") if r["pair"] == pair}
    lines = ["| Memory after load | " + " | ".join(mems) + " | Identical tokens |",
             "|---|" + "---:|" * len(mems) + "---|"]
    for setting, label in (("qdq_off", "`disable_quant_qdq = 1`"), ("default", "ONNX Runtime defaults")):
        cells = [f"{m['v1-' + setting]:.0f} → {m['v2-' + setting]:.0f} MiB "
                 f"({100 * (m['v2-' + setting] / m['v1-' + setting] - 1):+.0f} %)" for m in mems.values()]
        lines.append(f"| {label} | {' | '.join(cells)} | {same.get(setting, '?')} |")
    token = e["lang_token"]
    example = sentences(s)[0]
    js_text = (token + " " if token else "") + example
    base = e["onnx_export"] or e["upstream"]
    origin = (f"[{e['onnx_export']}](https://huggingface.co/{e['onnx_export']}) at revision "
              f"`{e['onnx_revision'][:12]}`" if e["onnx_export"] else
              f"an ONNX export of [{e['upstream']}](https://huggingface.co/{e['upstream']}) at revision "
              f"`{e['upstream_revision'][:12]}` (same layout as the Xenova exports)")
    lang_note = (f"\nStart every input with `{token}` (the target variant). Use transformers.js 3.x: 4.3.2 splits "
                 "this token, with the original export too.\n" if token else "")
    return Template((ROOT / "templates" / "model_card.md").read_text(encoding="utf-8")).substitute(
        name=CFG["name"], long_name=CFG["long_name"], src=s, tgt=t, src_name=LANGS[s], tgt_name=LANGS[t],
        pair=pair, repo_id=repo_id(pair), base=base, origin=origin, upstream=e["upstream"],
        memory_table="\n".join(lines), js_text=js_text, upstream_url=f"https://huggingface.co/{e['upstream']}",
        results_url=results_url(), results_name=results_url().split("spaces/")[1],
        py_lang=f' --lang "{token}"' if token else "", py_text=example,
        ort=json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))["onnxruntime"], lang_note=lang_note,
        github=CFG["github_repo"], count=len(sources()), scheme=scheme(),
        files_note="Only `onnx/decoder_model_merged_quantized.onnx` is rebuilt; every other file is the original, byte "
        "for byte." if e["onnx_export"] else "`onnx/decoder_model_merged_quantized.onnx` is rebuilt; `tokenizer.json`, "
        "`special_tokens_map.json` and `tokenizer_config.json` were written in the Xenova layout; every other file "
        "is the original, byte for byte.",
        vocab=f"{len(json.loads((CACHE / pair / 'vocab.json').read_text(encoding="utf-8"))):,}")


def org_card() -> None:
    """The organization page (a Space named README under the namespace), from templates/org_card.md."""
    import statistics
    sub = {}
    for os_, suffix in OSES.items():
        mem = rows(f"memory_all{suffix}.csv")
        if not mem:
            return
        val = {(r["pair"], r["config"]): float(r["load_mib"]) for r in mem}
        pairs = sorted({p for p, _ in val})
        pre = "" if not suffix else "win_"
        for key, s in (("qdq", "qdq_off"), ("def", "default")):
            sub[f"{pre}{key}_v1"] = f"{statistics.median(val[(p, 'v1-' + s)] for p in pairs):.0f}"
            sub[f"{pre}{key}_v2"] = f"{statistics.median(val[(p, 'v2-' + s)] for p in pairs):.0f}"
            cut = statistics.median(100 * (1 - val[(p, "v2-" + s)] / val[(p, "v1-" + s)]) for p in pairs)
            sub[f"{pre}{key}_pct"] = f"{-cut:.0f}"
    text = Template((ROOT / "templates" / "org_card.md").read_text(encoding="utf-8")).substitute(
        name=CFG["name"], long_name=CFG["long_name"], tagline=tagline(), count=len(sources()),
        results=results_url(), github=CFG["github_repo"], example=repo_id("en-ar"), scheme=scheme(), **sub)
    (HF / "_org_card").mkdir(parents=True, exist_ok=True)
    (HF / "_org_card" / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    models = []
    checked = 0
    for e in sources():
        pair = e["pair"]
        d = original_folder(e)
        if e["onnx_export"]:  # the generator must reproduce the export's own tokenizer.json
            assert tokenizer_json(d) == json.loads((d / "tokenizer.json").read_text(encoding="utf-8")), pair
            checked += 1
        out = HF / repo_name(pair)
        if out.exists():
            shutil.rmtree(out)
        files = []
        for f in sorted(p for p in d.rglob("*") if p.is_file() and not p.name.endswith(".part")):
            rel = f.relative_to(d).as_posix()
            src = v2_decoder(pair) if rel == f"onnx/{DEC}" else f
            link(src, out / rel)
            files.append({"path": rel, "size": src.stat().st_size, "sha256": sha256(src),
                          "changed": rel == f"onnx/{DEC}"})
        (out / "README.md").write_text(card(e, files), encoding="utf-8")
        models.append({"pair": pair, "source": e["source"], "target": e["target"], "repo_id": repo_id(pair),
                       "url": f"https://huggingface.co/{repo_id(pair)}", "upstream": e["upstream"],
                       "onnx_export": e["onnx_export"] or None,
                       "onnx_export_revision": e["onnx_revision"] or e["upstream_revision"],
                       "lang_token": e["lang_token"],
                       "vocab_size": len(json.loads((d / "vocab.json").read_text(encoding="utf-8"))), "files": files})
        print(pair, "->", out.name, flush=True)
    write_json(ROOT / "models.json", {"name": CFG["name"], "namespace": CFG["hf_namespace"],
                                      "repo_id_scheme": scheme(),
                                      "licence": "CC BY 4.0 (model files); MIT (scripts)", "models": models})
    org_card()
    print(f"{len(models)} folders; tokenizer.json generator matches {checked} of {checked} exports")


if __name__ == "__main__":
    main()
