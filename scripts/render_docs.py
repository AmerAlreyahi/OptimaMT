"""Writes README.md and REPORT.md from templates/ and the measured data in
results/, so every number in them is the measured one and the project name
comes from project.toml.

    python scripts/render_docs.py
"""

import csv
import json
import statistics
from string import Template

from common import (CFG, REFERENCE, RESULTS, ROOT, collection_url, headline, repo_id, results_url, sources, tagline,
                    used_in)

SETTINGS = {"default": "ONNX Runtime defaults", "qdq_off": "`disable_quant_qdq = 1`"}


def read(name: str) -> list[dict]:
    p = RESULTS / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def pct(new: float, old: float) -> str:
    return f"{100 * (new - old) / old:+.0f} %"


def med(xs) -> float:
    return statistics.median(xs)


def main() -> None:
    same = {(r["pair"], r["setting"]): r for r in read("identical.csv")}
    models = {m["pair"]: m for m in json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["models"]}
    graph = json.loads((RESULTS / "graph.json").read_text(encoding="utf-8"))
    dropin = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(RESULTS.glob("dropin_transformersjs-*.json"))]
    src = sources()
    pairs = [e["pair"] for e in src]
    v = {}

    # --- per operating system: whole-set summary, reference medians and table (keys prefixed "win_" etc.)
    def arrow(x, y, d=0):
        return f"{x:.{d}f} → {y:.{d}f}"

    def per_os(suffix: str, pre: str) -> None:
        ref = {(r["pair"], r["config"]): r for r in read(f"reference{suffix}.csv")}
        mem = {(r["pair"], r["config"]): float(r["load_mib"]) for r in read(f"memory_all{suffix}.csv")}
        env = json.loads((RESULTS / f"environment{suffix}.json").read_text(encoding="utf-8"))
        v.update({pre + "ort": env["onnxruntime"], pre + "os": env["os"], pre + "cpu": env["cpu"],
                  pre + "cores": str(env["logical_cpus"]), pre + "ram": str(env["ram_gb"]),
                  pre + "python": env["python"], pre + "threads": str(env["threads"])})
        for s in SETTINGS:
            saved = [mem[(p, "v1-" + s)] - mem[(p, "v2-" + s)] for p in pairs]
            share = [100 * (mem[(p, "v1-" + s)] - mem[(p, "v2-" + s)]) / mem[(p, "v1-" + s)] for p in pairs]
            v[f"{pre}{s}_saved_min"] = f"{min(saved):.0f}"
            v[f"{pre}{s}_saved_max"] = f"{max(saved):.0f}"
            v[f"{pre}{s}_pct_med"] = f"{med(share):.0f}"
            v[f"{pre}{s}_pct_min"] = f"{min(share):.0f}"
            v[f"{pre}{s}_pct_max"] = f"{max(share):.0f}"
            v[f"{pre}{s}_v1_med"] = f"{med(mem[(p, 'v1-' + s)] for p in pairs):.0f}"
            v[f"{pre}{s}_v2_med"] = f"{med(mem[(p, 'v2-' + s)] for p in pairs):.0f}"
        for c in ("v1-default", "v1-qdq_off", "v2-qdq_off", "v2-default"):
            key = c.replace("-", "_")
            for name, field, d in (("ms", "ms_per_step", 1), ("peak", "batch_peak_mib", 0), ("load", "load_mib", 0),
                                   ("loadpeak", "load_peak_mib", 0)):
                v[f"{pre}{name}_{key}"] = f"{med(float(ref[(p, c)][field]) for p in REFERENCE):.{d}f}"
        v[pre + "speedup"] = f"{float(v[pre + 'ms_v1_default']) / float(v[pre + 'ms_v2_qdq_off']):.1f}"
        v[pre + "peak_pct"] = pct(float(v[pre + "peak_v2_qdq_off"]), float(v[pre + "peak_v1_qdq_off"]))
        v[pre + "ref_identical"] = "yes" if all(r["ids_identical_to_v1"] == "True" for r in ref.values()) else "no"
        t = ["| Pair | Settings | After load (MiB) | Peak, 20 lines (MiB) | ms/step | Identical |",
             "|---|---|---:|---:|---:|---|"]
        for p in REFERENCE:
            for s, label in SETTINGS.items():
                a, b = ref[(p, "v1-" + s)], ref[(p, "v2-" + s)]
                f = lambda k, d=0: arrow(float(a[k]), float(b[k]), d)  # noqa: E731
                t.append(f"| {p} | {label} | {f('load_mib')} ({pct(float(b['load_mib']), float(a['load_mib']))}) "
                         f"| {f('batch_peak_mib')} | {f('ms_per_step', 1)} "
                         f"| {'yes' if b['ids_identical_to_v1'] == 'True' else 'no'} |")
        v[pre + "reference_table"] = "\n".join(t)
        if not pre:
            v["mem"] = mem

    per_os("", "")
    per_os("_windows", "win_")
    mem = v.pop("mem")
    env = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    runs = list(same.values())
    v["identical_runs"] = str(sum(r["identical"] == "yes" for r in runs))
    v["all_runs"] = str(len(runs))
    v["identical_tokens"] = f"{sum(int(r['tokens']) for r in runs):,}"
    v["pairs"] = str(len(pairs))
    v["ref_pairs"] = ", ".join(REFERENCE)

    t = ["| Pair | Vocabulary | Defaults (MiB) | qdq off (MiB) | Identical | Model |", "|---|---:|---:|---:|---|---|"]
    for e in src:
        p = e["pair"]
        cells = [f"{arrow(mem[(p, 'v1-' + s)], mem[(p, 'v2-' + s)])} ({pct(mem[(p, 'v2-' + s)], mem[(p, 'v1-' + s)])})"
                 for s in SETTINGS]
        ids = "yes" if all(same[(p, s)]["identical"] == "yes" for s in SETTINGS) else "no"
        t.append(f"| {p} | {models[p]['vocab_size']:,} | {' | '.join(cells)} | {ids} "
                 f"| [{repo_id(p)}](https://huggingface.co/{repo_id(p)}) |")
    v["all_table"] = "\n".join(t)

    t = ["| transformers.js | Pair | Same as original (defaults / qdq off) | Python same | Node RSS, defaults (MiB) "
         "| Node RSS, qdq off (MiB) |", "|---|---|---|---|---:|---:|"]
    for d in dropin:
        for r in d["rows"]:
            m = r["node_rss_loaded_mib"]
            t.append(f"| {d['transformers_js']} | {r['pair']} | "
                     f"{'yes' if r['transformers_js_identical'] else 'no'} / "
                     f"{'yes' if r['transformers_js_qdq_off_identical'] else 'no'} | "
                     f"{'yes' if r['onnxruntime_python_identical'] else 'no'} | "
                     f"{arrow(m['original'], m['rebuilt'])} | {arrow(m['original_qdq_off'], m['rebuilt_qdq_off'])} |")
    v["dropin_table"] = "\n".join(t)
    v["tjs_versions"] = " and ".join(d["transformers_js"] for d in dropin)

    # --- the graph (from results/graph.json)
    o, n = graph["original"], graph["rebuilt"]
    tr = next(x for x in o["projection_nodes"] if x["op"] == "Transpose")
    dq = next(x for x in o["projection_nodes"] if x["op"] == "DequantizeLinear")
    v.update(g_pair=graph["pair"], g_if=o["if_node"], g_cond=o["if_input"], g_b1=o["branches"][0],
             g_b2=o["branches"][1], g_tr=tr["name"], g_tr_in=tr["inputs"][0], g_tr_out=tr["outputs"][0],
             g_dq=dq["name"], g_dq_scale=dq["inputs"][1], g_dq_zp=dq["inputs"][2], g_dq_out=dq["outputs"][0],
             g_branches_orig=", ".join(sorted({x["graph"] for x in o["projection_nodes"]})),
             g_branches_new=", ".join(sorted({x["graph"] for x in n["projection_nodes"]})))
    w = next(x for x in o["large_initializers"] if x["name"] == tr["inputs"][0])
    v["g_weight"] = f"{w['dtype']} {w['shape']} ({w['mib']} MiB)"

    def held(rows):
        return "<br>".join(f"{x['graph']}: `{x['name']}` {x['dtype']} {x['mib']} MiB" for x in rows
                           if "shared" in x["name"])
    v["g_held_table"] = "\n".join([
        "| Decoder | After ONNX Runtime's optimizations, defaults | With `disable_quant_qdq = 1` |",
        "|---|---|---|",
        f"| original | {held(o['after_ort_default'])} | {held(o['after_ort_qdq_off'])} |",
        f"| rebuilt | {held(n['after_ort_default'])} | {held(n['after_ort_qdq_off'])} |"])

    # --- optional: the same reference measurement with another ONNX Runtime version
    other = [f for f in sorted(RESULTS.glob("reference_ort*.csv"))]
    t = []
    for f in other:
        rows = {(r["pair"], r["config"]): r for r in csv.DictReader(f.open(encoding="utf-8"))}
        ver = f.name.replace("reference", "environment").replace(".csv", ".json")
        ver = json.loads((RESULTS / ver).read_text(encoding="utf-8"))
        for s, label in SETTINGS.items():
            a = med(float(rows[(p, "v1-" + s)]["load_mib"]) for p in REFERENCE)
            b = med(float(rows[(p, "v2-" + s)]["load_mib"]) for p in REFERENCE)
            ms = med(float(rows[(p, "v2-" + s)]["ms_per_step"]) for p in REFERENCE)
            ids = all(rows[(p, "v2-" + s)]["ids_identical_to_v1"] == "True" for p in REFERENCE)
            t.append(f"| {ver['onnxruntime']} | {label} | {a:.0f} | {b:.0f} | {b - a:+.0f} ({pct(b, a)}) | {ms:.1f} "
                     f"| {'yes' if ids else 'no'} |")
    for s, label in SETTINGS.items():
        a, b = float(v[f"load_v1_{s}"]), float(v[f"load_v2_{s}"])
        t.append(f"| {env['onnxruntime']} | {label} | {a:.0f} | {b:.0f} | {b - a:+.0f} ({pct(b, a)}) "
                 f"| {v['ms_v2_' + s]} | yes |")
    v["versions_table"] = "\n".join(["| ONNX Runtime | Session options | Original (MiB) | Rebuilt (MiB) | Change "
                                     "| ms/step, rebuilt | Identical tokens |",
                                     "|---|---|---:|---:|---:|---:|---|"] + t)

    # --- FLORES-200 and the browser
    fl = read("flores.csv")
    if fl:
        ident = sum(r["identical_sentences"] == r["sentences"] for r in fl)
        same_bleu = sum(r["bleu_original"] == r["bleu_rebuilt"] and r["chrf_original"] == r["chrf_rebuilt"] for r in fl)
        v["flores_summary"] = (f"{len({r['pair'] for r in fl})} pairs × {len({r['setting'] for r in fl})} settings × "
                               f"{fl[0]['sentences']} sentences: identical output in {ident} of {len(fl)} runs, "
                               f"identical BLEU and chrF in {same_bleu} of {len(fl)}")
        t = ["| Pair | BLEU | chrF | Sentences identical (fast / default) |", "|---|---:|---:|---|"]
        by = {(r["pair"], r["setting"]): r for r in fl}
        for p in pairs:
            q, d = by[(p, "qdq_off")], by[(p, "default")]
            def one(k):
                a, b = q[k + "_original"], q[k + "_rebuilt"]
                return a if a == b else f"{a} / {b}"
            bleu, chrf = one("bleu"), one("chrf")
            t.append(f"| {p} | {bleu} | {chrf} | {q['identical_sentences']} / {d['identical_sentences']} "
                     f"of {q['sentences']} |")
        v["flores_table"] = "\n".join(t)
    else:
        v["flores_summary"], v["flores_table"] = "not measured yet", ""
    br = read("browser.csv")
    if br:
        def bmed(s, k):
            return med(float(r[k]) for r in br if r["setting"] == s)
        q = ("qdq_off", "default")
        qa, qb = bmed("qdq_off", "loaded_mib_original"), bmed("qdq_off", "loaded_mib_rebuilt")
        da, db = bmed("default", "loaded_mib_original"), bmed("default", "loaded_mib_rebuilt")
        b_ident = sum(r["identical"] == "True" for r in br)
        v["browser_summary"] = (
            f"fast mode {qa:.0f} → {qb:.0f} MiB after load ({pct(qb, qa)}); defaults {da:.0f} → {db:.0f} MiB "
            f"({'no saving' if db >= da else pct(db, da)} after load); fast mode "
            f"{bmed('default', 'ms_per_step_original') / bmed('qdq_off', 'ms_per_step_rebuilt'):.1f}× faster; "
            f"{'identical tokens' if b_ident == len(br) else f'identical tokens in {b_ident} of {len(br)} runs'}")
        t = ["| Pair | Settings | After load (MiB) | After the batch (MiB) | ms/step | Identical |",
             "|---|---|---:|---:|---:|---|"]
        for r in br:
            t.append(f"| {r['pair']} | {SETTINGS[r['setting']]} | "
                     f"{arrow(float(r['loaded_mib_original']), float(r['loaded_mib_rebuilt']))} | "
                     f"{arrow(float(r['after_batch_mib_original']), float(r['after_batch_mib_rebuilt']))} | "
                     f"{arrow(float(r['ms_per_step_original']), float(r['ms_per_step_rebuilt']), 1)} | "
                     f"{'yes' if r['identical'] == 'True' else 'no'} |")
        v["browser_table"] = "\n".join(t)
        v["chromium"] = br[0]["chromium"]
    else:
        v["browser_summary"], v["browser_table"], v["chromium"] = "not measured yet", "", "?"

    # --- languages: one row per source language
    by_src = {}
    for e in src:
        by_src.setdefault(e["source"], []).append(e)
    t = ["| From | To |", "|---|---|"]
    only_en = [x for x in by_src if [e["target"] for e in by_src[x]] == ["en"]]
    for lang in sorted((x for x in by_src if x not in only_en), key=lambda x: -len(by_src[x])):
        t.append(f"| {lang} | " + " · ".join(f"[{e['target']}](https://huggingface.co/{repo_id(e['pair'])})"
                                             for e in sorted(by_src[lang], key=lambda e: e["target"])) + " |")
    if only_en:
        t.append("| " + " · ".join(f"[{x}](https://huggingface.co/{repo_id(x + '-en')})" for x in only_en) + " | en |")
    v["language_table"] = "\n".join(t)

    def badge(label, value, color):
        q = lambda x: x.replace("-", "--").replace("_", "__").replace(" ", "%20")  # noqa: E731
        return f"![{label}: {value}](https://img.shields.io/badge/{q(label)}-{q(value)}-{color})"
    v["badges"] = " ".join([badge("models", "CC BY 4.0", "blue"), badge("code", "MIT", "green"),
                            badge("pairs", str(len(pairs)), "orange"), badge("ONNX Runtime", v["ort"], "informational"),
                            badge("FLORES-200", "identical output" if fl and ident == len(fl) else "see report",
                                  "brightgreen")])
    v["used_in"] = used_in()
    v["collection_url"] = collection_url()
    v["collection_label"] = f"the {CFG['name']} collection" if CFG.get("hf_collection") else \
        collection_url().replace("https://", "")
    v.update(headline())

    v.update(name=CFG["name"], long_name=CFG["long_name"], tagline=tagline(),
             results_url=results_url(), results_name=results_url().split("spaces/")[1], github=CFG["github_repo"],
             namespace=CFG["hf_namespace"], example=repo_id("en-ar"), example_de=repo_id("de-en"),
             models_list=" · ".join(f"[{p}](https://huggingface.co/{repo_id(p)})" for p in pairs))
    for template, name in (("README.md", "README.md"), ("report.md.tmpl", "REPORT.md")):
        (ROOT / name).write_text(Template((ROOT / "templates" / template).read_text(encoding="utf-8")).substitute(v))
        print("wrote", name)


if __name__ == "__main__":
    main()
