"""Draws the charts from the measured data in results/; nothing is typed in.

    python scripts/charts.py

- assets/charts/memory.svg     memory after load, 5 reference pairs, original vs rebuilt (Linux)
- assets/charts/saving.svg     memory saved across all pairs, both session settings, each OS
- work/hf/_space/index.html    the interactive results page (a static Hugging Face Space)
"""

import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CFG, CHARTS, HF, OSES, REFERENCE, RESULTS, ROOT, repo_id, sentences, sources, tagline  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORIGINAL, OURS = "#eb6834", "#2a78d6"
SETTINGS = {"qdq_off": "disable_quant_qdq = 1", "default": "ONNX Runtime defaults"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 1, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "svg.hashsalt": CFG["slug"], "ytick.left": False,
})


def read(name: str) -> list[dict]:
    return list(csv.DictReader((RESULTS / name).open(encoding="utf-8")))


def save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / name, bbox_inches="tight", metadata={"Creator": None, "Date": None, "Format": None,
                                                                "Type": None})
    plt.close(fig)
    svg = CHARTS / name
    svg.write_text(svg.read_text(encoding="utf-8").replace('id="matplotlib.', 'id="'))  # plain element ids


def dumbbell(ax, rows, label_first=False):
    """rows: (label, original, rebuilt) top to bottom."""
    for i, (label, a, b) in enumerate(rows):
        y = -i
        ax.plot([b, a], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=90, color=ORIGINAL, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="original" if i == 0 else None)
        ax.scatter([b], [y], s=90, color=OURS, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=CFG["name"] if i == 0 else None)
        ax.text(a + 12, y, f"{100 * (b - a) / a:+.0f} %", va="center", color=MUTED, fontsize=11)
        if label_first and i == 0:
            ax.text(a, y + 0.45, f"{a:.0f}", ha="center", color=MUTED, fontsize=10)
            ax.text(b, y + 0.45, f"{b:.0f}", ha="center", color=MUTED, fontsize=10)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows])
    ax.set_ylim(-len(rows) + 0.4, 0.9)


def memory(ref) -> None:
    val = {(r["pair"], r["config"]): float(r["load_mib"]) for r in ref}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"hspace": 0.45})
    for ax, s in zip(axes, SETTINGS):
        dumbbell(ax, [(p, val[(p, "v1-" + s)], val[(p, "v2-" + s)]) for p in REFERENCE], label_first=True)
        ax.set_title(SETTINGS[s], loc="left", fontsize=12, color=INK, fontweight="bold")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, fontsize=11, handletextpad=0.2)
    axes[1].set_xlabel("Memory after load (MiB)")
    axes[1].set_xlim(0, max(val.values()) * 1.15)
    save(fig, "memory.svg")


def saving(mems: dict) -> None:
    """mems: {os: rows of memory_all}; one row of dots per os and setting."""
    rows = [(os_, s) for os_ in mems for s in SETTINGS]
    fig, ax = plt.subplots(figsize=(7.2, 1.0 + 0.7 * len(rows)))
    for i, (os_, s) in enumerate(rows):
        val = {(r["pair"], r["config"]): float(r["load_mib"]) for r in mems[os_]}
        pairs = sorted({p for p, _ in val})
        xs = [100 * (val[(p, "v1-" + s)] - val[(p, "v2-" + s)]) / val[(p, "v1-" + s)] for p in pairs]
        ax.scatter(xs, [-i] * len(xs), s=60, color=OURS, alpha=0.55, edgecolor=SURFACE, linewidth=1, zorder=3)
        m = statistics.median(xs)
        ax.plot([m, m], [-i - 0.28, -i + 0.28], color=INK, lw=2, zorder=4)
        ax.text(m, -i + 0.32, f"median {m:.0f} %", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_yticks([-i for i in range(len(rows))], [f"{o}, {SETTINGS[s]}" for o, s in rows])
    ax.set_ylim(-len(rows) + 0.4, 0.8)
    ax.set_xlim(0, 55)
    ax.set_xlabel(f"Memory saved after load, each dot one of {len(pairs)} pairs (%)")
    save(fig, "saving.svg")


def page(refs: dict, mems: dict) -> None:
    """The interactive page: one self-contained HTML file, data inlined; one data set per operating system."""
    same = {(r["pair"], r["setting"]): r["identical"] == "yes" for r in read("identical.csv")}
    data = {"name": CFG["name"], "tagline": tagline(), "github": CFG["github_repo"],
            "namespace": CFG["hf_namespace"], "os": {},
            "demo": [{"pair": p, "repo": repo_id(p), "text": sentences(p.split("-")[0])[0]}
                     for p in ("de-en", "en-fr", "fr-en", "en-de", "es-en", "en-es")]}
    for os_, suffix in OSES.items():
        if os_ not in mems:
            continue
        env = json.loads((RESULTS / f"environment{suffix}.json").read_text(encoding="utf-8"))
        mem = {(r["pair"], r["config"]): float(r["load_mib"]) for r in mems[os_]}
        data["os"][os_] = {
            "ort": env["onnxruntime"], "cpu": env["cpu"], "threads": env["threads"], "platform": env["os"],
            "pairs": [{"pair": e["pair"], "repo": repo_id(e["pair"]),
                       **{c: mem[(e["pair"], c)] for c in ("v1-default", "v2-default", "v1-qdq_off", "v2-qdq_off")},
                       "identical": all(same[(e["pair"], s)] for s in SETTINGS)} for e in sources()],
            "reference": [{"pair": r["pair"], "config": r["config"], "load": float(r["load_mib"]),
                           "peak": float(r["batch_peak_mib"]), "ms": float(r["ms_per_step"])} for r in refs[os_]],
        }
    flores = read("flores.csv") if (RESULTS / "flores.csv").exists() else []
    data["flores"] = [{k: r[k] for k in ("pair", "setting", "sentences", "identical_sentences", "bleu_original",
                                         "bleu_rebuilt", "chrf_original", "chrf_rebuilt")} for r in flores]
    data["browser"] = read("browser.csv") if (RESULTS / "browser.csv").exists() else []
    html = (ROOT / "templates" / "results.html").read_text(encoding="utf-8")
    html = html.replace("__TITLE__", f"{CFG['name']} results").replace("__DATA__", json.dumps(data))
    out = HF / "_space"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "README.md").write_text(
        f"---\ntitle: {CFG['name']} results\ncolorFrom: blue\ncolorTo: indigo\nsdk: static\npinned: true\n---\n\n"
        f"Interactive memory and speed results of {CFG['name']}. Source and method: "
        f"https://github.com/{CFG['github_repo']}\n", encoding="utf-8")


def main() -> None:
    refs = {o: read(f"reference{x}.csv") for o, x in OSES.items() if (RESULTS / f"reference{x}.csv").exists()}
    mems = {o: read(f"memory_all{x}.csv") for o, x in OSES.items() if (RESULTS / f"memory_all{x}.csv").exists()}
    memory(refs["Linux"])
    saving(mems)
    page(refs, mems)
    print("charts in assets/charts, results page in", (HF / "_space").name)


if __name__ == "__main__":
    main()
