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

from common import CFG, CHARTS, DEC, HF, OSES, REFERENCE, RESULTS, ROOT, repo_id, sentences, sources, tagline  # noqa: E402

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


def save(fig, name: str, folder=None) -> None:
    folder = folder or CHARTS
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", metadata={"Creator": None, "Date": None, "Format": None,
                                                                "Type": None})
    plt.close(fig)
    svg = folder / name
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


def pair_bars(ax, groups, a_vals, b_vals, a_label, b_label, unit, digits=0):
    """Horizontal bars, two per group (orange = original/defaults, blue = rebuilt/fast), value at the bar end."""
    h = 0.34
    for i, (g, a, b) in enumerate(zip(groups, a_vals, b_vals)):
        y = -i
        ax.barh(y + h / 2 + 0.02, a, h, color=ORIGINAL, label=a_label if i == 0 else None, zorder=2)
        ax.barh(y - h / 2 - 0.02, b, h, color=OURS, label=b_label if i == 0 else None, zorder=2)
        for v, yy in ((a, y + h / 2 + 0.02), (b, y - h / 2 - 0.02)):
            ax.text(v, yy, f"  {v:.{digits}f} {unit}", va="center", fontsize=11, color=MUTED)
    ax.set_yticks([-i for i in range(len(groups))], groups)
    ax.set_xlim(0, max(a_vals + b_vals) * 1.3)
    ax.xaxis.set_visible(False)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=11, handlelength=1, handletextpad=0.4)


def medians(rows, field, config):
    pairs = sorted({r["pair"] for r in rows})
    return statistics.median(float(next(r[field] for r in rows if r["pair"] == p and r["config"] == config))
                             for p in pairs)


def speed(refs) -> None:
    oses = list(refs)
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.9 * len(oses)))
    pair_bars(ax, oses, [medians(refs[o], "ms_per_step", "v1-default") for o in oses],
              [medians(refs[o], "ms_per_step", "v2-qdq_off") for o in oses],
              "ONNX Runtime defaults", "fast mode (disable_quant_qdq = 1)", "ms", 1)
    save(fig, "speed.svg")


def fast_memory(mems) -> None:
    oses = list(mems)
    val = lambda o, c: statistics.median(float(r["load_mib"]) for r in mems[o] if r["config"] == c)  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.9 * len(oses)))
    pair_bars(ax, oses, [val(o, "v1-qdq_off") for o in oses], [val(o, "v2-qdq_off") for o in oses],
              "original", CFG["name"], "MiB")
    save(fig, "memory_fast.svg")


def flores_chart() -> None:
    rows = [r for r in read("flores.csv") if r["setting"] == "qdq_off"]
    a = [float(r["bleu_original"]) for r in rows]
    b = [float(r["bleu_rebuilt"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    top = max(a + b) * 1.08
    ax.plot([0, top], [0, top], color=GRID, lw=2, zorder=1)
    ax.scatter(a, b, s=46, color=OURS, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.grid(axis="both")
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", left=True, colors=MUTED)
    ax.set_xlabel("BLEU, original")
    ax.set_ylabel(f"BLEU, {CFG['name']}")
    same = sum(x == y for x, y in zip(a, b))
    ax.text(top * 0.04, top * 0.92, f"{same} of {len(rows)} pairs on the line", fontsize=11, color=INK)
    save(fig, "flores.svg")


def browser_chart() -> None:
    rows = read("browser.csv")
    med = lambda s, k: statistics.median(float(r[k]) for r in rows if r["setting"] == s)  # noqa: E731
    groups = [("fast mode", "qdq_off"), ("defaults", "default")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    pair_bars(ax, [g for g, _ in groups], [med(s, "loaded_mib_original") for _, s in groups],
              [med(s, "loaded_mib_rebuilt") for _, s in groups], "original", CFG["name"], "MiB")
    save(fig, "browser.svg")


def tradeoff(refs, vertical: bool = False, folder=None) -> None:
    """Memory after load against decoder ms per step: four configurations per OS (medians, reference pairs)."""
    oses = list(refs)
    if vertical:
        fig, axes = plt.subplots(len(oses), 1, figsize=(4.6, 3.3 * len(oses)), squeeze=False)
        axes = axes.T
    else:
        fig, axes = plt.subplots(1, len(oses), figsize=(3.9 * len(oses), 3.6), squeeze=False)
    for ax, o in zip(axes[0], oses):
        pt = {c: (medians(refs[o], "ms_per_step", c), medians(refs[o], "load_mib", c))
              for c in ("v1-default", "v1-qdq_off", "v2-default", "v2-qdq_off")}
        ax.annotate("", xy=pt["v2-qdq_off"], xytext=pt["v1-qdq_off"],
                    arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.5, "shrinkA": 8, "shrinkB": 8})
        for c, (x, y) in pt.items():
            ax.scatter([x], [y], s=110, color=ORIGINAL if c.startswith("v1") else OURS, edgecolor=SURFACE,
                       linewidth=2, zorder=3, marker="o" if c.endswith("default") else "s")
            above = c.startswith("v1")
            ax.text(x, y + (22 if above else -42), "defaults" if c.endswith("default") else "fast", ha="center",
                    fontsize=10, color=MUTED)
        ax.set_title(o, loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.set_xlim(0, max(x for x, _ in pt.values()) * 1.2)
        ax.set_ylim(0, max(y for _, y in pt.values()) * 1.2)
        ax.grid(axis="both")
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", left=True, colors=MUTED)
        ax.set_xlabel("Decoder ms per step (lower is faster)")
    for ax in axes[0] if vertical else axes[0][:1]:
        ax.set_ylabel("Memory after load (MiB)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=ORIGINAL, markersize=9, label="original"),
               Line2D([], [], marker="o", ls="", color=OURS, markersize=9, label=CFG["name"])]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=10, handletextpad=0.2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if vertical else 0.92))
    save(fig, "tradeoff-portrait.svg" if vertical else "tradeoff.svg", folder)


def breakdown(refs, pair: str = "en-ar") -> None:
    """Fast-mode memory of one pair, original vs rebuilt: the float copy of the output layer is computed from its
    tensor shape (results/graph.json); the totals are measured; the rest is the same files in both."""
    g = json.loads((RESULTS / "graph.json").read_text(encoding="utf-8"))
    copy = next(t["mib"] for t in g["original"]["after_ort_qdq_off"] if t["dtype"] == "float")
    oses = list(refs)
    fig, axes = plt.subplots(len(oses), 1, figsize=(7.2, 1.2 + 1.3 * len(oses)), sharex=True, squeeze=False)
    for ax, o in zip(axes[:, 0], oses):
        tot = {c: float(next(r["load_mib"] for r in refs[o] if r["pair"] == pair and r["config"] == c))
               for c in ("v1-qdq_off", "v2-qdq_off")}
        rest = tot["v2-qdq_off"] - copy
        rows = [("original", [rest, copy, tot["v1-qdq_off"] - rest - copy]), (CFG["name"], [rest, copy, 0])]
        colors = [GRID, OURS, ORIGINAL]
        labels = ["encoder, decoder weights, the rest", "float copy of the output layer",
                  "second copy (as measured)"]
        for i, (name, parts) in enumerate(rows):
            left = 0
            for j, w in enumerate(parts):
                if w <= 0:
                    continue
                ax.barh(-i, w, 0.55, left=left, color=colors[j], edgecolor=SURFACE, linewidth=2, zorder=2,
                        label=labels[j] if (o == oses[0] and name == "original") else None)
                left += w
            ax.text(left, -i, f"  {left:.0f} MiB", va="center", fontsize=11, color=MUTED)
        ax.set_yticks([0, -1], [r[0] for r in rows])
        ax.set_title(f"{o} · {pair} · fast mode", loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.xaxis.set_visible(False)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
    axes[-1][0].set_xlim(0, max(float(r["load_mib"]) for o in oses for r in refs[o]) * 1.18)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.25), ncol=3, fontsize=9.5, handlelength=1)
    fig.tight_layout()
    save(fig, "breakdown.svg")


def how_it_works(refs, pair: str = "en-ar") -> None:
    """The explanation diagram, top to bottom, with the measured numbers filled in (assets/charts/how-it-works.svg)."""
    g = json.loads((RESULTS / "graph.json").read_text(encoding="utf-8"))
    cp = next(t for t in g["original"]["after_ort_qdq_off"] if t["dtype"] == "float")
    shape = " × ".join(f"{d:,}" for d in reversed(cp["shape"]))
    diff = sorted({e["v1"][DEC]["size"] - e["v2_decoder"]["size"] for e in sources()})
    lin = {c: float(next(r["load_mib"] for r in refs["Linux"] if r["pair"] == pair and r["config"] == c))
           for c in ("v1-qdq_off", "v2-qdq_off")}
    ms = {o: (medians(refs[o], "ms_per_step", "v1-default"), medians(refs[o], "ms_per_step", "v2-qdq_off"))
          for o in refs}
    speed_line = " · ".join(f"{o} {a:.1f} ms" for o, (a, b) in ms.items())
    fast_line = " · ".join(f"{o} {b:.1f} ms" for o, (a, b) in ms.items())
    extra = lin["v1-qdq_off"] - lin["v2-qdq_off"]
    W, ink, muted, grey, orange, blue = 960, INK, MUTED, "#c9c8c3", ORIGINAL, OURS
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 960" width="{W}" height="960" '
           'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="15">',
           f'<rect width="{W}" height="960" rx="14" fill="{SURFACE}"/>',
           '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
           f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']

    def text(x, y, t, size=15, color=ink, anchor="middle", weight="400"):
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
                   f'font-weight="{weight}">{t}</text>')

    def box(x, y, w, h, fill="#ffffff", stroke=grey, sw=1.5, dash=""):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" '
                   f'stroke-width="{sw}"{dash}/>')

    def node(x, y, label, kind="plain", w=170, h=36):
        fill, stroke = {"plain": ("#ffffff", grey), "dup": ("#fdeee7", orange), "one": ("#e6f0fb", blue)}[kind]
        box(x - w / 2, y, w, h, fill, stroke, 2 if kind != "plain" else 1.5)
        text(x, y + 23, label)

    def arrow(x1, y1, x2, y2):
        out.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{MUTED}" stroke-width="1.6" fill="none" '
                   'marker-end="url(#a)"/>')

    def diamond(x, y):
        out.append(f'<path d="M{x} {y} L{x + 34} {y + 20} L{x} {y + 40} L{x - 34} {y + 20} Z" fill="#ffffff" '
                   f'stroke="{grey}" stroke-width="1.5"/>')
        text(x, y + 25, "If", weight="600")

    def banner(y, num, title, who, sub):
        color = blue if who == "our change" else MUTED
        out.append(f'<rect x="24" y="{y}" width="{W - 48}" height="34" rx="8" fill="#f1f0ec"/>')
        out.append(f'<rect x="24" y="{y}" width="6" height="34" rx="3" fill="{color}"/>')
        text(44, y + 23, f"{num}  {title}", 17, ink, "start", "700")
        text(W - 40, y + 23, who, 13, color, "end", "700")
        if sub:
            text(44, y + 56, sub, 14, muted, "start")

    # key
    text(24, 32, "How it works", 20, ink, "start", "700")
    out.append(f'<rect x="{W - 360}" y="18" width="6" height="18" rx="3" fill="{blue}"/>')
    text(W - 346, 32, "our change (in the file)", 13, muted, "start")
    out.append(f'<rect x="{W - 180}" y="18" width="6" height="18" rx="3" fill="{MUTED}"/>')
    text(W - 166, 32, "ONNX Runtime", 13, muted, "start")

    # 1. the file
    banner(52, "1", "The file", "our change", "")
    for cx, title in ((240, "Old"), (720, "New")):
        text(cx, 112, title, 16, ink, "middle", "700")
        node(cx, 124, "model.shared.weight (8-bit)", w=240)
    arrow(240, 160, 240, 176)
    diamond(240, 178)
    for bx, name in ((125, "branch 1: no_past"), (355, "branch 2: with_past")):
        arrow(240 + (bx - 240) * 0.15, 208, bx, 236)
        box(bx - 100, 238, 200, 210, "none", grey, 1.5, ' stroke-dasharray="5 4"')
        text(bx, 260, name, 14, ink, "middle", "600")
        node(bx, 272, "Transpose", "dup")
        arrow(bx, 308, bx, 322)
        node(bx, 324, "DequantizeLinear", "dup")
        arrow(bx, 360, bx, 374)
        node(bx, 376, "MatMul → logits")
    text(240, 470, "the same two steps in both branches", 13, orange, "middle")
    arrow(720, 160, 720, 174)
    node(720, 176, "Transpose", "one")
    arrow(720, 212, 720, 226)
    node(720, 228, "DequantizeLinear", "one")
    arrow(720, 264, 720, 280)
    diamond(720, 282)
    for bx, name in ((605, "branch 1: no_past"), (835, "branch 2: with_past")):
        arrow(720 + (bx - 720) * 0.15, 312, bx, 340)
        box(bx - 100, 342, 200, 106, "none", grey, 1.5, ' stroke-dasharray="5 4"')
        text(bx, 364, name, 14, ink, "middle", "600")
        node(bx, 376, "MatMul → logits")
    text(720, 470, "the two steps once, before the If", 13, blue, "middle")
    text(W / 2, 498, f"Same weights, same tokens. The file is {' to '.join(map(str, diff))} bytes smaller.", 14,
         muted)

    # 2. loading
    banner(516, "2", "Loading in fast mode", "ONNX Runtime",
           "session.disable_quant_qdq = 1 folds DequantizeLinear into a float table when the model loads")
    for i, x in enumerate((140, 340)):
        box(x - 90, 588, 180, 52, "#fdeee7", orange, 2)
        text(x, 609, f"float table, branch {i + 1}", 14, ink, "middle", "600")
        text(x, 629, f"{shape} · {cp['mib']} MiB", 12, muted)
    box(630, 588, 180, 52, "#e6f0fb", blue, 2)
    text(720, 609, "float table, shared", 14, ink, "middle", "600")
    text(720, 629, f"{shape} · {cp['mib']} MiB", 12, muted)
    text(240, 668, f"{pair}, Linux: {lin['v1-qdq_off']:.0f} MiB after load", 15, ink, "middle", "700")
    text(720, 668, f"{pair}, Linux: {lin['v2-qdq_off']:.0f} MiB after load", 15, ink, "middle", "700")
    text(240, 688, f"the second copy costs about {extra:.0f} MiB as measured (ONNX Runtime 1.31)", 12, muted)

    # 3. translating
    banner(712, "3", "Translating", "ONNX Runtime", "")
    box(40, 762, 420, 86, "#ffffff", grey)
    text(250, 786, "Default setting: 8-bit path", 15, ink, "middle", "600")
    text(250, 806, "dequantizes every step: less memory, slower", 13, muted)
    text(250, 830, f"{speed_line} per step", 14, ink)
    box(500, 762, 420, 86, "#e6f0fb", blue, 2)
    text(710, 786, "Fast mode: MatMul on the ready float table", 15, ink, "middle", "600")
    text(710, 806, f"with {CFG['name']}: one table instead of two", 13, muted)
    text(710, 830, f"{fast_line} per step", 14, ink)
    text(W / 2, 880, f"The speed is ONNX Runtime's fast mode; {CFG['name']} makes it affordable in memory.", 15,
         muted)
    text(W / 2, 925, "One change in the file; ONNX Runtime's fast mode does the rest, now without the double memory.",
         16, ink, "middle", "700")
    out.append("</svg>")
    CHARTS.mkdir(parents=True, exist_ok=True)
    (CHARTS / "how-it-works.svg").write_text("\n".join(out) + "\n", encoding="utf-8")


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
    speed(refs)
    tradeoff(refs)
    breakdown(refs)
    how_it_works(refs)
    fast_memory(mems)
    if (RESULTS / "flores.csv").exists():
        flores_chart()
    if (RESULTS / "browser.csv").exists():
        browser_chart()
    page(refs, mems)
    print("charts in assets/charts, results page in", (HF / "_space").name)


if __name__ == "__main__":
    main()
