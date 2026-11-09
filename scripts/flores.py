"""Translation quality on FLORES-200 devtest (1,012 sentences per language),
original vs rebuilt decoder, for every pair and both session settings.

    python scripts/flores.py              # all pairs
    python scripts/flores.py en-ar de-en

Greedy decoding, batches of 16 sentences sorted by length. For each pair it
checks that every output token id is identical and scores both outputs with
sacrebleu (BLEU and chrF). Result: results/flores.csv.

FLORES-200: NLLB Team et al., CC BY-SA 4.0 (downloaded once, sha256 pinned).
"""

import csv
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor

from common import DEC, ENC, RESULTS, WORK, Tok, download, greedy, session, source, sources, v1_dir, v2_decoder

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SHA = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
CODES = {"ar": "arb_Arab", "de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn",
         "hi": "hin_Deva", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
         "nl": "nld_Latn", "pl": "pol_Latn", "ru": "rus_Cyrl", "th": "tha_Thai", "uk": "ukr_Cyrl",
         "vi": "vie_Latn", "zh": "zho_Hans"}
BATCH = 16


def devtest(lang: str) -> list[str]:
    tar = download(URL, WORK / "flores" / "flores200_dataset.tar.gz", sha=SHA)
    with tarfile.open(tar) as t:
        f = t.extractfile(f"./flores200_dataset/devtest/{CODES[lang]}.devtest")
        return f.read().decode("utf-8").splitlines()


def translate(enc, dec, tok, lines, pad):
    order = sorted(range(len(lines)), key=lambda i: len(lines[i]))
    out = [None] * len(lines)
    for k in range(0, len(order), BATCH):
        idx = order[k:k + BATCH]
        rows = [tok.encode(lines[i]) for i in idx]
        for i, ids in zip(idx, greedy(enc, dec, rows, pad, max_steps=2 * max(map(len, rows)) + 10)):
            out[i] = ids
    return out


def run(pair: str) -> list[dict]:
    import sacrebleu
    e = source(pair)
    d = v1_dir(pair)
    tok = Tok(d, e["lang_token"])
    src, ref = devtest(e["source"]), devtest(e["target"])
    tokenize = "zh" if e["target"] == "zh" else "13a"
    rows = []
    for setting, qdq_off in (("qdq_off", True), ("default", False)):
        enc = session(d / "onnx" / ENC, qdq_off)
        old = translate(enc, session(d / "onnx" / DEC, qdq_off), tok, src, e["pad"])
        new = translate(enc, session(v2_decoder(pair), qdq_off), tok, src, e["pad"])
        hyp = {k: [tok.decode(x) for x in ids] for k, ids in (("old", old), ("new", new))}
        score = {k: (sacrebleu.corpus_bleu(h, [ref], tokenize=tokenize).score,
                     sacrebleu.corpus_chrf(h, [ref]).score) for k, h in hyp.items()}
        rows.append({"pair": pair, "setting": setting, "sentences": len(src),
                     "identical_sentences": sum(a == b for a, b in zip(old, new)),
                     "tokens": sum(map(len, old)),
                     "bleu_original": round(score["old"][0], 2), "bleu_rebuilt": round(score["new"][0], 2),
                     "chrf_original": round(score["old"][1], 2), "chrf_rebuilt": round(score["new"][1], 2)})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    pairs = sys.argv[1:] or [e["pair"] for e in sources()]
    devtest("en")  # download once before the workers start
    with ProcessPoolExecutor(5) as pool:
        rows = [r for rs in pool.map(run, pairs) for r in rs]
    with (RESULTS / "flores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    same = sum(r["identical_sentences"] == r["sentences"] for r in rows)
    print(f"{same} of {len(rows)} pair/setting runs identical on every sentence")


if __name__ == "__main__":
    main()
