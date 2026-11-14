"""Translate with one of the model folders using only onnxruntime,
sentencepiece and numpy (greedy decoding, KV cache).

    python scripts/translate.py PATH_OR_REPO_ID "Text to translate" ["more text" ...]

A folder from Hugging Face can be fetched first with
`huggingface_hub.snapshot_download(repo_id)`. Pairs whose target has several
variants (en-ar, en-zh, en-pt, fr-ar, es-ar) need a target token: pass it with
--lang ">>ara<<" (see the model card).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm


class Tokenizer:
    """Marian tokenization: source.spm pieces mapped to vocab.json ids, an
    optional target-language token first, then </s>."""

    def __init__(self, folder: Path, lang_token: str | None = None):
        self.sp = spm.SentencePieceProcessor(model_file=str(folder / "source.spm"))
        self.vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        self.inv = {v: k for k, v in self.vocab.items()}
        self.unk = self.vocab["<unk>"]
        self.pad = self.vocab["<pad>"]
        self.token = lang_token

    def encode(self, text: str) -> list[int]:
        ids = [self.vocab.get(x, self.unk) for x in self.sp.encode(text, out_type=str)]
        return ([self.vocab[self.token]] if self.token else []) + ids + [0]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] for i in ids if i != 0).replace("▁", " ").strip()


def greedy(enc, dec, rows: list[list[int]], pad: int, max_steps: int = 256, times: list | None = None):
    """Greedy decoding of one padded batch with the merged decoder. The pad id
    is never chosen (as in the models' generation config). `times`, when
    given, receives the seconds of every decoder step."""
    width = max(map(len, rows))
    ids = np.full((len(rows), width), pad, np.int64)
    mask = np.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = r
        mask[i, :len(r)] = 1
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    past = {i.name: np.zeros((len(rows), i.shape[1], 0, i.shape[3]), np.float32)
            for i in dec.get_inputs() if i.name.startswith("past_key_values")}
    outs = [o.name for o in dec.get_outputs()]
    cur = np.full((len(rows), 1), pad, np.int64)
    got = [[] for _ in rows]
    done = [False] * len(rows)
    for step in range(max_steps):
        feed = {"encoder_attention_mask": mask, "input_ids": cur,
                "encoder_hidden_states": hidden, "use_cache_branch": np.array([step > 0])}
        feed.update(past)
        start = time.perf_counter()
        res = dec.run(None, feed)
        if times is not None:
            times.append(time.perf_counter() - start)
        logits = res[0][:, -1, :]
        logits[:, pad] = -np.inf
        nxt = logits.argmax(-1)
        for i, t in enumerate(nxt):
            if not done[i]:
                got[i].append(int(t))
                done[i] = t == 0
        if all(done):
            break
        for name, value in zip(outs[1:], res[1:]):
            key = name.replace("present", "past_key_values")
            if step == 0 or ".decoder." in key:  # the encoder keys/values are computed once
                past[key] = value
        cur = nxt[:, None].astype(np.int64)
    return got


def translate(folder, texts: list[str], lang_token: str | None = None) -> list[str]:
    folder = Path(folder)
    tok = Tokenizer(folder, lang_token)
    enc = ort.InferenceSession(str(folder / "onnx" / "encoder_model_quantized.onnx"))
    dec = ort.InferenceSession(str(folder / "onnx" / "decoder_model_merged_quantized.onnx"))
    return [tok.decode(ids) for ids in greedy(enc, dec, [tok.encode(t) for t in texts], tok.pad)]


if __name__ == "__main__":
    args = sys.argv[1:]
    lang = None
    if "--lang" in args:
        i = args.index("--lang")
        lang = args[i + 1]
        del args[i:i + 2]
    path = Path(args[0])
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(args[0]))
    for line in translate(path, args[1:], lang):
        print(line)
