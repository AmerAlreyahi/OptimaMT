# OptimaMT

**Optima Machine Translation: same translations, 14–51 % less memory.**

49 OPUS-MT translation models for ONNX Runtime and transformers.js. Each is a drop-in copy of the
widely used int8 ONNX export with one file rebuilt: the merged decoder now dequantizes its output
projection once instead of once per branch. Same weights, same tokens.

![Memory after load, original and OptimaMT (Linux)](assets/charts/memory.svg)

| Memory after load (median, 49 pairs) | Linux | Windows |
|---|---:|---:|
| `session.disable_quant_qdq = 1` | 525 → 281 MiB (−46 %) | 500 → 257 MiB (−49 %) |
| ONNX Runtime defaults | 277 → 217 MiB (−22 %) | 196 → 166 MiB (−16 %) |
| Identical tokens | 98 / 98 runs | yes (reference pairs) |

**Interactive results:** [OptimaMT/results](https://huggingface.co/spaces/OptimaMT/results) · **Method and all numbers:** [REPORT.md](REPORT.md)

## Quick start

transformers.js:

```js
import { pipeline } from "@huggingface/transformers";

const translator = await pipeline("translation", "OptimaMT/de-en", {
  dtype: "q8",
  session_options: { extra: { session: { disable_quant_qdq: "1" } } }, // faster decoding, see REPORT.md
});
const [out] = await translator("Der Leuchtturm steht an der Küste.");
console.log(out.translation_text); // The lighthouse is on the coast.
```

Python (onnxruntime, sentencepiece, numpy):

```bash
pip install onnxruntime sentencepiece numpy huggingface_hub
python scripts/translate.py OptimaMT/de-en "Der Leuchtturm steht an der Küste."
```

## Models

`OptimaMT/<source>-<target>`: [en-ar](https://huggingface.co/OptimaMT/en-ar) · [en-fr](https://huggingface.co/OptimaMT/en-fr) · [en-es](https://huggingface.co/OptimaMT/en-es) · [en-de](https://huggingface.co/OptimaMT/en-de) · [en-ru](https://huggingface.co/OptimaMT/en-ru) · [en-zh](https://huggingface.co/OptimaMT/en-zh) · [ar-en](https://huggingface.co/OptimaMT/ar-en) · [fr-en](https://huggingface.co/OptimaMT/fr-en) · [zh-en](https://huggingface.co/OptimaMT/zh-en) · [ja-en](https://huggingface.co/OptimaMT/ja-en) · [es-en](https://huggingface.co/OptimaMT/es-en) · [de-en](https://huggingface.co/OptimaMT/de-en) · [ru-en](https://huggingface.co/OptimaMT/ru-en) · [it-en](https://huggingface.co/OptimaMT/it-en) · [nl-en](https://huggingface.co/OptimaMT/nl-en) · [uk-en](https://huggingface.co/OptimaMT/uk-en) · [pl-en](https://huggingface.co/OptimaMT/pl-en) · [hi-en](https://huggingface.co/OptimaMT/hi-en) · [id-en](https://huggingface.co/OptimaMT/id-en) · [vi-en](https://huggingface.co/OptimaMT/vi-en) · [th-en](https://huggingface.co/OptimaMT/th-en) · [ko-en](https://huggingface.co/OptimaMT/ko-en) · [en-it](https://huggingface.co/OptimaMT/en-it) · [en-nl](https://huggingface.co/OptimaMT/en-nl) · [en-uk](https://huggingface.co/OptimaMT/en-uk) · [en-hi](https://huggingface.co/OptimaMT/en-hi) · [en-id](https://huggingface.co/OptimaMT/en-id) · [en-vi](https://huggingface.co/OptimaMT/en-vi) · [de-es](https://huggingface.co/OptimaMT/de-es) · [de-fr](https://huggingface.co/OptimaMT/de-fr) · [es-de](https://huggingface.co/OptimaMT/es-de) · [es-fr](https://huggingface.co/OptimaMT/es-fr) · [es-it](https://huggingface.co/OptimaMT/es-it) · [es-ru](https://huggingface.co/OptimaMT/es-ru) · [fr-de](https://huggingface.co/OptimaMT/fr-de) · [fr-es](https://huggingface.co/OptimaMT/fr-es) · [fr-ru](https://huggingface.co/OptimaMT/fr-ru) · [it-es](https://huggingface.co/OptimaMT/it-es) · [it-fr](https://huggingface.co/OptimaMT/it-fr) · [ru-es](https://huggingface.co/OptimaMT/ru-es) · [ru-fr](https://huggingface.co/OptimaMT/ru-fr) · [ru-uk](https://huggingface.co/OptimaMT/ru-uk) · [uk-ru](https://huggingface.co/OptimaMT/uk-ru) · [nl-fr](https://huggingface.co/OptimaMT/nl-fr) · [fr-ar](https://huggingface.co/OptimaMT/fr-ar) · [es-ar](https://huggingface.co/OptimaMT/es-ar) · [ar-fr](https://huggingface.co/OptimaMT/ar-fr) · [ar-es](https://huggingface.co/OptimaMT/ar-es) · [ar-de](https://huggingface.co/OptimaMT/ar-de).

en-ar, en-zh, fr-ar and es-ar take a target token first (e.g. `>>ara<<`); use transformers.js 3.x for those.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/convert.py && python scripts/verify.py
python scripts/measure.py reference && python scripts/measure.py all
python scripts/inspect_graph.py && python scripts/build_hf_repos.py
python scripts/charts.py && python scripts/render_docs.py
```

Walkthrough: [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb). Used in
[OpenCircle](https://github.com/AmerAlreyahi/OpenCircle), a desktop screen translator.

## Licence

Models: OPUS-MT (University of Helsinki), ONNX exports by Xenova, modified; CC BY 4.0 ([NOTICE](NOTICE)).
Code: MIT ([LICENSE](LICENSE)).
