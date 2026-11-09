# $name

**$long_name: $tagline**

$pairs OPUS-MT translation models for ONNX Runtime and transformers.js. Each is a drop-in copy of the
widely used int8 ONNX export with one file rebuilt: the merged decoder now dequantizes its output
projection once instead of once per branch. Same weights, same tokens.

![Memory after load, original and $name (Linux)](assets/charts/memory.svg)

| Memory after load (median, $pairs pairs) | Linux | Windows |
|---|---:|---:|
| `session.disable_quant_qdq = 1` | $qdq_off_v1_med → $qdq_off_v2_med MiB (−$qdq_off_pct_med %) | $win_qdq_off_v1_med → $win_qdq_off_v2_med MiB (−$win_qdq_off_pct_med %) |
| ONNX Runtime defaults | $default_v1_med → $default_v2_med MiB (−$default_pct_med %) | $win_default_v1_med → $win_default_v2_med MiB (−$win_default_pct_med %) |
| Identical tokens | $identical_runs / $all_runs runs | $win_ref_identical (reference pairs) |

**Interactive results:** [$results_name]($results_url) · **Method and all numbers:** [REPORT.md](REPORT.md)

## Quick start

transformers.js:

```js
import { pipeline } from "@huggingface/transformers";

const translator = await pipeline("translation", "$example_de", {
  dtype: "q8",
  session_options: { extra: { session: { disable_quant_qdq: "1" } } }, // faster decoding, see REPORT.md
});
const [out] = await translator("Der Leuchtturm steht an der Küste.");
console.log(out.translation_text); // The lighthouse is on the coast.
```

Python (onnxruntime, sentencepiece, numpy):

```bash
pip install onnxruntime sentencepiece numpy huggingface_hub
python scripts/translate.py $example_de "Der Leuchtturm steht an der Küste."
```

## Models

`$namespace/<source>-<target>`: $models_list.

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
