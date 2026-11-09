# $name

$badges

**$tagline**

ONNX Runtime decodes these OPUS-MT models $speedup× faster with `disable_quant_qdq = 1`, but the original
int8 exports pay for it with a second float copy of the output layer. $name removes that copy.

| | Linux | Windows |
|---|---:|---:|
| Decoder time per step: defaults → fast mode | $ms_v1_default → $ms_v2_qdq_off ms | $win_ms_v1_default → $win_ms_v2_qdq_off ms |
| Memory in fast mode: original → $name | $qdq_off_v1_med → $qdq_off_v2_med MiB | $win_qdq_off_v1_med → $win_qdq_off_v2_med MiB |
| Memory with defaults: original → $name | $default_v1_med → $default_v2_med MiB | $win_default_v1_med → $win_default_v2_med MiB |

Medians over $pairs pairs (memory) and 5 reference pairs (time).

- **Same translations:** FLORES-200 devtest, $flores_summary.
- **In the browser** (transformers.js, onnxruntime-web): $browser_summary.

Interactive results and a live demo:
[$results_name]($results_url) · method: [REPORT.md](REPORT.md)

![What changed in the decoder](assets/decoder-change.svg)

## Used in production

$used_in

## Quick start

```js
import { pipeline } from "@huggingface/transformers";

const translator = await pipeline("translation", "$example_de", {
  dtype: "q8",
  session_options: { extra: { session: { disable_quant_qdq: "1" } } }, // fast mode
});
const [out] = await translator("Der Leuchtturm steht an der Küste.");
console.log(out.translation_text); // The lighthouse is on the coast.
```

```bash
pip install onnxruntime sentencepiece numpy huggingface_hub
python scripts/translate.py $example_de "Der Leuchtturm steht an der Küste."
```

## Verify it yourself

```bash
pip install -r requirements.txt
python scripts/convert.py en-ar   # rebuilds the decoder from Xenova's file: byte-identical to ours
python scripts/verify.py en-ar    # same token ids as the original, both session settings
```

## Models

All $pairs repos: [$collection_label]($collection_url), named `$namespace/<source>-<target>`.

$language_table

en-ar, en-zh, fr-ar and es-ar take a target token first (`>>ara<<`, `>>cmn_Hans<<`); use transformers.js 3.x for these.

## Licence

Models: OPUS-MT (University of Helsinki), ONNX exports by Xenova, modified; CC BY 4.0 ([NOTICE](NOTICE)).
Code: MIT ([LICENSE](LICENSE)).
