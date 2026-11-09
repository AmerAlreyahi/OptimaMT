---
license: cc-by-4.0
language:
- $src
- $tgt
pipeline_tag: translation
library_name: transformers.js
base_model: $base
tags:
- translation
- marian
- opus-mt
- onnx
---

# $name $src → $tgt ($src_name to $tgt_name)

Drop-in copy of $origin with a lighter merged decoder: same weights, same tokens, less memory.
Part of [$name](https://github.com/$github) ($count pairs, `$scheme`).

$memory_table

ONNX Runtime $ort, CPU, 4 threads (Linux; Windows in a VM). Interactive results: [$results_name]($results_url) ·
method: [REPORT.md](https://github.com/$github/blob/main/REPORT.md).

## Use

```js
import { pipeline } from "@huggingface/transformers";

const translator = await pipeline("translation", "$repo_id", {
  dtype: "q8",
  session_options: { extra: { session: { disable_quant_qdq: "1" } } },
});
const [out] = await translator("$js_text");
console.log(out.translation_text);
```

Python ([translate.py](https://github.com/$github/blob/main/scripts/translate.py): onnxruntime, sentencepiece, numpy):

```bash
python translate.py $repo_id "$py_text"$py_lang
```
$lang_note
## Files

$files_note
Vocabulary: $vocab tokens.

## Licence and credits

- Weights: [OPUS-MT]($upstream_url) by the University of Helsinki, CC BY 4.0.
- ONNX export: $origin.
- Modified: the decoder's output projection was moved out of its two branches (graph change only). CC BY 4.0.
