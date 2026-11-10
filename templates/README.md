# $name

$badges

**$name is a set of free translation models that run offline**, on your own computer or in the browser: 49 translation
directions between 17 languages, including English, Arabic, Chinese, French, German, Japanese, Russian and Spanish.
They are the OPUS-MT models from the University of Helsinki with one small fix inside the model file. The translations
stay exactly the same, and in ONNX Runtime's fast mode they need about half the memory. Same files, same code: swap
the model name and you're done.

**$tagline**

![Memory against speed](assets/charts/tradeoff.svg)

<sub>Fast mode used to cost double memory; $name keeps it fast and light. Medians of 5 reference pairs.</sub>

![How it works](assets/charts/how-it-works.svg)

Same translations: on FLORES-200 devtest all $pairs pairs give identical output, so identical BLEU.

![BLEU, original vs $name](assets/charts/flores.svg)

More charts, the browser results and every number: [REPORT.md](REPORT.md) · interactive results and a live demo:
[$results_name]($results_url)

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

Models: OPUS-MT (University of Helsinki), ONNX exports by Xenova, modified; CC BY 4.0 ([NOTICE](NOTICE.md)).
Code: MIT ([LICENSE](LICENSE.md)).
