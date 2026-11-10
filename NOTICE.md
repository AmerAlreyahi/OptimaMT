# Notice

Licences of this repository and of the models it describes

The code in this repository (scripts/, tests/, notebooks/) is licensed under the
MIT License (see [LICENSE](LICENSE.md)).

The model files described here and published on Hugging Face are NOT covered by
the MIT License. They are distributed under the Creative Commons Attribution 4.0
International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/):

- Model weights: OPUS-MT, by Jörg Tiedemann and the Language Technology Research
  Group at the University of Helsinki (https://github.com/Helsinki-NLP/Opus-MT,
  https://huggingface.co/Helsinki-NLP), CC BY 4.0.
- ONNX exports: 44 pairs from Xenova (https://huggingface.co/Xenova, the
  `Xenova/opus-mt-<pair>` repositories, at the revisions pinned in
  data/sources.json). Five pairs (fr-ar, es-ar, ar-fr, ar-es, ar-de) exported
  from the `Helsinki-NLP/opus-mt-<pair>` weights with Hugging Face Optimum in the
  same layout (first published in the OpenCircle-models repository).
- Modifications: in every pair, the file onnx/decoder_model_merged_quantized.onnx
  was modified. The Transpose and DequantizeLinear nodes that build the output
  projection were moved out of both branches of the decoder's If node into the
  main graph (scripts/convert.py). No weight was changed, retrained or
  re-quantized. For the five pairs exported from the Helsinki-NLP weights,
  tokenizer.json and special_tokens_map.json were generated and
  tokenizer_config.json completed in the layout of the Xenova exports
  (scripts/build_hf_repos.py). All other files are unchanged.

No endorsement by the original authors is implied.

Evaluation data: FLORES-200 (NLLB Team et al., CC BY-SA 4.0) is downloaded by
scripts/flores.py for scoring only; it is not redistributed here.
