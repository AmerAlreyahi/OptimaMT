---
title: $name
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# $name

**$long_name: $tagline**

$count OPUS-MT translation models for ONNX Runtime and transformers.js: drop-in copies of the int8 exports
with a lighter merged decoder. Same weights, same tokens.

| Memory after load (median) | Linux | Windows |
|---|---:|---:|
| `disable_quant_qdq = 1` | $qdq_v1 → $qdq_v2 MiB ($qdq_pct %) | $win_qdq_v1 → $win_qdq_v2 MiB ($win_qdq_pct %) |
| ONNX Runtime defaults | $def_v1 → $def_v2 MiB ($def_pct %) | $win_def_v1 → $win_def_v2 MiB ($win_def_pct %) |

Repo ids `$scheme`, e.g. [`$example`](https://huggingface.co/$example) ·
[interactive results]($results) · [method and code](https://github.com/$github)

Models CC BY 4.0 (OPUS-MT, University of Helsinki; ONNX exports by Xenova); code MIT.
