// Translates lines with transformers.js, from a local model folder or a Hub repo.
//   node translate.mjs local <parent dir> <folder name> <lines.json>
//   node translate.mjs hub <repo id> <revision> <lines.json>
// QDQ_OFF=1 sets ONNX Runtime's session.disable_quant_qdq = 1.
// Prints one JSON object: the translations and the process RSS (MiB) before and after loading.
import { readFileSync } from "node:fs";
import { env, pipeline } from "@huggingface/transformers";

const [mode, a, b, file] = process.argv.slice(2);
const lines = JSON.parse(readFileSync(file, "utf8"));
let model, options = { dtype: "q8" };
if (process.env.QDQ_OFF === "1") {
  options.session_options = { extra: { session: { disable_quant_qdq: "1" } } };
}
if (mode === "local") {
  env.localModelPath = a;
  env.allowRemoteModels = false;
  model = b;
} else {
  env.allowLocalModels = false;
  model = a;
  options.revision = b;
}
const mib = () => Math.round((process.memoryUsage().rss / 2 ** 20) * 10) / 10;
const before = mib();
const translator = await pipeline("translation", model, options);
const loaded = mib();
const out = [];
for (const line of lines) {
  const [r] = await translator(line, { max_new_tokens: 256 });
  out.push(r.translation_text);
}
console.log(JSON.stringify({ translations: out, rss_before_mib: before, rss_loaded_mib: loaded, rss_after_mib: mib() }));
