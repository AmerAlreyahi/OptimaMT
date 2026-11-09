// Runs bench.html in headless Chromium (transformers.js with onnxruntime-web, WASM, 4 threads) for one
// configuration and prints one JSON line: the browser's total RSS (all its processes) before, after loading
// the model and after the batch, and the decoder speed.
//   node browser_bench.mjs <serve root> <chromium executable> <pair> <original|optimamt> <0|1 qdq off> <lines.json>
import { createServer } from "node:http";
import { readFileSync, readdirSync, statSync, createReadStream } from "node:fs";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright-core";

const [root, exe, pair, variant, qdq, linesFile] = process.argv.slice(2);
const dist = new URL("./node_modules/@huggingface/transformers/dist/", import.meta.url).pathname;
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                ".json": "application/json" };

const server = createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const base = url.startsWith("/tjs/") ? dist : root;
  const path = normalize(join(base, url.replace(/^\/tjs\//, "/")));
  if (!path.startsWith(normalize(base))) { res.writeHead(403).end(); return; }
  let size;
  try { size = statSync(path).size; } catch { res.writeHead(404).end(); return; }
  res.writeHead(200, { "Content-Type": types[extname(path)] || "application/octet-stream", "Content-Length": size,
                       "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" });
  createReadStream(path).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

function treeRss(pid) {  // MiB of every descendant process
  let kb = 0;
  for (const tid of readdirSync(`/proc/${pid}/task`)) {
    let kids = "";
    try { kids = readFileSync(`/proc/${pid}/task/${tid}/children`, "utf8").trim(); } catch {}
    for (const c of kids.split(/\s+/).filter(Boolean)) {
      try {
        const m = readFileSync(`/proc/${c}/status`, "utf8").match(/VmRSS:\s+(\d+)/);
        kb += m ? Number(m[1]) : 0;
      } catch {}
      kb += treeRss(c) * 1024;
    }
  }
  return kb / 1024;
}

const browser = await chromium.launch({ executablePath: exe, headless: true,
                                        args: ["--enable-blink-features=ForceEagerMeasureMemory"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("pageerror:", e.message));
page.on("console", (m) => { console.error("console:", m.type(), m.text()); });
const lines = readFileSync(linesFile, "utf8");
const baseRss = treeRss(process.pid);
await page.goto(`http://127.0.0.1:${port}/bench.html?pair=${pair}&variant=${variant}&qdq=${qdq}` +
                `&lines=${encodeURIComponent(lines)}`);
const timeout = Number(process.env.BENCH_TIMEOUT || 600000);
await page.waitForFunction(() => window.__loaded || window.__result, null, { timeout });
const loadedRss = treeRss(process.pid);
await page.evaluate(() => window.__continue && window.__continue());
await page.waitForFunction(() => window.__result, null, { timeout });
const result = await page.evaluate(() => window.__result);
result.rss_before_mib = baseRss;
result.rss_loaded_mib = loadedRss;
result.rss_after_mib = treeRss(process.pid);
result.chromium = browser.version();
console.log(JSON.stringify({ pair, variant, qdq_off: qdq === "1", ...result }));
await browser.close();
server.close();
