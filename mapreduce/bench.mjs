// Benchmark: semantic MapReduce (Jev) vs a pure-code keyword baseline.
// Phases timed separately (map = n parallel requests, reduce = 1 batched
// request). Reports wall time, per-doc map latency, and topic-key agreement
// with a rule-based classifier across input sizes.
import { mapDoc, semanticMapReduce } from "./mapreduce.mjs";

// --- Pure-code baseline: keyword rules emit (key, values) ---
const RULES = [
  ["pricing", /charg|pric|refund|renewal|competitor|plan|invoice|billing/i],
  ["performance", /slow|latency|load|seconds|speed|lag|slow to open/i],
  ["support", /support|docs|documentation|answered|help/i],
  ["reliability", /outage|lost|data loss|crash|bug|backup|stability/i],
];

function baselineMap(docs) {
  return docs.map((text) => {
    const key = RULES.find(([, re]) => re.test(text))?.[0] ?? "other";
    return { key };
  });
}

function shuffleGroups(mapped) {
  const g = {};
  for (const m of mapped) (g[m.key] ??= []).push(m);
  return g;
}

const POOL = [
  "Pricing doubled this year for the same plan. We're evaluating competitors at renewal.",
  "We were charged twice for October and the refund flow failed.",
  "The dashboard takes 12 seconds to load on our biggest account. Sales demos are embarrassing.",
  "Search latency is terrible on large workspaces, requests lag for seconds.",
  "Support answered in minutes and fixed our webhook issue the same day. Genuinely great.",
  "The API docs are three versions out of date.",
  "We lost a day of data during Tuesday's outage and never got a postmortem.",
  "The mobile app crashes on sync and support never acknowledged the bug.",
  "Fine overall, though billing emails land in spam.",
  "Plans and pricing pages are confusing; the enterprise tier pricing isn't published.",
  "The editor is slow to open large files.",
  "Chats with support were helpful and quick.",
  "Backup reliability worries me after last week's data loss incident.",
  "Refund took a month and the invoice was wrong.",
  "Autosave works well now, no crashes in weeks.",
];

const corpus = (n) => Array.from({ length: n }, (_, i) => POOL[i % POOL.length]);
const agreement = (jev, base) => `${jev.filter((m, i) => m.key === base[i].key).length}/${base.length}`;

console.log("n   map_ms  reduce_ms  total_ms  requests  agree(jev vs keyword rules)");
for (const n of [3, 6, 12]) {
  const docs = corpus(n);
  const base = baselineMap(docs);

  const t0 = performance.now();
  const mapped = await Promise.all(docs.map((d, i) => mapDoc(d, i)));
  const mapMs = performance.now() - t0;

  const out = await semanticMapReduce(docs);
  const total = performance.now() - t0;
  const reduceMs = total - mapMs;

  console.log(
    `${String(n).padEnd(3)} ${String(Math.round(mapMs)).padEnd(7)} ${String(Math.round(reduceMs)).padEnd(10)} ${String(Math.round(total)).padEnd(9)} ${n + 1}         ${agreement(out.mapped, base)}`,
  );
}
