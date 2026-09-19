// Benchmark harness. Protocol: 3 seeds x full Jev pipeline (map + batched
// reduce); A1 (topic-only map) and A2 (per-group reduce) run once.
// Writes results to bench-results.json and a markdown report to report.md.
import fs from "node:fs";
import { choice, noul, score } from "@typesafe-ai/sdk";
import { client } from "../samples/shared/client.mjs";
import { mapDoc } from "./mapreduce.mjs";
import { b0Topic, b1Sentiment, verdictTruth, accuracy, mae, brier, auc, precisionRecall, limiter } from "./lib.mjs";

const SEEDS = 3;
const CONCURRENCY = 6;
const { docs } = JSON.parse(fs.readFileSync("fixtures/corpus.json", "utf8"));
const TIERS = ["T1", "T2", "T3"];
const run = limiter(CONCURRENCY);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sd = (a) => (a.length > 1 ? Math.sqrt(mean(a.map((x) => (x - mean(a)) ** 2)) / (a.length - 1)) : 0);
const byTier = (docs) => TIERS.map((t) => docs.filter((d) => d.tier === t));
const fmt = (x, d = 2) => Number.isNaN(x) ? "n/a" : x.toFixed(d);
const msd = (a, d = 3) => `${fmt(mean(a), d)} ± ${fmt(sd(a), d)}`;

// ---------- Full pipeline per seed ----------
const topicAccSeeds = Object.fromEntries(TIERS.map((t) => [t, []]));
const sentMaeSeeds = Object.fromEntries(TIERS.map((t) => [t, []]));
const actionableProbs = [], actionableTruth = [];
const seedRuns = [];

for (let seed = 0; seed < SEEDS; seed++) {
  const t0 = performance.now();
  const mapped = await Promise.all(docs.map((d, i) => run(() => mapDoc({ text: d.text }, i))));
  const mapMs = performance.now() - t0;

  // shuffle
  const groups = {};
  for (const [i, m] of mapped.entries()) (groups[m.key] ??= []).push({ ...m, truth: docs[i].truth });

  // batched reduce (the app's design)
  const agg = Object.fromEntries(Object.entries(groups).map(([k, g]) => [k, {
    doc_count: g.length,
    mean_sentiment: +(mean(g.map((v) => v.sentiment))).toFixed(2),
    max_urgency: Math.max(...g.map((v) => v.urgency)),
    actionable_docs: g.filter((v) => v.actionable >= 0.6).length,
  }]));
  const rq = Object.fromEntries(Object.keys(groups).map((k) => [`verdict_${k}`, choice(`Given the aggregated signals for the '${k}' topic, what should the team do?`, { act_now: "Escalate this week: assign an owner and ship a fix or response.", track: "Log and monitor; no immediate action.", ignore: "Not worth tracking; low value or out of scope." })]));
  const t1 = performance.now();
  const rr = await client.systemOne({ state: { groups: agg }, questions: rq });
  const reduceMs = performance.now() - t1;

  // ---- per-doc metrics (truth is per-doc; evaluate mapped docs) ----
  for (const t of TIERS) {
    const idx = docs.map((d, i) => [d, i]).filter(([d]) => d.tier === t);
    topicAccSeeds[t].push(accuracy(idx.map(([, i]) => mapped[i].key), idx.map(([d]) => d.truth.topic)));
    sentMaeSeeds[t].push(mae(idx.map(([, i]) => mapped[i].sentiment), idx.map(([d]) => d.truth.s)));
  }
  mapped.forEach((m, i) => { actionableProbs.push(m.actionable); actionableTruth.push(docs[i].truth.actionable ? 1 : 0); });

  // ---- reduce verdict metrics: policy truth per Jev key group ----
  const truthByGroup = {}, predByGroup = {};
  for (const [k, g] of Object.entries(groups)) {
    truthByGroup[k] = verdictTruth(g);
    predByGroup[k] = rr.answers[`verdict_${k}`].choice;
  }
  const keys = Object.keys(groups);
  const tp = precisionRecall(keys.map((k) => predByGroup[k]), keys.map((k) => truthByGroup[k]), "act_now");

  seedRuns.push({ mapMs, reduceMs, totalMs: mapMs + reduceMs, requests: docs.length + 1, act_now: tp, verdictAgreement: mean(keys.map((k) => predByGroup[k] === truthByGroup[k] ? 1 : 0)) });
  console.log(`seed ${seed}: map ${Math.round(mapMs)}ms, reduce ${Math.round(reduceMs)}ms, act_now P/R ${fmt(tp.precision)}/${fmt(tp.recall)}`);
}

// ---------- A1: topic-only map (single seed) ----------
const a1Questions = { topic: choice("Which topic does this feedback concern?", { pricing: "Cost, plans, billing, value for money.", performance: "Speed, latency, responsiveness, throughput.", support: "Help, documentation, responsiveness of the team.", reliability: "Bugs, outages, data loss, stability.", other: null }) };
const a1 = await Promise.all(docs.map((d, i) => run(() => client.systemOne({ state: { text: d.text }, questions: a1Questions }).then((r) => r.answers.topic.choice))));
const a1Acc = Object.fromEntries(TIERS.map((t) => [t, accuracy(docs.map((d, i) => [d, i]).filter(([d]) => d.tier === t).map(([, i]) => a1[i]), docs.filter((d) => d.tier === t).map((d) => d.truth.topic))]));

// ---------- A2: one reduce request per group (single seed) ----------
const mapped0 = await Promise.all(docs.map((d, i) => run(() => mapDoc({ text: d.text }, i))));
const groups0 = {};
for (const [i, m] of mapped0.entries()) (groups0[m.key] ??= []).push({ ...m, truth: docs[i].truth });
const agg0 = Object.fromEntries(Object.entries(groups0).map(([k, g]) => [k, { doc_count: g.length, mean_sentiment: +mean(g.map((v) => v.sentiment)).toFixed(2), max_urgency: Math.max(...g.map((v) => v.urgency)), actionable_docs: g.filter((v) => v.actionable >= 0.6).length }]));
const rq0 = Object.fromEntries(Object.keys(groups0).map((k) => [`verdict_${k}`, choice(`Given the aggregated signals for the '${k}' topic, what should the team do?`, { act_now: "Escalate this week: assign an owner and ship a fix or response.", track: "Log and monitor; no immediate action.", ignore: "Not worth tracking; low value or out of scope." })]));
const rr0 = await client.systemOne({ state: { groups: agg0 }, questions: rq0 });
const keys0 = Object.keys(groups0);
const batchedAgree = mean(keys0.map((k) => rr0.answers[`verdict_${k}`].choice === verdictTruth(groups0[k]) ? 1 : 0));
const a2Pred = {};
for (const k of keys0) {
  const r = await client.systemOne({ state: { group: agg0[k] }, questions: { verdict: choice(`Given the aggregated signals for the '${k}' topic, what should the team do?`, { act_now: "Escalate this week: assign an owner and ship a fix or response.", track: "Log and monitor; no immediate action.", ignore: "Not worth tracking; low value or out of scope." }) } });
  a2Pred[k] = r.answers.verdict.choice;
}
const a2Agree = mean(keys0.map((k) => a2Pred[k] === verdictTruth(groups0[k]) ? 1 : 0));

// ---------- Baselines (deterministic, no seeds) ----------
const b0Acc = Object.fromEntries(TIERS.map((t) => [t, accuracy(docs.filter((d) => d.tier === t).map((d) => b0Topic(d.text)), docs.filter((d) => d.tier === t).map((d) => d.truth.topic))]));
const b1Mae = Object.fromEntries(TIERS.map((t) => [t, mae(docs.filter((d) => d.tier === t).map((d) => b1Sentiment(d.text)), docs.filter((d) => d.tier === t).map((d) => d.truth.s))]));

// ---------- Report ----------
const rows = TIERS.map((t) => `| ${t} | ${msd(topicAccSeeds[t])} | ${fmt(b0Acc[t])} | ${fmt(a1Acc[t])} | ${msd(sentMaeSeeds[t])} | ${fmt(b1Mae[t])} |`).join("\n");
const verdictRows = seedRuns.map((s, i) => `| ${i} | ${fmt(s.act_now.precision)} | ${fmt(s.act_now.recall)} | ${fmt(s.verdictAgreement)} | ${s.act_now.tp}/${s.act_now.fp}/${s.act_now.fn} |`).join("\n");

const report = `# Semantic MapReduce benchmark — results

Fixture: fixtures/corpus.json (90 docs, 30/tier, labels planted by construction, seed 20260918).
Protocol: ${SEEDS} seeds x full pipeline (map n=90 parallel requests + 1 batched reduce); A1/A2 single seed.

## Per-doc quality (mean ± sd across seeds)

| Tier | Jev topic acc | B0 keywords | A1 topic-only Jev | Jev sentiment MAE | B1 lexicon MAE |
|------|--------------|-------------|-------------------|-------------------|----------------|
${rows}

Actionable noul (pooled over ${SEEDS} seeds x 90 docs): Brier ${fmt(brier(actionableProbs, actionableTruth))}, AUC ${fmt(auc(actionableProbs, actionableTruth))}

## Reduce verdicts vs policy ground truth (per seed)

| Seed | act_now precision | act_now recall | verdict agreement | tp/fp/fn |
|------|-------------------|----------------|-------------------|----------|
${verdictRows}

## Ablation: batched vs per-group reduce (single seed)

Verdict agreement with policy truth (same map, same groups) — batched (1 request): ${fmt(batchedAgree)}, per-group (${keys0.length} requests): ${fmt(a2Agree)}
Per-group verdicts: ${JSON.stringify(a2Pred)}

## Latency (wall time, per seed)

| Seed | map_ms (90 parallel) | reduce_ms | total_ms | requests |
|------|----------------------|-----------|----------|----------|
${seedRuns.map((s, i) => `| ${i} | ${Math.round(s.mapMs)} | ${Math.round(s.reduceMs)} | ${Math.round(s.totalMs)} | ${s.requests} |`).join("\n")}
`;
fs.writeFileSync("report.md", report);
fs.writeFileSync("bench-results.json", JSON.stringify({ topicAccSeeds, sentMaeSeeds, b0Acc, b1Mae, a1Acc, brier: brier(actionableProbs, actionableTruth), auc: auc(actionableProbs, actionableTruth), seedRuns, a2Agree }, null, 2));
console.log(report);
