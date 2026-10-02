// Benchmark: Jev classifiers vs traditional ML across four generated datasets
// (see datasets.mjs). For each dataset and seed we run: keyword rules,
// Naive Bayes, TF-IDF softmax regression (n = 10/25/50), and zero-shot Jev.
// Tasks: a choice task (all datasets) and an optional score task (tickets,
// reviews). Metrics: accuracy, macro-F1, and MAE for score tasks. Seeds are
// aggregated into mean ± std. sklearn-check.py cross-checks seed 42 corpora.
import { DATASETS, TEST_SIZE, TRAIN_SIZE, SEEDS } from "./datasets.mjs";
import { trainNaiveBayes, trainLogReg } from "./ml-baselines.mjs";
import { jevClassifyAll } from "./jev.mjs";
import { clefClassifyAll } from "./clef.mjs";
import fs from "node:fs";

const TRAIN_SIZES = [10, 25, 50];
const DECISION_SYSTEMS = [
  ["jev-zero-shot", (texts, ds) => jevClassifyAll(texts, ds)],
  ["clef", (texts, ds) => clefClassifyAll(texts, ds, { model: "clef" })],
  ["clef-flash", (texts, ds) => clefClassifyAll(texts, ds, { model: "clef-flash" })],
];

const acc = (pred, gold) => pred.filter((p, i) => p === gold[i]).length / pred.length;
const mae = (pred, gold) => pred.reduce((s, p, i) => s + Math.abs(Number(p) - gold[i]), 0) / pred.length;

function macroF1(pred, gold, classes) {
  const f1s = classes.map((c) => {
    const tp = pred.filter((p, i) => p === c && gold[i] === c).length;
    const fp = pred.filter((p, i) => p === c && gold[i] !== c).length;
    const fn = pred.filter((p, i) => p !== c && gold[i] === c).length;
    return tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
  });
  return f1s.reduce((a, b) => a + b, 0) / classes.length;
}

let train, test, ds;

function runSystem(predictChoice, predictScore, inferMs, trainMs, requests) {
  const goldC = test.map((t) => t.label);
  const predC = test.map(predictChoice);
  const m = {
    choiceAcc: acc(predC, goldC),
    choiceF1: macroF1(predC, goldC, ds.choice.classes),
    trainMs,
    inferMs,
    perItemMs: inferMs / test.length,
    requests,
  };
  if (ds.choice.scoreQuestion) {
    const goldS = test.map((t) => t.score);
    const predS = test.map(predictScore);
    m.scoreAcc = acc(predS.map(String), goldS.map(String));
    m.scoreMae = mae(predS, goldS);
  }
  return m;
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const std = (xs) => Math.sqrt(mean(xs.map((x) => (x - mean(xs)) ** 2)));
const agg = (runs) => {
  const keys = [...new Set(runs.flatMap((r) => Object.keys(r)))];
  return Object.fromEntries(keys.map((k) => {
    const vals = runs.map((r) => r[k]).filter((v) => v !== undefined);
    return vals.every((v) => typeof v === "number") ? [k, { mean: mean(vals), std: std(vals) }] : [k, vals[0]];
  }));
};

const report = { seeds: SEEDS, trainSizes: TRAIN_SIZES, testSize: TEST_SIZE, datasets: {} };
const corpusDump = {};

for (const [name, dsConfig] of Object.entries(DATASETS)) {
  ds = dsConfig;
  const systems = {};
  const push = (sysName, n, runs) => (systems[sysName] = { n, ...agg(runs) });

  const kwRuns = [], nbRuns = { 10: [], 25: [], 50: [] }, lrRuns = { 10: [], 25: [], 50: [] };
  const dmRuns = Object.fromEntries(DECISION_SYSTEMS.map(([name]) => [name, []]));
  let corpusSeed42 = null;

  for (const seed of SEEDS) {
    const corpus = ds.gen({ nTrain: TRAIN_SIZE, nTest: TEST_SIZE, seed });
    if (seed === SEEDS[0]) corpusDump[name] = corpus;
    train = corpus.train, test = corpus.test;

    {
      const t0 = performance.now();
      kwRuns.push(runSystem((t) => ds.keywords.choice(t.text), ds.keywords.score && ((t) => ds.keywords.score(t.text)), performance.now() - t0, 0, 0));
    }
    for (const n of TRAIN_SIZES) {
      const tr = train.slice(0, n);
      let t0 = performance.now();
      const nbC = trainNaiveBayes(tr.map((t) => t.text), tr.map((t) => t.label));
      const nbS = ds.choice.scoreQuestion && trainNaiveBayes(tr.map((t) => t.text), tr.map((t) => String(t.score)));
      const nbMs = performance.now() - t0;
      t0 = performance.now();
      const lrC = trainLogReg(tr.map((t) => t.text), tr.map((t) => t.label));
      const lrS = ds.choice.scoreQuestion && trainLogReg(tr.map((t) => t.text), tr.map((t) => String(t.score)));
      const lrMs = performance.now() - t0;
      nbRuns[n].push(runSystem((t) => nbC.predict(t.text), (t) => Number(nbS.predict(t.text)), 0, nbMs, 0));
      lrRuns[n].push(runSystem((t) => lrC.predict(t.text), (t) => Number(lrS.predict(t.text)), 0, lrMs, 0));
    }
    {
      const texts = test.map((t) => t.text);
      await Promise.all(DECISION_SYSTEMS.map(async ([sysName, run]) => {
        const t0 = performance.now();
        const out = await run(texts, name);
        const inferMs = performance.now() - t0;
        const runMetrics = runSystem((t, i) => out[i].choice, (t, i) => out[i].score, inferMs, 0, test.length);
        runMetrics.lowConfCategory = out.filter((o) => o.choiceConfidence < 0.5).length;
        runMetrics.meanConfidence = mean(out.map((o) => o.choiceConfidence));
        dmRuns[sysName].push(runMetrics);
      }));
    }
  }

  push("keyword-rules", 0, kwRuns);
  for (const n of TRAIN_SIZES) {
    push(`naive-bayes(n=${n})`, n, nbRuns[n]);
    push(`logreg-tfidf(n=${n})`, n, lrRuns[n]);
  }
  for (const [sysName] of DECISION_SYSTEMS) push(sysName, 0, dmRuns[sysName]);
  report.datasets[name] = { description: ds.description, systems };

  // console table
  console.log(`\n=== ${name} — ${ds.description} (test=${TEST_SIZE}, seeds=${SEEDS.join(",")}) ===`);
  console.log("system           n   choiceAcc      choiceF1       scoreAcc       scoreMAE    train_ms  infer_ms  req");
  for (const [sysName, s] of Object.entries(systems)) {
    const f = (k, digits = 1, suffix = "%") =>
      s[k] === undefined ? "      —   " : `${(s[k].mean * (suffix === "%" ? 100 : 1)).toFixed(digits)}±${(s[k].std * (suffix === "%" ? 100 : 1)).toFixed(digits)}${suffix}`.padStart(13);
    console.log(
      [
        sysName.padEnd(16),
        String(s.n).padStart(2),
        f("choiceAcc"), f("choiceF1"), f("scoreAcc"), f("scoreMae", 2, ""),
        s.trainMs ? String(Math.round(s.trainMs.mean)).padStart(8) : "       0",
        s.inferMs ? String(Math.round(s.inferMs.mean)).padStart(8) : "       0",
        String(Math.round(s.requests.mean)).padStart(4),
      ].join("  "),
    );
  }
}

fs.writeFileSync(new URL("./corpus.json", import.meta.url), JSON.stringify(corpusDump, null, 2));
fs.writeFileSync(new URL("./bench-results.json", import.meta.url), JSON.stringify(report, null, 2));
console.log("\nwrote corpus.json and bench-results.json");
