// Experiment 2 — calibration: are reported confidences trustworthy enough to
// route on? Compares Jev's per-answer confidence against TF-IDF logreg's max
// softmax probability (n=50 gold labels), on the choice and score tasks of
// support-tickets and tickets-hard (80 test items each, seed 42).
//
// Metrics per system/task:
//   ECE        expected calibration error over 10 confidence bins
//   bins       accuracy within each confidence bin (reliability table)
//   cov@90/95  fraction of items you can auto-accept while keeping
//              accuracy >= 0.90 / 0.95 on the accepted set (risk-coverage)
//   AURC       area under the risk-coverage curve (lower is better)
import { DATASETS } from "./datasets.mjs";
import { trainLogReg } from "./ml-baselines.mjs";
import { jevClassifyAll } from "./jev.mjs";
import { clefClassifyAll } from "./clef.mjs";
import fs from "node:fs";

const N_TEST = 80;
const BIN_EDGES = Array.from({ length: 11 }, (_, i) => i / 10);

function calibration(confGold) {
  // confGold: [{conf, correct}]
  const bins = BIN_EDGES.slice(0, -1).map((lo, i) => ({ lo, hi: BIN_EDGES[i + 1], n: 0, correct: 0, confSum: 0 }));
  for (const { conf, correct } of confGold) {
    const b = bins[Math.min(9, Math.floor(conf * 10))];
    b.n++; b.correct += correct ? 1 : 0; b.confSum += conf;
  }
  const ece = bins.reduce((s, b) => (b.n ? s + (b.n / confGold.length) * Math.abs(b.correct / b.n - b.confSum / b.n) : s), 0);

  // risk-coverage: sort by confidence desc; err@k = error rate among top k
  const sorted = [...confGold].sort((a, b) => b.conf - a.conf);
  const errAt = [];
  let wrong = 0;
  sorted.forEach((d, i) => { wrong += d.correct ? 0 : 1; errAt.push(wrong / (i + 1)); });
  const aurc = errAt.reduce((a, b) => a + b, 0) / errAt.length;
  const covAt = (target) => {
    for (let k = errAt.length; k >= 1; k--) if (errAt[k - 1] <= 1 - target) return k / errAt.length;
    return 0;
  };
  return { ece, bins, covAt90: covAt(0.9), covAt95: covAt(0.95), aurc };
}

const out = {};
for (const name of ["support-tickets", "tickets-hard"]) {
  const ds = DATASETS[name];
  const { train, test } = ds.gen({ nTrain: 50, nTest: N_TEST, seed: 42 });

  const lrC = trainLogReg(train.map((t) => t.text), train.map((t) => t.label));
  const lrS = trainLogReg(train.map((t) => t.text), train.map((t) => String(t.score)));
  const [jev, clef, flash] = await Promise.all([
    jevClassifyAll(test.map((t) => t.text), name),
    clefClassifyAll(test.map((t) => t.text), name, { model: "clef" }),
    clefClassifyAll(test.map((t) => t.text), name, { model: "clef-flash" }),
  ]);

  // Clef also exposes full per-option probabilities; "maxp" uses the highest
  // option probability instead of the reported confidence, to see which of the
  // two signals routes better.
  const clefMaxp = (key) => (o) => Math.max(...Object.values(o[key]));
  const systems = {
    "jev": {
      choice: jev.map((o, i) => ({ conf: o.choiceConfidence, correct: o.choice === test[i].label })),
      score: jev.map((o, i) => ({ conf: o.scoreConfidence, correct: o.score === test[i].score })),
    },
    "clef": {
      choice: clef.map((o, i) => ({ conf: o.choiceConfidence, correct: o.choice === test[i].label })),
      score: clef.map((o, i) => ({ conf: o.scoreConfidence, correct: o.score === test[i].score })),
    },
    "clef(maxp)": {
      choice: clef.map((o, i) => ({ conf: clefMaxp("choiceProbs")(o), correct: o.choice === test[i].label })),
      score: clef.map((o, i) => ({ conf: clefMaxp("scoreProbs")(o), correct: o.score === test[i].score })),
    },
    "clef-flash": {
      choice: flash.map((o, i) => ({ conf: o.choiceConfidence, correct: o.choice === test[i].label })),
      score: flash.map((o, i) => ({ conf: o.scoreConfidence, correct: o.score === test[i].score })),
    },
    "logreg": {
      choice: test.map((t) => { const p = lrC.predictProba(t.text); return { conf: p.max, correct: p.argmax === t.label }; }),
      score: test.map((t) => { const p = lrS.predictProba(t.text); return { conf: p.max, correct: String(p.argmax) === String(t.score) }; }),
    },
  };

  out[name] = {};
  for (const [sys, tasks] of Object.entries(systems)) {
    out[name][sys] = {};
    for (const [task, confGold] of Object.entries(tasks)) {
      const c = calibration(confGold);
      out[name][sys][task] = {
        acc: confGold.filter((d) => d.correct).length / confGold.length,
        ece: c.ece, covAt90: c.covAt90, covAt95: c.covAt95, aurc: c.aurc,
        bins: c.bins.filter((b) => b.n > 0).map((b) => ({ conf: +((b.lo + b.hi) / 2).toFixed(1), n: b.n, acc: +(b.correct / b.n).toFixed(2), meanConf: +(b.confSum / b.n).toFixed(2) })),
      };
    }
  }

  console.log(`\n=== ${name} (n=${N_TEST}) ===`);
  console.log("system  task     acc     ECE    cov@90  cov@95   AURC");
  for (const [sys, tasks] of Object.entries(out[name]))
    for (const [task, m] of Object.entries(tasks))
      console.log(
        sys.padEnd(8), task.padEnd(8),
        `${(m.acc * 100).toFixed(1)}%`.padStart(6),
        m.ece.toFixed(3).padStart(6),
        `${(m.covAt90 * 100).toFixed(0)}%`.padStart(7),
        `${(m.covAt95 * 100).toFixed(0)}%`.padStart(7),
        m.aurc.toFixed(3).padStart(7),
      );
}

fs.writeFileSync(new URL("./exp-calibration-results.json", import.meta.url), JSON.stringify(out, null, 2));
console.log("\nwrote exp-calibration-results.json");
