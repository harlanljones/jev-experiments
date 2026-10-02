// Experiment 3 — decision models as labeling engines (distillation): label a
// 1000-item synthetic pool with Clef and Clef-flash, distill each into TF-IDF
// logreg, and evaluate on held-out gold-labeled test sets (Jev kept as a
// zero-shot reference row; its distillation results from the original run
// were: choice 99.0%, score 67.7%).
//
// Deployment options compared:
//   A. logreg trained on 50 gold labels (classic supervised)
//   B. logreg trained on 1000 Clef labels (distilled, no human labels)
//   C. Clef zero-shot at runtime (requests scale with traffic)
import { DATASETS } from "./datasets.mjs";
import { trainLogReg } from "./ml-baselines.mjs";
import { jevClassifyAll } from "./jev.mjs";
import { clefClassifyAll } from "./clef.mjs";
import fs from "node:fs";

const ds = DATASETS["support-tickets"];
const POOL = 1000;

const goldTests = [42, 7, 123].flatMap((s) => ds.gen({ nTrain: 50, nTest: 32, seed: s }).test);
const goldTexts = new Set(goldTests.map((t) => t.text));
const pool = ds.gen({ nTrain: POOL + goldTexts.size, nTest: 0, seed: 500 }).train
  .filter((t) => !goldTexts.has(t.text))
  .slice(0, POOL);

const acc = (pred, gold) => pred.filter((p, i) => p === gold[i]).length / gold.length;
const mae = (pred, gold) => pred.reduce((s, p, i) => s + Math.abs(Number(p) - gold[i]), 0) / gold.length;

async function labelPool(model) {
  const t0 = performance.now();
  const labels = await clefClassifyAll(pool.map((t) => t.text), "support-tickets", { model, concurrency: 12 });
  const ms = performance.now() - t0;
  const agree = {
    choice: labels.filter((o, i) => o.choice === pool[i].label).length / POOL,
    score: labels.filter((o, i) => o.score === pool[i].score).length / POOL,
  };
  return { labels, ms, agree };
}

async function zeroShot(model) {
  const out = model === "jev"
    ? await jevClassifyAll(goldTests.map((t) => t.text), "support-tickets")
    : await clefClassifyAll(goldTests.map((t) => t.text), "support-tickets", { model });
  return {
    name: `${model}-zero-shot`,
    requests: goldTests.length,
    choiceAcc: acc(out.map((o) => o.choice), goldTests.map((t) => t.label)),
    scoreAcc: acc(out.map((o) => o.score).map(String), goldTests.map((t) => String(t.score))),
    scoreMae: mae(out.map((o) => o.score), goldTests.map((t) => t.score)),
  };
}

async function distillRow(model, labeling) {
  const t0 = performance.now();
  const distC = trainLogReg(pool.map((t) => t.text), labeling.labels.map((o) => o.choice));
  const distS = trainLogReg(pool.map((t) => t.text), labeling.labels.map((o) => String(o.score)));
  const trainMs = performance.now() - t0;
  return {
    name: `logreg(1000 ${model}-labels)`,
    requests: POOL,
    choiceAcc: acc(goldTests.map((t) => distC.predict(t.text)), goldTests.map((t) => t.label)),
    scoreAcc: acc(goldTests.map((t) => distS.predict(t.text)).map(String), goldTests.map((t) => String(t.score))),
    scoreMae: mae(goldTests.map((t) => Number(distS.predict(t.text))), goldTests.map((t) => t.score)),
    trainMs: Math.round(trainMs),
  };
}

// Classic baseline: 50 gold labels
const goldTrain = ds.gen({ nTrain: 50, nTest: 0, seed: 42 }).train;
const goldC = trainLogReg(goldTrain.map((t) => t.text), goldTrain.map((t) => t.label));
const goldS = trainLogReg(goldTrain.map((t) => t.text), goldTrain.map((t) => String(t.score)));

const [labelingClef, labelingFlash, jevRow, clefRow, flashRow] = await Promise.all([
  labelPool("clef"),
  labelPool("clef-flash"),
  zeroShot("jev"),
  zeroShot("clef"),
  zeroShot("clef-flash"),
]);
const [distClef, distFlash] = await Promise.all([distillRow("clef", labelingClef), distillRow("clef-flash", labelingFlash)]);

const rows = [
  {
    name: "logreg(50 gold)",
    requests: 0,
    choiceAcc: acc(goldTests.map((t) => goldC.predict(t.text)), goldTests.map((t) => t.label)),
    scoreAcc: acc(goldTests.map((t) => goldS.predict(t.text)).map(String), goldTests.map((t) => String(t.score))),
    scoreMae: mae(goldTests.map((t) => Number(goldS.predict(t.text))), goldTests.map((t) => t.score)),
  },
  distClef, distFlash, jevRow, clefRow, flashRow,
];

const result = {
  pool: POOL, goldTestSize: goldTests.length,
  labeling: {
    clef: { wallMs: Math.round(labelingClef.ms), msPerItem: +(labelingClef.ms / POOL).toFixed(1), goldAgreement: labelingClef.agree },
    "clef-flash": { wallMs: Math.round(labelingFlash.ms), msPerItem: +(labelingFlash.ms / POOL).toFixed(1), goldAgreement: labelingFlash.agree },
  },
  rows,
};
fs.writeFileSync(new URL("./exp-distill-results.json", import.meta.url), JSON.stringify(result, null, 2));

console.log(`pool: ${POOL} items labeled per teacher | gold agreement:`);
for (const m of ["clef", "clef-flash"]) {
  const l = result.labeling[m];
  console.log(`  ${m.padEnd(11)} ${l.wallMs}ms wall (${l.msPerItem} ms/item) — choice ${(l.goldAgreement.choice * 100).toFixed(1)}%, score ${(l.goldAgreement.score * 100).toFixed(1)}%`);
}
console.log("\ndeployment option             requests  choiceAcc  scoreAcc  scoreMAE");
for (const r of rows) {
  console.log(
    r.name.padEnd(28),
    String(r.requests).padStart(6),
    `${(r.choiceAcc * 100).toFixed(1)}%`.padStart(9),
    `${(r.scoreAcc * 100).toFixed(1)}%`.padStart(9),
    r.scoreMae.toFixed(2).padStart(9),
  );
}
