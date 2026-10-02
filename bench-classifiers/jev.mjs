// Jev classifier runner: one System One request per item answers the choice
// task (and optional score task) in parallel, zero-shot — no training data.
// Question definitions come from the dataset registry in datasets.mjs.
import { client } from "../samples/shared/client.mjs";
import { DATASETS } from "./datasets.mjs";

export async function jevClassifyOne(text, questions) {
  const r = await client.systemOne({ state: { text }, questions });
  const a = r.answers;
  const choiceKey = Object.keys(questions).find((k) => a[k].type === "choice");
  const scoreKey = Object.keys(questions).find((k) => a[k].type === "score");
  return {
    choice: a[choiceKey].choice,
    choiceConfidence: a[choiceKey].confidence,
    score: scoreKey ? Math.round(a[scoreKey].score) : null,
    scoreConfidence: scoreKey ? a[scoreKey].confidence : null,
  };
}

// Run over items with bounded concurrency.
export async function jevClassifyAll(texts, datasetName, { concurrency = 8 } = {}) {
  const ds = DATASETS[datasetName];
  const questions = { ...ds.choice.question(), ...(ds.choice.scoreQuestion?.() ?? {}) };
  const out = new Array(texts.length);
  let next = 0;
  async function worker() {
    while (next < texts.length) {
      const i = next++;
      out[i] = await jevClassifyOne(texts[i], questions);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, texts.length) }, worker));
  return out;
}
