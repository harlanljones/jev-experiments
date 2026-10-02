// Clef client (Cloudflare Workers AI decision model, @cf/cloudflare/clef[-flash]).
// Follows the System One API: the dataset question objects built by
// datasets.mjs via @typesafe-ai/sdk are wire-compatible with the /ai/run
// endpoint, so they are passed through unchanged.
//
// Auth: CLOUDFLARE_API_TOKEN env var, or the OAuth token wrangler stores at
// ~/.config/.wrangler/config/default.toml (requires the `ai` scope).
// Account: CLOUDFLARE_ACCOUNT_ID env var, or parsed from wrangler whoami config.
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { DATASETS } from "./datasets.mjs";

let cachedAccount = null;
async function accountId() {
  if (cachedAccount) return cachedAccount;
  if (process.env.CLOUDFLARE_ACCOUNT_ID) return (cachedAccount = process.env.CLOUDFLARE_ACCOUNT_ID);
  const res = await fetch("https://api.cloudflare.com/client/v4/accounts", {
    headers: { Authorization: `Bearer ${authToken()}` },
  });
  const body = await res.json();
  if (!body.success || !body.result?.length) throw new Error(`Cannot resolve account id: ${JSON.stringify(body.errors)}`);
  return (cachedAccount = body.result[0].id);
}

function authToken() {
  if (process.env.CLOUDFLARE_API_TOKEN) return process.env.CLOUDFLARE_API_TOKEN;
  const cfg = fs.readFileSync(path.join(os.homedir(), ".config/.wrangler/config/default.toml"), "utf8");
  const m = cfg.match(/oauth_token\s*=\s*"([^"]+)"/);
  if (m) return m[1];
  throw new Error("No CLOUDFLARE_API_TOKEN env var and no oauth_token in wrangler config");
}

export async function clefRun(text, questions, model = "clef") {
  const res = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${await accountId()}/ai/run/@cf/cloudflare/${model}`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${authToken()}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, state: { text }, questions }),
    },
  );
  const body = await res.json();
  if (!body.success) throw new Error(`Clef request failed: ${JSON.stringify(body.errors)}`);
  const a = body.result.answers;
  const choiceKey = Object.keys(a).find((k) => a[k].type === "choice");
  const scoreKey = Object.keys(a).find((k) => a[k].type === "score");
  return {
    choice: choiceKey ? a[choiceKey].choice : null,
    choiceConfidence: choiceKey ? a[choiceKey].confidence : null,
    choiceProbs: choiceKey ? a[choiceKey].probabilities : null,
    score: scoreKey ? Math.round(a[scoreKey].score) : null,
    scoreConfidence: scoreKey ? a[scoreKey].confidence : null,
    scoreProbs: scoreKey ? a[scoreKey].probabilities : null,
  };
}

// Run a dataset's questions over texts with bounded concurrency.
export async function clefClassifyAll(texts, datasetName, { model = "clef", concurrency = 8 } = {}) {
  const ds = DATASETS[datasetName];
  const questions = { ...ds.choice.question(), ...(ds.choice.scoreQuestion?.() ?? {}) };
  const out = new Array(texts.length);
  let next = 0;
  async function worker() {
    while (next < texts.length) {
      const i = next++;
      out[i] = await clefRun(texts[i], questions, model);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, texts.length) }, worker));
  return out;
}
