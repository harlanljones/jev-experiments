// Baselines + metrics library.
// B0: keyword topic rules. B1: sentiment lexicon. Metrics: accuracy, MAE,
// Brier, AUC (rank-based), precision/recall for the act_now class.

export const KEYWORD_RULES = [
  ["pricing", /charg|pric|refund|renewal|competitor|invoice|billing|subscrip|budget/i],
  ["performance", /slow|sluggish|lag|latency|seconds|speed|responsive|faster|profile the hot/i],
  ["support", /support|ticket|docs|documentation|acknowledg|help article|walked us|churn response|reply/i],
  ["reliability", /crash|outage|lost|went down|fell over|corrupt|backup|data loss|postmortem|failover|retention|down/i],
];

export function b0Topic(text) {
  return KEYWORD_RULES.find(([, re]) => re.test(text))?.[0] ?? "other";
}

// B1: AFINN-style lexicon, scored to the 0-3 level scale used by the planted labels.
const LEX = { unacceptable: -3, "not acceptable": -3, double: -2, twice: -2, lost: -2, crashed: -2, crash: -2, outage: -2, corrupt: -2, nowhere: -2, "not okay": -2, "not reasonable": -2, "off the table": -2, "never acknowledged": -2, "went quiet": -2, slow: -1, sluggish: -1, lag: -1, laggy: -1, outdated: -1, frustrating: -1, crawl: -1, terrible: -2, embarrassing: -2, "cannot happen again": -2, vanished: -2, gone: -1, pending: -1, incorrect: -1, painfully: -2, great: 3, excellent: 3, lovely: 3, pleasure: 3, impressed: 3, happy: 2, delight: 2, smooth: 2, helpful: 2, quick: 1, fast: 1, "just worked": 3, "worth every cent": 3, steady: 1, "no complaints": 1, fine: 1, nice: 1, solid: 1, improved: 1 };

export function b1Sentiment(text) {
  const t = text.toLowerCase();
  let v = 0;
  for (const [w, s] of Object.entries(LEX)) if (t.includes(w)) v += s;
  if (v <= -3) return 0;
  if (v <= -1) return 1;
  if (v === 0) return 2;
  return 3;
}

// Reduce policy ground truth (same shape as the app's thresholds).
export const SENTIMENT_NEG = 1.5, ACTIONABLE = 0.6;
export function verdictTruth(group) {
  const mean = group.reduce((s, d) => s + d.truth.s, 0) / group.length;
  const hasActionable = group.some((d) => d.truth.actionable);
  if (mean <= SENTIMENT_NEG && hasActionable) return "act_now";
  if (hasActionable || mean <= SENTIMENT_NEG) return "track";
  return "ignore";
}

export function accuracy(pred, truth) {
  return pred.filter((p, i) => p === truth[i]).length / truth.length;
}
export function mae(pred, truth) {
  return pred.reduce((s, p, i) => s + Math.abs(p - truth[i]), 0) / pred.length;
}
export function brier(probs, truth) {
  return probs.reduce((s, p, i) => s + (p - (truth[i] ? 1 : 0)) ** 2, 0) / probs.length;
}
export function auc(probs, truth) {
  const pos = probs.filter((_, i) => truth[i]), neg = probs.filter((_, i) => !truth[i]);
  if (!pos.length || !neg.length) return NaN;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}
export function precisionRecall(pred, truth, cls) {
  const tp = pred.filter((p, i) => p === cls && truth[i] === cls).length;
  const fp = pred.filter((p) => p === cls).length - tp;
  const fn = truth.filter((t) => t === cls).length - tp;
  return { precision: tp + fp ? tp / (tp + fp) : NaN, recall: tp + fn ? tp / (tp + fn) : NaN, tp, fp, fn };
}

// Simple concurrency limiter to stay polite with the API.
export function limiter(max) {
  let active = 0; const q = [];
  const next = () => { active--; q.length && q.shift()(); };
  return (fn) => new Promise((res, rej) => {
    const run = () => fn().then(res, rej).finally(next);
    active < max ? (active++, run()) : q.push(run);
  });
}
