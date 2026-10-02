// Traditional-ML baselines implemented in pure JS so inference timing is
// apples-to-apples with the Jev run in the same process:
//   - keyword rules (no learning)
//   - multinomial Naive Bayes on bag-of-words
//   - multinomial logistic regression on L2-normalized TF-IDF vectors (full-batch GD)
// Also an urgency regressor variant of each (treats levels 0-3 as classes /
// regression target respectively).

const tokenize = (t) =>
  t.toLowerCase().replace(/[^a-z0-9$% ]/g, " ").split(/\s+/).filter((w) => w.length > 1);

// --- keyword rules (mirrors the mapreduce baseline style) ---
const RULES = [
  ["billing", /charg|refund|invoice|bill|payment|price|pricing|subscription|renew/i],
  ["technical", /crash|slow|lag|latency|outage|bug|error|hang|fail|down|timeout/i],
  ["sales", /quote|trial|upgrade|discount|license|tier|plan|procurement|annual/i],
  ["account", /login|log in|password|sso|export|permission|delete|workspace|org/i],
];
export const keywordRules = (text) => RULES.find(([, re]) => re.test(text))?.[0] ?? "other";
export const keywordUrgency = (text) => {
  if (/right now|immediately|as we speak|down|bleeding|every minute|urgent/i.test(text)) return 3;
  if (/today|blocking|prioritize|escalat/i.test(text)) return 2;
  if (/this week|soon|nice to have/i.test(text)) return 1;
  return 0;
};

// --- shared vectorizer ---
function buildVocab(docs, minCount = 1) {
  const counts = new Map();
  for (const d of docs) for (const w of tokenize(d)) counts.set(w, (counts.get(w) ?? 0) + 1);
  return [...counts.entries()].filter(([, c]) => c >= minCount).map(([w]) => w);
}

function tfidfVectors(vocab, docs) {
  const index = new Map(vocab.map((w, i) => [w, i]));
  const df = new Array(vocab.length).fill(0);
  const tf = docs.map((d) => {
    const v = new Array(vocab.length).fill(0);
    for (const w of tokenize(d)) { const i = index.get(w); if (i !== undefined) v[i]++; }
    for (let i = 0; i < v.length; i++) if (v[i] > 0) df[i]++;
    return v;
  });
  const idf = df.map((d) => Math.log((1 + docs.length) / (1 + d)) + 1);
  return tf.map((v) => {
    const x = v.map((c, i) => (c > 0 ? (1 + Math.log(c)) * idf[i] : 0));
    const norm = Math.hypot(...x) || 1;
    return x.map((v2) => v2 / norm);
  });
}

// --- multinomial Naive Bayes (classes can be labels or integers) ---
export function trainNaiveBayes(docs, labels) {
  const vocab = buildVocab(docs);
  const classes = [...new Set(labels)];
  const count = new Map(classes.map((c) => [c, new Array(vocab.length).fill(0)]));
  const classTotal = new Map(classes.map((c) => [c, 0]));
  const classDoc = new Map(classes.map((c) => [c, 0]));
  docs.forEach((d, i) => {
    const c = labels[i];
    classDoc.set(c, classDoc.get(c) + 1);
    for (const w of tokenize(d)) {
      const j = vocab.indexOf(w); // vocab small here; fine for bench sizes
      if (j >= 0) { count.get(c)[j]++; classTotal.set(c, classTotal.get(c) + 1); }
    }
  });
  const vocabLen = vocab.length;
  const logPrior = new Map(classes.map((c) => [c, Math.log(classDoc.get(c) / docs.length)]));
  const logLike = new Map(
    classes.map((c) => {
      const tot = classTotal.get(c) + vocabLen;
      return [c, count.get(c).map((n) => Math.log((n + 1) / tot))];
    }),
  );
  return {
    kind: "nb",
    predict(text) {
      let best, bestScore = -Infinity;
      for (const c of classes) {
        let s = logPrior.get(c);
        for (const w of tokenize(text)) {
          const j = vocab.indexOf(w);
          if (j >= 0) s += logLike.get(c)[j];
        }
        if (s > bestScore) { bestScore = s; best = c; }
      }
      return best;
    },
  };
}

// --- softmax regression on TF-IDF (full-batch gradient descent, L2) ---
export function trainLogReg(docs, labels, { epochs = 300, lr = 0.5, l2 = 1e-4 } = {}) {
  const vocab = buildVocab(docs);
  const X = tfidfVectors(vocab, docs);
  const classes = [...new Set(labels)];
  const k = classes.length, d = vocab.length;
  const W = Array.from({ length: k }, () => new Array(d + 1).fill(0)); // +1 bias
  const Y = labels.map((l) => classes.indexOf(l));

  for (let ep = 0; ep < epochs; ep++) {
    const gW = Array.from({ length: k }, () => new Array(d + 1).fill(0));
    for (let i = 0; i < X.length; i++) {
      const scores = W.map((w) => {
        let s = w[d];
        for (let j = 0; j < d; j++) s += w[j] * X[i][j];
        return s;
      });
      const max = Math.max(...scores);
      const exp = scores.map((s) => Math.exp(s - max));
      const z = exp.reduce((a, b) => a + b, 0);
      for (let c = 0; c < k; c++) {
        const p = exp[c] / z;
        const err = p - (c === Y[i] ? 1 : 0);
        gW[c][d] += err;
        for (let j = 0; j < d; j++) if (X[i][j] !== 0) gW[c][j] += err * X[i][j];
      }
    }
    const step = lr / X.length;
    for (let c = 0; c < k; c++)
      for (let j = 0; j <= d; j++) W[c][j] -= step * (gW[c][j] + (j < d ? l2 * W[c][j] * X.length : 0));
  }

  const dot = (w, x) => {
    let s = w[d];
    for (let j = 0; j < d; j++) if (x[j] !== 0) s += w[j] * x[j];
    return s;
  };

  return {
    kind: "logreg",
    predict(text) {
      const [x] = tfidfVectors(vocab, [text]);
      let best = 0, bestScore = -Infinity;
      for (let c = 0; c < k; c++) {
        const s = dot(W[c], x);
        if (s > bestScore) { bestScore = s; best = c; }
      }
      return classes[best];
    },
    // Softmax over class scores: {probs: Map(class -> p), argmax, max}
    predictProba(text) {
      const [x] = tfidfVectors(vocab, [text]);
      const scores = W.map((w) => dot(w, x));
      const max = Math.max(...scores);
      const exp = scores.map((s) => Math.exp(s - max));
      const z = exp.reduce((a, b) => a + b, 0);
      const probs = new Map(classes.map((c, i) => [c, exp[i] / z]));
      const argmax = classes[scores.indexOf(max)];
      return { probs, argmax, max: probs.get(argmax) };
    },
  };
}
