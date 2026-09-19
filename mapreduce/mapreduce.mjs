// App 4 — Semantic MapReduce.
// Replaces programmer-written map/reduce functions with Jev judgments:
//   MAP    -> per document, Jev emits a key (topic) + typed values (sentiment,
//             urgency, actionable). One systemOne request per doc.
//   SHUFFLE-> plain code groups values by key.
//   REDUCE -> ONE systemOne request over ALL keys answers per-group questions
//             in parallel; thresholds in this file own every decision.
// Questions and thresholds are named constants here for review/tuning.
import { choice, score, noul } from "@typesafe-ai/sdk";
import { client } from "../samples/shared/client.mjs";

// --- Thresholds (code owns every decision) ---
const SENTIMENT_NEG = 1.5; // score 0-3: mean sentiment below this = negative topic
const ACTIONABLE = 0.6; // noul: group with any doc above this is actionable
const ACT_NOW = 0.75; // noul for reduce verdict: act on this topic now

// --- MAP questions: same shape as emitting (k, v) pairs ---
const mapQuestions = {
  topic: choice("Which topic does this feedback concern?", {
    pricing: "Cost, plans, billing, value for money.",
    performance: "Speed, latency, responsiveness, throughput.",
    support: "Help, documentation, responsiveness of the team.",
    reliability: "Bugs, outages, data loss, stability.",
    other: null,
  }),
  sentiment: score("How does the customer feel about the product here?", [
    "Angry: actively churning or demanding refunds.",
    "Negative: frustrated, would not recommend.",
    "Neutral: mixed or purely factual.",
    "Positive: satisfied or enthusiastic.",
  ]),
  urgency: score("How urgently does this need the team's attention?", [
    "None: informational only.",
    "Low: can wait a sprint.",
    "High: this week.",
    "Critical: losing money or users right now.",
  ]),
  actionable: noul("Does this feedback contain a specific, concrete request or complaint the team could act on?"),
};

// --- MAP: one request per document ---
export async function mapDoc(doc, i) {
  const r = await client.systemOne({ state: { text: doc }, questions: mapQuestions });
  const a = r.answers;
  return { key: a.topic.choice, doc: i, sentiment: a.sentiment.score, urgency: a.urgency.score, actionable: a.actionable.noul };
}

// --- REDUCE questions: one per group, all answered in ONE request ---
// reduceQuestions(groupKey, values) builds the named questions; state carries
// the aggregated group so the model judges the group, not raw docs.
const TOPIC_ACTIONS = {
  act_now: "Escalate this week: assign an owner and ship a fix or response.",
  track: "Log and monitor; no immediate action.",
  ignore: "Not worth tracking; low value or out of scope.",
};

function reduceState(groups) {
  return Object.fromEntries(
    Object.entries(groups).map(([topic, vals]) => [
      topic,
      {
        doc_count: vals.length,
        mean_sentiment: +(vals.reduce((s, v) => s + v.sentiment, 0) / vals.length).toFixed(2),
        max_urgency: Math.max(...vals.map((v) => v.urgency)),
        actionable_docs: vals.filter((v) => v.actionable >= ACTIONABLE).length,
        sample: vals.slice(0, 3).map((v) => v.key),
      },
    ]),
  );
}

function reduceQuestions(groups) {
  const q = {};
  for (const topic of Object.keys(groups)) {
    q[`verdict_${topic}`] = choice(
      `Given the aggregated signals for the '${topic}' topic, what should the team do?`,
      TOPIC_ACTIONS,
    );
  }
  return q;
}

// --- Semantic MapReduce entry point ---
export async function semanticMapReduce(docs) {
  const mapped = await Promise.all(docs.map(mapDoc));

  // SHUFFLE
  const groups = {};
  for (const m of mapped) (groups[m.key] ??= []).push(m);

  // REDUCE: single batched request across all groups
  const rr = await client.systemOne({ state: { groups: reduceState(groups) }, questions: reduceQuestions(groups) });
  const reduced = {};
  for (const topic of Object.keys(groups)) {
    const g = groups[topic];
    const mean = g.reduce((s, v) => s + v.sentiment, 0) / g.length;
    const verdict = rr.answers[`verdict_${topic}`].choice;
    reduced[topic] = {
      docs: g.length,
      mean_sentiment: +mean.toFixed(2),
      negative: mean <= SENTIMENT_NEG,
      actionable: g.some((v) => v.actionable >= ACTIONABLE),
      verdict,
      escalate: verdict === "act_now",
    };
  }
  return { mapped, reduced };
}

// --- Demo ---
if (process.argv[1]?.endsWith("mapreduce.mjs")) {
  const docs = [
    "Pricing doubled this year for the same plan. We're evaluating competitors at renewal.",
    "The dashboard takes 12 seconds to load on our biggest account. Sales demos are embarrassing.",
    "Support answered in minutes and fixed our webhook issue the same day. Genuinely great.",
    "We lost a day of data during Tuesday's outage and never got a postmortem.",
    "Fine overall, though the docs for the API are three versions out of date.",
    "Charged twice for October, and the refund flow failed twice as well.",
  ];
  const out = await semanticMapReduce(docs);
  console.log("MAP (key + emitted values per doc):");
  for (const m of out.mapped) console.log(`  doc${m.doc} -> ${m.key}  sentiment=${m.sentiment.toFixed(2)} urgency=${m.urgency.toFixed(2)} actionable=${m.actionable.toFixed(2)}`);
  console.log("\nREDUCE (per key):");
  console.log(JSON.stringify(out.reduced, null, 2));
}
