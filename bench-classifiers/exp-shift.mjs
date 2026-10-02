// Experiment 1 — distribution shift: train on the original support-ticket
// phrasing, test on a paraphrased twin (same categories, same label logic,
// entirely different surface vocabulary; cue pools avoid the original urgency
// keywords). In-distribution control uses the SAME original generator with a
// different seed, so deltas isolate the phrasing shift from seed noise.
import {
  DATASETS, makeTicketGen, TICKET_BODIES, URGENCY_CUES_EXPLICIT,
} from "./datasets.mjs";
import { trainNaiveBayes, trainLogReg } from "./ml-baselines.mjs";
import { jevClassifyAll } from "./jev.mjs";
import { clefClassifyAll } from "./clef.mjs";
import fs from "node:fs";

// Paraphrased ticket bodies: same five categories, different vocabulary.
const ALT_BODIES = {
  billing: [
    "We got billed two times this period and the credit never posted.",
    "Your statement shows equipment we returned months ago.",
    "The renewal quote jumped 40% overnight.",
    "Our card on file was declined even though it works everywhere else.",
    "Still waiting on the credit from the double withdrawal.",
    "Finance flagged discrepancies between the contract and the statements.",
  ],
  technical: [
    "The editor freezes whenever we open big repositories.",
    "Pushes fail with a timeout roughly half the time.",
    "The console renders blank on our largest tenant.",
    "Mobile keeps dropping sync halfway through.",
    "Background jobs stall at random and pile up in the queue.",
    "The client SDK omits pagination that the reference claims to support.",
  ],
  sales: [
    "We are comparing platforms before our contract expires — what is your best rate for {n2} hundred users?",
    "Could you send a proposal including identity features and bulk seats?",
    "Our pilot finishes {when}; we would like to roll out to every department.",
    "Is there a charity rate, and what does it include?",
    "Legal needs a signed order form for fifty seats before sign-off.",
    "What separates the flagship package from the standard one?",
  ],
  account: [
    "Nobody on our team can get in and the recovery mail never shows up.",
    "We must pull everything out of the tenant before it lapses.",
    "Ex-staff still have owner rights; strip their access please.",
    "Federated sign-in broke after we swapped identity vendors.",
    "Combine the two tenants that were created by mistake.",
    "Erase the profile of someone who resigned.",
  ],
  other: [
    "It would be neat to have spreadsheet-style filters.",
    "Kudos to whoever shipped the bulk editor.",
    "Please consider adding an audit trail for guest access.",
    "The welcome sequence was a pleasant surprise.",
    "Do you have a user group or forum where admins hang out?",
    "Small gripe: the favicon still shows the old logo.",
  ],
};
const ALT_CUES = {
  0: ["There is no urgency here.", "This can sit for a while.", "Zero rush.", "Just documenting this."],
  1: ["Sometime in the next week or two would work.", "Not a big deal, but sooner is better than later.", "When someone has a spare moment this week.", "No fire, just do not let it sit a month."],
  2: ["We could use an answer by end of day.", "This is holding up a customer commitment.", "It is becoming a talking point in our standups.", "Would really appreciate movement on this today."],
  3: ["Every minute offline is costing us contracts.", "We have live customers affected as we speak.", "This is costing real money by the hour.", "The situation is deteriorating right now and needs humans on it."],
};

const shiftGen = makeTicketGen(ALT_CUES, ALT_BODIES);
const N_TEST = 48;

const train = DATASETS["support-tickets"].gen({ nTrain: 50, nTest: 0, seed: 42 }).train;
const testIn = makeTicketGen(URGENCY_CUES_EXPLICIT, TICKET_BODIES)({ nTrain: 0, nTest: N_TEST, seed: 99 }).test;
const testShift = shiftGen({ nTrain: 0, nTest: N_TEST, seed: 99 }).test;

const acc = (pred, gold) => pred.filter((p, i) => p === gold[i]).length / pred.length;

function evalOn(test, predictC, predictS) {
  const m = {
    choiceAcc: acc(test.map(predictC), test.map((t) => t.label)),
  };
  if (predictS) m.scoreAcc = acc(test.map(predictS).map(String), test.map((t) => String(t.score)));
  return m;
}

// Train once on original phrasing, n=50.
const nbC = trainNaiveBayes(train.map((t) => t.text), train.map((t) => t.label));
const nbS = trainNaiveBayes(train.map((t) => t.text), train.map((t) => String(t.score)));
const lrC = trainLogReg(train.map((t) => t.text), train.map((t) => t.label));
const lrS = trainLogReg(train.map((t) => t.text), train.map((t) => String(t.score)));
const kw = DATASETS["support-tickets"].keywords;

// Jev + Clef on both test sets (same questions, zero-shot).
const testTexts = (test) => test.map((t) => t.text);
const [jevIn, jevShift, clefIn, clefShift, flashIn, flashShift] = await Promise.all([
  jevClassifyAll(testTexts(testIn), "support-tickets"),
  jevClassifyAll(testTexts(testShift), "support-tickets"),
  clefClassifyAll(testTexts(testIn), "support-tickets", { model: "clef" }),
  clefClassifyAll(testTexts(testShift), "support-tickets", { model: "clef" }),
  clefClassifyAll(testTexts(testIn), "support-tickets", { model: "clef-flash" }),
  clefClassifyAll(testTexts(testShift), "support-tickets", { model: "clef-flash" }),
]);

const systems = {
  "keyword-rules": (test) => evalOn(test, (t) => kw.choice(t.text), (t) => kw.score(t.text)),
  "naive-bayes(n=50)": (test) => evalOn(test, (t) => nbC.predict(t.text), (t) => String(nbS.predict(t.text))),
  "logreg-tfidf(n=50)": (test) => evalOn(test, (t) => lrC.predict(t.text), (t) => String(lrS.predict(t.text))),
  "jev-zero-shot": (test, out) => evalOn(test, (t, i) => out[i].choice, (t, i) => String(out[i].score)),
  "clef": (test, out) => evalOn(test, (t, i) => out[i].choice, (t, i) => String(out[i].score)),
  "clef-flash": (test, out) => evalOn(test, (t, i) => out[i].choice, (t, i) => String(out[i].score)),
};

const rows = [];
const PAIRS = {
  "keyword-rules": [null, null],
  "naive-bayes(n=50)": [null, null],
  "logreg-tfidf(n=50)": [null, null],
  "jev-zero-shot": [jevIn, jevShift],
  "clef": [clefIn, clefShift],
  "clef-flash": [flashIn, flashShift],
};
for (const [name, fn] of Object.entries(systems)) {
  const [inOut, shiftOut] = PAIRS[name];
  const inDist = fn(testIn, inOut);
  const shifted = fn(testShift, shiftOut);
  rows.push({
    name,
    inDistChoice: inDist.choiceAcc, shiftChoice: shifted.choiceAcc,
    deltaChoice: shifted.choiceAcc - inDist.choiceAcc,
    inDistScore: inDist.scoreAcc, shiftScore: shifted.scoreAcc,
    deltaScore: shifted.scoreAcc - inDist.scoreAcc,
  });
}

fs.writeFileSync(new URL("./exp-shift-results.json", import.meta.url), JSON.stringify({ nTest: N_TEST, rows }, null, 2));

console.log(`train: original phrasing (n=50, seed 42) | test: ${N_TEST} items, seed 99, in-dist vs paraphrased twin`);
console.log("system               choice in→shift   Δ       score in→shift   Δ");
for (const r of rows) {
  const pct = (x) => (x * 100).toFixed(1).padStart(5) + "%";
  console.log(
    r.name.padEnd(20),
    `${pct(r.inDistChoice)}→${pct(r.shiftChoice)}`, (r.deltaChoice * 100).toFixed(1).padStart(6),
    r.inDistScore !== undefined ? `${pct(r.inDistScore)}→${pct(r.shiftScore)}` : "",
    r.deltaScore !== undefined ? (r.deltaScore * 100).toFixed(1).padStart(6) : "",
  );
}
