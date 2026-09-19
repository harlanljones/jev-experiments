// Corpus generator — writes fixtures/corpus.json. Seeded; rerun with a new
// SEED only to regenerate a fresh fixture, never between benchmark runs.
// Labels are planted by construction: every doc carries known topic,
// sentiment level (0=angry..3=positive) and actionable flag.
// Tiers: T1 clear, T2 paraphrase (surface variation, same truth),
//        T3 adversarial (mixed topic, negation flips, no-signal).
import fs from "node:fs";

const SEED = 20260918;
const rng = (() => { let s = SEED >>> 0; return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
const pick = (a) => a[Math.floor(rng() * a.length)];
const shuffle = (a) => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

const PRODUCTS = ["the workspace", "the dashboard", "the sync client", "the export tool", "the mobile app"];
const AMOUNTS = ["40%", "half", "double", "a third"];

// Truth combos: 5 topics x {sentiment, actionable} profiles.
// S = sentiment level index (0 angry, 1 negative, 2 neutral, 3 positive).
const PROFILES = {
  pricing_neg: { topic: "pricing", s: 1, actionable: true },
  pricing_angry: { topic: "pricing", s: 0, actionable: true },
  perf_neg: { topic: "performance", s: 1, actionable: true },
  perf_neutral: { topic: "performance", s: 2, actionable: false },
  support_pos: { topic: "support", s: 3, actionable: false },
  support_neg: { topic: "support", s: 1, actionable: true },
  rel_neg: { topic: "reliability", s: 0, actionable: true },
  rel_mixed: { topic: "reliability", s: 2, actionable: true },
  other_neutral: { topic: "other", s: 2, actionable: false },
  other_pos: { topic: "other", s: 3, actionable: false },
};
const COMBOS = Object.entries(PROFILES); // 10 combos, 3 docs each per tier = 30

// --- T1: one literal template per doc ---
const T1 = {
  pricing_neg: ["The plan price went up {amount} this year and the invoice line items don't add up. Please fix the billing.", "We're paying {amount} more for the same tier. The new pricing is not reasonable and we need it reviewed.", "Price increased {amount} at renewal without notice. The billing team should explain this."],
  pricing_angry: ["We were charged twice for October and the refund still hasn't arrived. This is unacceptable.", "Pricing doubled with zero notice and I demand a refund for the difference. Fix this today.", "You billed our card {amount} more than agreed. We are evaluating competitors at renewal if this isn't refunded."],
  perf_neg: ["{product} takes 12 seconds to load on our biggest account. This lag is hurting our demos.", "Search on {product} is noticeably slow and requests lag for seconds at a time. Needs work.", "{product} became sluggish after the last release; large files crawl to open."],
  perf_neutral: ["Load times on {product} seem unchanged since last quarter, roughly the same speed.", "{product} performance is about average for this class of tool, nothing notable.", "The sync speed on {product} is comparable to what we had before the migration."],
  support_pos: ["Support answered in minutes and fixed our webhook issue the same day. Genuinely great.", "The docs team walked us through the migration personally. Excellent experience.", "Chats with support were helpful, quick, and friendly throughout."],
  support_neg: ["We opened three tickets and the support team never acknowledged the bug for a week.", "The API documentation is three versions out of date and support gave no timeline to fix it.", "Support responses take days and the help articles don't cover our error."],
  rel_neg: ["{product} crashed on sync twice today and we lost unsaved work. Unacceptable.", "An outage on Tuesday cost us a day of data and there was no postmortem. This cannot happen again.", "{product} corrupted our export twice this month. We need a fix immediately."],
  rel_mixed: ["Backup reliability worries me after last week's incident, though restores have worked since. Can you confirm the failover procedure?", "Stability has been mostly fine lately but the one data loss incident in March still needs a written explanation.", "No crashes in weeks now, but I'd like a documented guarantee about data retention before we renew."],
  other_neutral: ["The changelog format changed recently and I'm still finding my way around it.", "Would love a dark mode toggle in a future release, whenever it's convenient.", "The onboarding checklist is a nice touch, just noting it exists."],
  other_pos: ["The keyboard shortcuts alone are worth the subscription. Lovely product.", "Autosave works well now and the new UI is a pleasure to use.", "Setup took five minutes and everything just worked. Very impressed."],
};

// --- T2: paraphrase templates (different surface form, same planted truth) ---
// Slot banks vary phrasing without touching the labels.
const T2 = {
  pricing_neg: ["Renewal came in {amount} above what we budgeted. {question_billing}", "The subscription cost jumped {amount} and nobody flagged it. {question_billing}", "{amount} increase on the same plan this cycle. {question_billing}"],
  pricing_angry: ["{double_charge} Fix this immediately or {threat}.", "{double_charge} {threat}.", "{double_charge} We're done if {threat_short}."],
  perf_neg: ["Honestly, {product} feels {slow_word} since the update. {perf_ask}", "{slow_word_cap} is the only way to describe {product} on big accounts right now. {perf_ask}", "Every request on {product} {lags}. {perf_ask}"],
  perf_neutral: ["Speed-wise {product} is {same_word} to last quarter as far as we can tell.", "We haven't noticed much difference in {product} responsiveness either way.", "Performance on {product} is {same_word} to before, nothing to report."],
  support_pos: ["{support_praise} Great team.", "{support_praise} Made our week.", "{support_praise} Exactly how it should work."],
  support_neg: ["{support_gripe} A reply within days would be nice.", "{support_gripe} It's frustrating not to get an ETA.", "{support_gripe} Please at least acknowledge the ticket."],
  rel_neg: ["{product} {crash_word} again and we {lost_word}. Not okay.", "{crash_word_cap} on {product} during peak hours, {lost_word}. This must be fixed now.", "Twice this month {product} {crash_word} mid-session and {lost_word}. Absolutely unacceptable."],
  rel_mixed: ["Things have been steadier lately, {rel_ask}", "Mostly stable these days, {rel_ask}", "No complaints since the patch, {rel_ask}"],
  other_neutral: ["{wishlist_neutral}", "{wishlist_neutral2}", "Just noting {product} looks different after the redesign, no opinion yet."],
  other_pos: ["{small_praise} Really happy with it.", "{small_praise} Keep it up.", "{small_praise} Worth every cent."],
};
const BANKS = {
  question_billing: ["Can someone from billing walk me through it?", "Please review the invoice.", "I'd like an explanation."],
  double_charge: ["Our card got billed twice this month and the refund request went nowhere.", "We're seeing duplicate charges and an incorrect invoice on top.", "Billed twice, refund pending for weeks."],
  threat: ["we're moving to a competitor", "we're taking our budget elsewhere", "renewal is off the table"],
  threat_short: ["this isn't resolved by Friday", "the charges stand", "nobody calls us back"],
  slow_word: ["really sluggish", "noticeably slower", "painfully slow"],
  slow_word_cap: ["Slow", "Sluggish", "Laggy"],
  perf_ask: ["Can engineering look into it?", "Please profile the hot paths.", "We need this faster."],
  lags: ["take several seconds to complete", "time out more often than they should", "queue behind something for seconds"],
  same_word: ["roughly the same", "about on par", "comparable"],
  support_praise: ["Your team turned our ticket around in under an hour and stayed on it until it was fixed.", "Support stayed with us through the whole migration and resolved it same-day.", "Got a fast, competent answer from support within minutes."],
  support_gripe: ["Three tickets in and no acknowledgment after a week.", "The documentation doesn't match the current API and support has no answers.", "Support went quiet on our open issue for days."],
  crash_word: ["crashed", "went down", "fell over"],
  crash_word_cap: ["It crashed", "It went down", "The client fell over"],
  lost_word: ["we lost a chunk of unsaved work", "half an hour of edits vanished", "our in-progress export was gone"],
  rel_ask: ["but can you confirm the failover path in writing?", "though I'd still like that postmortem from March.", "yet a retention guarantee would help before renewal."],
  wishlist_neutral: ["A CSV export option would be handy at some point, whenever it fits the roadmap.", "Maybe consider keyboard navigation for the settings pane someday.", "It'd be nice if the changelog had an RSS feed eventually."],
  wishlist_neutral2: ["Dark mode would be welcome down the line, no rush.", "An API for the audit log would be useful eventually.", "Group mentions would be a nice-to-have someday."],
  small_praise: ["Setup was painless and everything works as promised.", "The shortcuts are a small delight and the product feels solid.", "Migrating took minutes and it has been smooth sailing."],
};
const fill = (t) => t.replace(/\{(\w+)\}/g, (_, k) => (BANKS[k] ? pick(BANKS[k]) : k === "product" ? pick(PRODUCTS) : k === "amount" ? pick(AMOUNTS) : k));
// T3 constructors reuse T1 texts, so apply the same slot substitution.

// --- T3 adversarial constructors ---
function mixedTopicDoc() {
  const pairs = shuffle(COMBOS.filter(([k]) => !k.startsWith("other")).slice(0, 4));
  const [ka, a] = pairs[0]; const [kb, b] = pairs[1];
  const aTxt = fill(pick(T1[ka])); const bTxt = fill(pick(T1[kb]));
  return { text: `${aTxt} Separately, ${bTxt.charAt(0).toLowerCase()}${bTxt.slice(1)}`, truth: a.s <= b.s ? a : b, note: `mixed:${ka}+${kb}` };
}
function negationDoc() {
  // Positive mention of one topic, planted truth is the OTHER topic negative.
  const pos = fill(pick(T1.support_pos)); // positive support mention
  const [k, prof] = pick(COMBOS.filter(([k]) => k.startsWith("rel_") || k.startsWith("perf_")));
  const neg = fill(pick(T1[k]));
  return { text: `${pos} That said, ${neg.charAt(0).toLowerCase()}${neg.slice(1)}`, truth: prof, note: `negation:${k}` };
}
function noSignalDoc() {
  const neutrals = [
    "What's the current status of the September release train?",
    "This is a copy of the notes from our internal sync, forwarded for reference.",
    "FYI we updated the contact person on the account to our new ops lead.",
    "Per the subject line, rescheduling our quarterly review to November.",
    "Ack, received the updated contract, no questions right now.",
  ];
  return { text: pick(neutrals), truth: { topic: "other", s: 2, actionable: false }, note: "no-signal" };
}

const PER_TIER = 30;
const docs = [];
for (const tier of ["T1", "T2", "T3"]) {
  const combos = shuffle(COMBOS);
  let made = 0, ci = 0;
  while (made < PER_TIER) {
    const [key, prof] = combos[ci % combos.length]; ci++;
    if (tier === "T1") {
      const text = fill(T1[key][made % T1[key].length]);
      docs.push({ id: `${tier}-${made}`, tier, text, truth: prof, note: key });
    } else if (tier === "T2") {
      const text = fill(pick(T2[key]));
      docs.push({ id: `${tier}-${made}`, tier, text, truth: prof, note: key });
    } else {
      const mode = made % 3;
      const d = mode === 0 ? mixedTopicDoc() : mode === 1 ? negationDoc() : noSignalDoc();
      docs.push({ id: `${tier}-${made}`, tier, text: d.text, truth: d.truth, note: d.note });
    }
    made++;
  }
}
fs.mkdirSync("fixtures", { recursive: true });
fs.writeFileSync("fixtures/corpus.json", JSON.stringify({ seed: SEED, generated: new Date().toISOString(), docs }, null, 1));
console.log(`froze ${docs.length} docs (${[...new Set(docs.map(d => d.tier))].join(",")}) to fixtures/corpus.json`);
