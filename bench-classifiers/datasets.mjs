// Dataset registry for the Jev-vs-traditional-ML benchmark. Every dataset is
// generated from a seeded RNG with labels known by construction; train/test
// items draw disjoint template/sentence picks so test items are not memorized
// copies of train items.
//
//   support-tickets    short tickets, 5-way category + urgency 0-3 with
//                      explicit intensity cues ("today", "right now")
//   tickets-hard       same categories, urgency expressed obliquely through
//                      stakes ("two teams paused work"), no urgency keywords
//   reviews            product reviews, 3-way sentiment + 0-4 rating
//   topics             multi-sentence notes, 6-way topic, imbalanced priors,
//                      confusable class pairs + distractor sentences
//
// Task config consumed by bench.mjs / jev.mjs: choice question, optional
// score question, class lists, and keyword-rule baselines per task.
import { choice as choiceDef, score as scoreDef } from "@typesafe-ai/sdk";

export const TEST_SIZE = 32;
export const TRAIN_SIZE = 50;
export const SEEDS = [42, 7, 123];

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];

function fill(t, rng, FILLERS) {
  return t.replace(/\{(\w+)\}/g, (_, k) => pick(rng, FILLERS[k]));
}

// ---------------------------------------------------------------- tickets --
export const TICKET_BODIES = {
  billing: [
    "We were charged {n} times on our last invoice and the refund flow failed.",
    "The invoice for {month} is wrong: it lists seats we removed in {month2}.",
    "Your price increase pushed our plan from ${a} to ${b} a seat with no notice.",
    "Payment keeps failing on a valid card, so our subscription may lapse.",
    "The refund for the duplicate charge still has not arrived after {n2} weeks.",
    "Billing emails land in spam and we missed the invoice due date.",
  ],
  technical: [
    "The dashboard takes {n2} seconds to load on our biggest workspace.",
    "The mobile app crashes on sync since the {month} update.",
    "Search requests lag for seconds on large projects and time out sometimes.",
    "Webhooks return {code} errors intermittently, breaking our integration.",
    "The exporter hangs at {pct}% and never finishes on large files.",
    "API responses are missing fields that the docs promise for the {lang} SDK.",
  ],
  sales: [
    "We are evaluating vendors for renewal: what can you do on a {n}-seat annual plan?",
    "Can we get an enterprise quote with SSO and a volume discount?",
    "Our trial ends {when} and we want to move {n2} teams onto the paid tier.",
    "Do you offer nonprofit pricing, and how does it compare to the business plan?",
    "We need a purchase order for {n} licenses before procurement will proceed.",
    "What is included in the top tier that the mid tier lacks?",
  ],
  account: [
    "Our whole team got logged out and password reset emails never arrive.",
    "We need to export all workspace data before our contract ends.",
    "Two former employees still have admin access; please revoke their logins.",
    "SSO stopped working for {n2} people after we changed our identity provider.",
    "Please merge our duplicate workspaces under one organization.",
    "Delete the personal data of a user who left the company.",
  ],
  other: [
    "Would love a dark mode and keyboard shortcuts for the annotation tool.",
    "The release notes are helpful; keep up the good work.",
    "Feature request: bulk tagging for saved views would save us hours.",
    "Your onboarding emails were genuinely useful, thanks to the team.",
    "Any plans to open an office or community meetup in {city}?",
    "Love the product overall, though the changelog formatting is odd.",
  ],
};

export const TICKET_FILLERS = {
  n: ["two", "three", "four", "five"],
  n2: ["2", "3", "4", "6", "8"],
  month: ["October", "November", "September", "August"],
  month2: ["July", "June", "August"],
  a: ["12", "15", "18"],
  b: ["24", "30", "36"],
  code: ["502", "500", "429"],
  pct: ["40", "60", "80"],
  lang: ["Python", "Node", "Go"],
  when: ["this Friday", "next week", "in ten days"],
  city: ["Austin", "Berlin", "Toronto"],
};

// Explicit intensity cues (tickets) vs oblique stakes (tickets-hard). Ground
// truth urgency is picked first; the cue pool consistent with it decorates the
// text. The hard pool deliberately avoids urgency adverbs: urgency must be
// inferred from the situation described, not matched from a keyword list.
export const URGENCY_CUES_EXPLICIT = {
  0: ["No rush on this.", "Whenever you get a chance is fine.", "Minor thing, ignore if busy.", "Not urgent at all."],
  1: ["Would be nice to have this week.", "Please look when convenient this week.", "No emergency, but soon-ish.", "This week would be good."],
  2: ["We need a response today.", "Please prioritize, this is blocking a launch.", "Same-day reply appreciated, this is escalating.", "Leadership is asking about this today."],
  3: ["This is an active outage and money is being lost right now.", "Production is down, every minute costs us deals.", "All hands on deck, customers are churning as we speak.", "We are bleeding revenue this very minute, respond immediately."],
};
const URGENCY_CUES_OBLIQUE = {
  0: ["Filing this so it is on the record for our documentation.", "This is for the notes we keep for the next planning cycle.", "Adding it to the backlog list we review monthly."],
  1: ["It would help to have this before our next quarterly planning session.", "Whenever the roadmap allows.", "Before the next release cycle would be handy.", "Our team sync is in two weeks, before then is fine."],
  2: ["Two teams have paused their work while this is unresolved.", "Our launch committee meets Thursday and this affects the agenda.", "We have had to reassign two engineers to work around it.", "A signed customer is waiting on this before their rollout."],
  3: ["Our largest customer has paused their renewal pending a fix here.", "The exec team has made this their single top ask this week.", "We have stopped all onboarding until this is resolved.", "Board reporting on Friday depends on these numbers being right."],
};
export const CHURN_CUES = [
  "If this is not fixed we are moving to a competitor.",
  "Honestly considering canceling over this.",
  "This is the last straw before we look at other vendors.",
  "Our renewal decision now depends on how this is handled.",
];
export const NEUTRAL_TAILS = ["Thanks.", "Appreciate the help.", "See attached logs.", "Ticket from our admin console."];

export function makeTicketGen(cuePool, bodies = TICKET_BODIES) {
  return function gen({ nTrain, nTest, seed }) {
    const rng = mulberry32(seed);
    const used = new Set();
    const tickets = [];
    function make(split) {
      for (let attempt = 0; attempt < 50; attempt++) {
        const label = pick(rng, Object.keys(TICKET_BODIES));
        const body = pick(rng, bodies[label]);
        const r = rng();
        let score;
        if (label === "technical" || label === "billing") score = r < 0.15 ? 0 : r < 0.4 ? 1 : r < 0.75 ? 2 : 3;
        else score = r < 0.4 ? 0 : r < 0.75 ? 1 : r < 0.95 ? 2 : 3;
        const churn = rng() < (score >= 2 ? 0.55 : 0.2);
        const cue = pick(rng, cuePool[score]);
        const tail = churn ? pick(rng, CHURN_CUES) : pick(rng, NEUTRAL_TAILS);
        const text = rng() < 0.5 ? `${fill(body, rng, TICKET_FILLERS)} ${cue} ${tail}` : `${cue} ${fill(body, rng, TICKET_FILLERS)} ${tail}`;
        if (used.has(text)) continue;
        used.add(text);
        tickets.push({ id: `${split}-${tickets.length}`, split, text, label, score });
        return;
      }
      throw new Error("could not generate unique ticket");
    }
    while (tickets.length < nTrain) make("train");
    while (tickets.length < nTrain + nTest) make("test");
    return { train: tickets.filter((t) => t.split === "train"), test: tickets.filter((t) => t.split === "test") };
  };
}

const TICKET_CHOICE = {
  question: () => ({
    category: choiceDef("What is this support ticket about?", {
      billing: "Charges, refunds, invoices, payment failures, pricing complaints.",
      technical: "Bugs, errors, outages, slowness, things not working.",
      sales: "Purchasing, upgrades, quotes, trials, licensing questions.",
      account: "Logins, permissions, SSO, data export, deletion, workspaces.",
      other: "Feature requests, feedback, praise, or anything else.",
    }),
  }),
  classes: ["billing", "technical", "sales", "account", "other"],
  scoreQuestion: () => ({
    urgency: scoreDef("How urgently does this ticket need a human response?", [
      "None: can wait days, purely informational.",
      "Low: this week is fine, mild inconvenience.",
      "High: needs a response today, blocking work or escalating.",
      "Critical: active outage or money being lost right now.",
    ]),
  }),
  scoreLevels: [0, 1, 2, 3],
};

const TICKET_KEYWORDS = {
  choice: (text) => {
    const RULES = [
      ["billing", /charg|refund|invoice|bill|payment|price|pricing|subscription|renew/i],
      ["technical", /crash|slow|lag|latency|outage|bug|error|hang|fail|down|timeout/i],
      ["sales", /quote|trial|upgrade|discount|license|tier|plan|procurement|annual/i],
      ["account", /login|log in|password|sso|export|permission|delete|workspace|org/i],
    ];
    return RULES.find(([, re]) => re.test(text))?.[0] ?? "other";
  },
  score: (text) => {
    if (/right now|immediately|as we speak|down|bleeding|every minute|urgent/i.test(text)) return 3;
    if (/today|blocking|prioritize|escalat|paused|stopped|pending/i.test(text)) return 2;
    if (/this week|soon|nice to have|before our|quarterly/i.test(text)) return 1;
    return 0;
  },
};

// --------------------------------------------------------------- reviews --
const POS = [
  "Absolutely love the new dashboard, it saves me hours every week.",
  "Setup took five minutes and everything just worked.",
  "The team behind this is responsive and clearly cares.",
  "Best purchase we made this year, our workflow is noticeably faster.",
  "The redesign is beautiful and intuitive.",
  "Support went above and beyond when we hit a snag.",
];
const NEG = [
  "It crashes constantly and I have lost work twice this week.",
  "Support has ignored three tickets in a row.",
  "The price went up while features were removed.",
  "Constant sync failures make this unusable for our team.",
  "The export feature mangles our data every single time.",
  "Onboarding was confusing and the docs are outdated.",
];
const NEU = [
  "We use the Pro plan with about 40 seats.",
  "Installed the update last Tuesday.",
  "Does this integrate with LDAP?",
  "Our team is spread across three time zones.",
  "We moved from a self-hosted setup in March.",
  "The workspace has six projects and two admins.",
];

function makeReviewGen() {
  return function gen({ nTrain, nTest, seed }) {
    const rng = mulberry32(seed);
    const used = new Set();
    const reviews = [];
    function make(split) {
      for (let attempt = 0; attempt < 50; attempt++) {
        const r = rng();
        const label = r < 0.4 ? "positive" : r < 0.7 ? "negative" : "mixed";
        const parts = [];
        const take = (pool, k) => { for (let i = 0; i < k; i++) parts.push(pick(rng, pool)); };
        if (label === "positive") take(POS, rng() < 0.5 ? 1 : 2);
        else if (label === "negative") take(NEG, rng() < 0.5 ? 1 : 2);
        else { take(POS, rng() < 0.5 ? 1 : 2); take(NEG, rng() < 0.5 ? 1 : 2); }
        if (rng() < 0.5) parts.splice(Math.floor(rng() * parts.length), 0, pick(rng, NEU));
        const text = parts.join(" ");
        if (used.has(text)) continue;
        used.add(text);
        const score =
          label === "positive" ? (rng() < 0.8 ? 4 : 3)
          : label === "negative" ? (rng() < 0.7 ? 0 : 1)
          : (rng() < 0.75 ? 2 : rng() < 0.5 ? 1 : 3);
        reviews.push({ id: `${split}-${reviews.length}`, split, text, label, score });
        return;
      }
      throw new Error("could not generate unique review");
    }
    while (reviews.length < nTrain) make("train");
    while (reviews.length < nTrain + nTest) make("test");
    return { train: reviews.filter((t) => t.split === "train"), test: reviews.filter((t) => t.split === "test") };
  };
}

const REVIEW_CHOICE = {
  question: () => ({
    sentiment: choiceDef("What is the overall sentiment of this product review?", {
      positive: "Praise, satisfaction, or recommendation.",
      negative: "Complaints, frustration, or a bad experience.",
      mixed: "Both meaningful praise and meaningful complaints.",
    }),
  }),
  classes: ["positive", "negative", "mixed"],
  scoreQuestion: () => ({
    rating: scoreDef("How satisfied is the reviewer overall, on a 0-4 scale?", [
      "0: Terrible, actively harmful to their work.",
      "1: Bad, major frustrations dominate.",
      "2: Mixed, real pros and real cons.",
      "3: Good, minor complaints only.",
      "4: Delighted, enthusiastic recommendation.",
    ]),
  }),
  scoreLevels: [0, 1, 2, 3, 4],
};

const REVIEW_KEYWORDS = {
  choice: (text) => {
    const pos = /love|great|best|beautiful|responsive|saves|just worked|above and beyond|faster/i.test(text);
    const neg = /crash|ignored|price went up|unusable|mangles|confusing|lost work|outdated|frustrat/i.test(text);
    if (pos && neg) return "mixed";
    return pos ? "positive" : neg ? "negative" : "mixed";
  },
  score: () => 2,
};

// ---------------------------------------------------------------- topics --
const TOPIC_POOLS = {
  hr: [
    "We need to onboard three new hires to the design tool.",
    "The benefits enrollment window closes at the end of the month.",
    "Two contractors are converting to full-time next quarter.",
    "The annual review cycle starts in November and managers need templates.",
  ],
  security: [
    "We flagged an unusual login pattern from an unrecognized device.",
    "The pen test found a stored-XSS issue in the comments module.",
    "Rotate the service credentials that were shared in the incident channel.",
    "Access reviews are due before the audit window opens.",
  ],
  infrastructure: [
    "The staging cluster is running out of disk on the database nodes.",
    "We are moving the batch jobs to a new region for latency.",
    "Terraform plans now take 20 minutes to apply.",
    "The CDN cache hit rate dropped after last week's config change.",
  ],
  product: [
    "Users are asking for inline comments on shared reports.",
    "The beta cohort gave strong feedback on the new editor.",
    "We should deprioritize the legacy import path in the next cycle.",
    "Adoption of the mobile app is growing fastest in field teams.",
  ],
  finance: [
    "The Q3 forecast assumes the enterprise discount holds.",
    "Two invoices from the vendor are still in dispute.",
    "Budget for tooling renewals is flat year over year.",
    "Procurement wants a consolidated contract before renewal season.",
  ],
  marketing: [
    "The launch campaign assets need final review this week.",
    "Webinar registrations are tracking 30% above last quarter.",
    "The positioning doc still references the old pricing tiers.",
    "We want to spotlight two customer stories in the newsletter.",
  ],
};
const TOPIC_PRIORS = { hr: 0.25, security: 0.08, infrastructure: 0.22, product: 0.25, finance: 0.12, marketing: 0.08 };
const TOPIC_FILLER = [
  "Notes from this morning's sync are in the shared doc.",
  "Action items are assigned in the tracker.",
  "Reminder that the office is closed Monday.",
  "The recording is available for anyone who missed it.",
  "Pinged the owner in the channel this morning.",
  "Follow-ups are due by end of week.",
];

function makeTopicGen() {
  return function gen({ nTrain, nTest, seed }) {
    const rng = mulberry32(seed);
    const used = new Set();
    const docs = [];
    const classes = Object.keys(TOPIC_POOLS);
    function sampleClass() {
      let r = rng();
      for (const c of classes) { r -= TOPIC_PRIORS[c]; if (r <= 0) return c; }
      return classes[0];
    }
    function make(split) {
      for (let attempt = 0; attempt < 50; attempt++) {
        const label = sampleClass();
        const others = classes.filter((c) => c !== label);
        const s1 = pick(rng, TOPIC_POOLS[label]);
        const s2 = pick(rng, TOPIC_POOLS[label].filter((s) => s !== s1));
        const parts = [s1, s2];
        if (rng() < 0.3) parts.splice(Math.floor(rng() * 2), 0, pick(rng, TOPIC_POOLS[pick(rng, others)])); // distractor
        if (rng() < 0.5) parts.push(pick(rng, TOPIC_FILLER));
        const text = parts.join(" ");
        if (used.has(text)) continue;
        used.add(text);
        docs.push({ id: `${split}-${docs.length}`, split, text, label });
        return;
      }
      throw new Error("could not generate unique topic doc");
    }
    while (docs.length < nTrain) make("train");
    while (docs.length < nTrain + nTest) make("test");
    return { train: docs.filter((t) => t.split === "train"), test: docs.filter((t) => t.split === "test") };
  };
}

const TOPIC_CHOICE = {
  question: () => ({
    topic: choiceDef("Which team's work is this note primarily about?", {
      hr: "Hiring, onboarding, benefits, reviews, contractors.",
      security: "Threats, vulnerabilities, credentials, access reviews, audits.",
      infrastructure: "Clusters, regions, deploys, infrastructure tooling, performance.",
      product: "User feedback, features, roadmap, adoption, betas.",
      finance: "Forecasts, budgets, invoices, contracts, discounts.",
      marketing: "Campaigns, webinars, positioning, newsletters, stories.",
    }),
  }),
  classes: Object.keys(TOPIC_POOLS),
  scoreQuestion: null,
  scoreLevels: null,
};

const TOPIC_KEYWORDS = {
  choice: (text) => {
    const RULES = [
      ["hr", /hire|onboard|benefits|review cycle|contractor|full-time/i],
      ["security", /vulnerab|xss|credential|audit|unrecognized|pen test|rotate/i],
      ["infrastructure", /cluster|region|terraform|cdn|disk|batch|deploy|latency/i],
      ["product", /user|feedback|feature|roadmap|beta|adoption|editor|import/i],
      ["finance", /forecast|budget|invoice|discount|procurement|contract|renewal/i],
      ["marketing", /campaign|webinar|positioning|newsletter|customer stor|launch/i],
    ];
    return RULES.find(([, re]) => re.test(text))?.[0] ?? "product";
  },
  score: null,
};

// ---------------------------------------------------------------- export --
export const DATASETS = {
  "support-tickets": {
    description: "Short support tickets; 5-way category + explicit-cue urgency 0-3",
    gen: makeTicketGen(URGENCY_CUES_EXPLICIT),
    choice: TICKET_CHOICE,
    keywords: TICKET_KEYWORDS,
  },
  "tickets-hard": {
    description: "Same categories; urgency implied by stakes, no urgency keywords",
    gen: makeTicketGen(URGENCY_CUES_OBLIQUE),
    choice: TICKET_CHOICE,
    keywords: TICKET_KEYWORDS,
  },
  reviews: {
    description: "Product reviews; 3-way sentiment + 0-4 rating",
    gen: makeReviewGen(),
    choice: REVIEW_CHOICE,
    keywords: REVIEW_KEYWORDS,
  },
  topics: {
    description: "Multi-sentence team notes; 6-way topic, imbalanced priors + distractors",
    gen: makeTopicGen(),
    choice: TOPIC_CHOICE,
    keywords: TOPIC_KEYWORDS,
  },
};
