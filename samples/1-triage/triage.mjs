// App 1 — Support ticket triage.
// Pattern: intent routing (https://docs.typesafe.ai/patterns/intent-routing).
// ONE TypeSafe request per ticket answers three questions in parallel:
// which team owns it, how urgent it is, and whether it reads as churn risk.
// Thresholds live here so they're easy to review and tune.
import { choice, score, noul } from "@typesafe-ai/sdk";
import { client } from "../shared/client.mjs";

const URGENCY_REVIEW = 2; // score 0-3: "High" or "Critical" pages on-call
const CHURN_ESCALATE = 0.7; // noul probability: above this, notify CS

const questions = {
  category: choice("What is this ticket about?", {
    billing: "Charges, refunds, invoices, payment failures.",
    technical: "Bugs, errors, outages, things not working.",
    sales: "Pricing, upgrades, new purchases, trials.",
    account: "Logins, permissions, data export, deletion.",
    other: null,
  }),
  urgency: score(
    "How urgently does this need a human response?",
    ["None: can wait days.", "Low: this week is fine.", "High: same day.", "Critical: active outage or money being lost right now."]
  ),
  churn_risk: noul("Does the customer sound like they might cancel because of this issue?"),
};

export async function triage(ticket) {
  const r = await client.systemOne({ state: ticket, questions });
  const a = r.answers;
  return {
    category: a.category.choice,
    urgency: Math.round(a.urgency.score),
    churn_risk: a.churn_risk.noul,
    route: route(a),
  };
}

function route(a) {
  if (a.urgency.score >= URGENCY_REVIEW || a.churn_risk.noul >= CHURN_ESCALATE)
    return "ESCALATE";
  return `queue:${a.category.choice}`;
}

// Demo
if (process.argv[1]?.endsWith("triage.mjs")) {
  const tickets = [
    "I was charged twice on my last invoice and my card is about to be charged a third time. Fix this today or I'm moving to a competitor.",
    "Minor thing: the logo on the settings page looks stretched on Safari. No rush.",
    "Our whole team got logged out and password reset emails aren't arriving. Sales call in 2 hours.",
  ];
  for (const t of tickets) {
    console.log("---");
    console.log(t.slice(0, 70) + "...");
    console.log(JSON.stringify(await triage(t), null, 2));
  }
}
