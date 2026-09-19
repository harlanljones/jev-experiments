# Jev sample apps

Three small apps demonstrating distinct System One (Jev) patterns, based on the
TypeSafe cookbooks (https://docs.typesafe.ai/cookbooks) and what people are
building with Jev (agent skill routing, LLM guardrails, structured extraction).
All constants (questions + thresholds) live at the top of each file for review.

Shared client in `shared/client.mjs` — key from `$TYPESAFE_API_KEY` or
`~/.hermes/jev/key`.

## 1-triage — Intent routing for support tickets
`node 1-triage/triage.mjs`
One request per ticket: category Choice + urgency Score + churn-risk Noul, all
in parallel. Code routes: ESCALATE (urgency >= 2 or churn >= 0.7) or a queue.

## 2-guardrail — Guardrails for an LLM app
`node 2-guardrail/guardrail.mjs`
One request screens any message: a Noul per hazard (jailbreak / harmful /
medical / self-harm) + a severity Score. Code routes pass / review / block /
support. Run on inputs AND outputs of the LLM.

## 3-dateread — Extraction where Jev reads and code resolves
`node 3-dateread/dateread.mjs`
The model only names the parts of a date (mode/month/day/year, or relative
anchors); all calendar math and validation is plain testable JS. Any part with
confidence < 0.6, or an incomplete/inconsistent read, goes to human review
instead of silently producing a wrong date.
