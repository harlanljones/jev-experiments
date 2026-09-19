// App 3 — Date extraction: Jev reads, code resolves.
// Pattern: extraction with code-side validation
// (https://docs.typesafe.ai/cookbooks/date_extraction_cookbook).
// The model only answers which parts the text names; all calendar math and
// validation happens here, so it can be unit-tested and never hallucinated.
import { choice } from "@typesafe-ai/sdk";
import { client } from "../shared/client.mjs";

const REVIEW_BELOW = 0.6; // any part weaker than this sends the date to review
const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];
const WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];

const absent = "The document does not state this.";
const parts = {
  mode: choice("How is the date written? 'absolute' = names a month; 'relative' = relative to today (today/tomorrow/a weekday); 'none' = not stated.", { absolute: null, relative: null, none: null }),
  month: choice("If absolute, which month?", Object.fromEntries([...MONTHS.map(m => [m, null]), ["none", absent]])),
  day: choice("If absolute, which day of month (1-31)?", Object.fromEntries([...Array(31).keys()].map(d => [String(d + 1), null]).concat([["none", absent]]))),
  year: choice("If absolute, which year? 'none' if the document states no year; 'out_of_range' if a year is stated but not in the list.", Object.fromEntries([...Array(12).keys()].map(i => [String(2024 + i), null]).concat([["out_of_range", "A year is stated but outside the listed range."], ["none", "No year is stated."]]))),
  anchor: choice("If relative, which day? 'today', 'tomorrow', or 'weekday' (a named day of week).", { today: null, tomorrow: null, weekday: null, none: absent }),
  weekday: choice("If a weekday is named, which one?", Object.fromEntries([...WEEKDAYS.map(w => [w, null]), ["none", absent]])),
  offset: choice("If a weekday: 'next' = 'next Thursday'; 'current' = 'this Thursday'; 'none' = bare weekday.", { next: null, current: null, none: absent }),
};

export async function extractDate(document, role) {
  const answers = (await client.systemOne({ state: document, questions: parts })).answers;
  const confs = [answers.mode.confidence];
  if (answers.mode.choice === "none")
    return { date: null, note: "no such date stated", needs_review: true };

  if (answers.mode.choice === "absolute") {
    confs.push(answers.month.confidence, answers.day.confidence, answers.year.confidence);
    const m = MONTHS.indexOf(answers.month.choice) + 1;
    const d = parseInt(answers.day.choice, 10);
    if (!m || !d) return { date: null, note: "absolute date incomplete", needs_review: true };
    const today = new Date();
    let y = parseInt(answers.year.choice, 10);
    if (Number.isNaN(y)) {
      if (answers.year.choice === "out_of_range") return { date: null, note: "year out of listed range", needs_review: true };
      y = today.getFullYear(); // no year stated -> infer
    }
    let dt = new Date(y, m - 1, d);
    if (Number.isNaN(y) && dt < today - 31 * 864e5) dt = new Date(y + 1, m - 1, d); // inferred year well past -> roll forward
    return finish(dt, confs);
  }

  if (answers.mode.choice === "relative") {
    confs.push(answers.anchor.confidence);
    const today = new Date();
    if (answers.anchor.choice === "today") return finish(today, confs);
    if (answers.anchor.choice === "tomorrow") return finish(new Date(today.getTime() + 864e5), confs);
    confs.push(answers.weekday.confidence, answers.offset.confidence);
    const w = WEEKDAYS.indexOf(answers.weekday.choice);
    if (w < 0) return { date: null, note: "weekday not read", needs_review: true };
    const monday = new Date(today.getTime() - ((today.getDay() + 6) % 7) * 864e5);
    const add = answers.offset.choice === "next" ? 7 + w + 1 : (w - ((today.getDay() + 6) % 7) + 7) % 7;
    return finish(new Date(monday.getTime() + add * 864e5), confs);
  }
  return { date: null, note: "unrecognized mode", needs_review: true };
}

function finish(dt, confs) {
  const confidence = Math.min(...confs);
  return {
    date: dt.toISOString().slice(0, 10),
    confidence,
    needs_review: confidence < REVIEW_BELOW,
  };
}

// Demo
if (process.argv[1]?.endsWith("dateread.mjs")) {
  const docs = [
    ["The license agreement takes effect on March 3, 2027 and expires August 14, 2027.", "the date the agreement expires"],
    ["Please return the signed form by the 15th of next month.", "the deadline to return the form"],
    ["The survey closes today at midnight.", "the date the survey closes"],
    ["The design review is next Thursday.", "the date of the design review"],
    ["Kickoff sometime soon.", "the date of the kickoff call"],
  ];
  for (const [doc, role] of docs) {
    const r = await extractDate(doc, role);
    console.log(`${r.needs_review ? "REVIEW " : "ACCEPT "} ${String(r.date)} (conf ${r.confidence?.toFixed?.(2) ?? r.confidence}) <- ${role}`);
  }
}
