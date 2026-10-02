# Cross-check of the JS baselines against sklearn on the same corpora dump
# (seed 42 only). Usage: python3 sklearn-check.py  (run after bench.mjs)
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

corpora = json.loads((Path(__file__).parent / "corpus.json").read_text())

for name, corpus in corpora.items():
    train, test = corpus["train"], corpus["test"]
    has_score = "score" in train[0]
    print(f"\n=== {name} (test={len(test)}) ===")
    print("name              n   choiceAcc  choiceF1  scoreAcc  scoreMAE")
    for n in (10, 25, 50):
        tr = train[:n]
        for mk, make_clf in [
            ("sk-nb", MultinomialNB),
            ("sk-logreg", lambda: LogisticRegression(max_iter=2000, C=10)),
            ("sk-linsvc", lambda: LinearSVC(C=1)),
        ]:
            cat = make_pipeline(TfidfVectorizer(), make_clf()).fit(
                [t["text"] for t in tr], [t["label"] for t in tr])
            pc = cat.predict([t["text"] for t in test])
            classes = sorted(set(t["label"] for t in corpus["train"] + corpus["test"]))
            f1s = []
            for c in classes:
                tp = sum(p == c and g == c for p, g in zip(pc, [t["label"] for t in test]))
                fp = sum(p == c and g != c for p, g in zip(pc, [t["label"] for t in test]))
                fn = sum(p != c and g == c for p, g in zip(pc, [t["label"] for t in test]))
                f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
            ca = sum(p == t["label"] for p, t in zip(pc, test)) / len(test)
            row = [f"{mk:<16} {n:>2}  {ca*100:8.1f}%  {sum(f1s)/len(f1s)*100:8.1f}%"]
            if has_score:
                urg = make_pipeline(TfidfVectorizer(), make_clf()).fit(
                    [t["text"] for t in tr], [t["score"] for t in tr])
                pu = urg.predict([t["text"] for t in test])
                ua = sum(p == t["score"] for p, t in zip(pu, test)) / len(test)
                mae = sum(abs(p - t["score"]) for p, t in zip(pu, test)) / len(test)
                row.append(f"  {ua*100:8.1f}%  {mae:8.2f}")
            print("  ".join(row))
