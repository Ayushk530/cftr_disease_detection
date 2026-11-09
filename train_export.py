"""Rebuild the paper pipeline and save everything the UI needs.

Usage (from the cftr_app folder):
  python train_export.py --csv ..\\cleaned_datasets\\a.csv ..\\cleaned_datasets\\b.csv --half 15
Each CSV needs columns: chr, pos, ref, alt, cause   (cause: 1 pathogenic, 0 benign)
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from core import (ART, FASTA, WINDOW_HALF, chr7_record, compose, embed, load_bert,
                  open_genome, split_window)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", nargs="+", required=True)
ap.add_argument("--fasta", default=str(FASTA))
ap.add_argument("--half", type=int, default=WINDOW_HALF)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# 1) load + unify + dedup
df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
df.columns = [c.lower().strip() for c in df.columns]
df = df.drop_duplicates(subset=["pos", "ref", "alt"]).reset_index(drop=True)
print(df.head())
print("merged variants:", len(df))

# 2) validate against GRCh38 + build highlighted windows
rec = chr7_record(open_genome(args.fasta))
rows, seqs = [], []
for r in df.itertuples():
    parts = split_window(rec, r.pos, r.ref, r.alt, args.half)
    if parts is None:
        continue
    rows.append((int(r.pos), str(r.ref).upper(), str(r.alt).upper(), int(r.cause)))
    seqs.append(compose(*parts))
meta = pd.DataFrame(rows, columns=["pos", "ref", "alt", "cause"])
meta["seq"] = seqs
print(f"passed reference validation: {len(meta)}  (paper: 1,972 of 2,301)")
print(meta["cause"].value_counts().to_dict(), "(paper: 1304 pathogenic / 668 benign)")

# 3) embeddings (frozen DNABERT2)
tok, bert = load_bert()
X = embed(meta["seq"].tolist(), tok, bert)
y = meta["cause"].values

# 4) leakage-safe split by 1 kb genomic bin
groups = (meta["pos"] // 1000).values
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(X, y, groups))
spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()          # neg/pos, as in the paper

params = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
              random_state=args.seed)
clf = xgb.XGBClassifier(**params).fit(X[tr], y[tr])

p = clf.predict_proba(X[te])[:, 1]
print(f"\nTEST  ROC-AUC {roc_auc_score(y[te], p):.3f} (paper 0.698)  "
      f"PR-AUC {average_precision_score(y[te], p):.3f} (paper 0.806)")
for t in (0.5, 0.3):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
    print(f"  threshold {t}: pathogenic recall {tp / (tp + fn):.3f}  benign specificity {tn / (tn + fp):.3f}")

cv = []
for a, b in GroupKFold(n_splits=5).split(X, y, groups):
    m = xgb.XGBClassifier(**params).fit(X[a], y[a])
    cv.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
print(f"5-fold GroupKFold ROC-AUC {np.mean(cv):.3f} +/- {np.std(cv):.3f} (paper 0.661 +/- 0.019)")

# 5) save artifacts for the UI
ART.mkdir(exist_ok=True)
clf.save_model(str(ART / "xgb.json"))
np.save(ART / "train_emb.npy", X[tr])
meta.iloc[tr][["pos", "ref", "alt", "cause"]].to_csv(ART / "train_meta.csv", index=False)
meta.iloc[te][["pos", "ref", "alt", "cause"]].to_csv(ART / "test_meta.csv", index=False)
print("\nSaved to", ART)
