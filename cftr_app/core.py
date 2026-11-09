"""Shared code for the CFTR pathogenicity app (training export + Streamlit UI)."""
from pathlib import Path
import gzip
import shutil
import urllib.request

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
# Half-width of the window (bases on each side of the mutation).
# CHECK THIS against your successful notebook. The paper text says 7, but the
# example string in the paper/notebook has ~15 bases on each side.
WINDOW_HALF = 15
MAX_LEN = 128
BASES = "ACGT"

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
FASTA = ROOT / "data" / "chr7.fa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- genome / sequence ----------
def ensure_fasta(path=FASTA):
    path = Path(path)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    compressed = path.with_suffix(path.suffix + ".gz")
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr7.fa.gz"

    try:
        print("Downloading GRCh38 chromosome 7...")
        urllib.request.urlretrieve(url, compressed)

        with gzip.open(compressed, "rb") as src:
            with path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    except Exception:
        if path.exists():
            path.unlink()
        raise

    finally:
        if compressed.exists():
            compressed.unlink()

    return path


def open_genome(path=FASTA):
    ensure_fasta(path)
    return Fasta(str(path), as_raw=True, sequence_always_upper=True)


def chr7_record(genome):
    for key in ("chr7", "7", "NC_000007.14"):
        if key in genome.keys():
            return genome[key]
    raise KeyError(f"chr7 not found in FASTA. Keys: {list(genome.keys())[:5]}")


def split_window(rec, pos, ref, alt, w=WINDOW_HALF):
    """Return (left, ref, alt, right) or None if the ref allele doesn't match GRCh38."""
    p = int(pos) - 1                      # 1-based VCF -> 0-based
    ref, alt = str(ref).upper(), str(alt).upper()
    s = max(0, p - w)
    e = p + len(ref) + w
    region = str(rec[s:e])
    r = p - s
    if region[r:r + len(ref)] != ref:     # Eq. (1) in the paper
        return None
    return region[:r], ref, alt, region[r + len(ref):]


def compose(left, ref, alt, right):
    return f"{left}[{ref}>{alt}]{right}"


def load_bert():
    from transformers import BertConfig
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cfg = BertConfig.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, config=cfg)
    return tok, bert.to(DEVICE).eval()


@torch.no_grad()
def embed(seqs, tok, bert, bs=32):
    """Masked mean-pooled 768-d embeddings (Algorithm 3). Uses outputs[0]."""
    out = []
    for i in range(0, len(seqs), bs):
        inp = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(DEVICE)
        h = bert(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])[0]
        m = inp["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.vstack(out)


# ---------- predictor + explainability ----------
class Predictor:
    def __init__(self):
        self.tok, self.bert = load_bert()
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(str(ART / "xgb.json"))
        self.train_emb = np.load(ART / "train_emb.npy")
        self.train_meta = pd.read_csv(ART / "train_meta.csv")
        self._explainer = None

    def proba(self, seqs):
        emb = embed(seqs, self.tok, self.bert)
        return self.clf.predict_proba(emb)[:, 1], emb

    # 1) Flank sensitivity (occlusion / in-silico mutagenesis of the context)
    def flank_sensitivity(self, left, ref, alt, right):
        base_p = self.proba([compose(left, ref, alt, right)])[0][0]
        seqs, owner = [], []
        n_l = len(left)
        full = list(left) + list(right)
        for i, c in enumerate(full):
            for b in BASES:
                if b == c:
                    continue
                f = full.copy()
                f[i] = b
                seqs.append(compose("".join(f[:n_l]), ref, alt, "".join(f[n_l:])))
                owner.append(i)
        probs = self.proba(seqs)[0]
        owner = np.array(owner)
        mean_abs = np.array([np.abs(probs[owner == i] - base_p).mean() for i in range(len(full))])
        mean_signed = np.array([(probs[owner == i] - base_p).mean() for i in range(len(full))])
        rel = list(range(-n_l, 0)) + list(range(1, len(right) + 1))
        return base_p, pd.DataFrame({"rel_pos": rel, "base": full,
                                     "sensitivity": mean_abs, "mean_change": mean_signed})

    # 2) Score every possible alt allele at this position (SNVs)
    def alt_scan(self, left, ref, right):
        if len(ref) != 1:
            return None
        alts = [b for b in BASES if b != ref]
        p, _ = self.proba([compose(left, ref, a, right) for a in alts])
        return pd.DataFrame({"alt": alts, "P(pathogenic)": p})

    # 3) Most similar known variants in embedding space
    def neighbours(self, emb, k=5):
        a = self.train_emb / np.linalg.norm(self.train_emb, axis=1, keepdims=True)
        q = emb / np.linalg.norm(emb)
        sims = a @ q.ravel()
        idx = np.argsort(-sims)[:k]
        df = self.train_meta.iloc[idx].copy()
        df["label"] = df["cause"].map({1: "pathogenic", 0: "benign"})
        df["cosine_sim"] = sims[idx].round(3)
        return df[["pos", "ref", "alt", "label", "cosine_sim"]]

    # 4) SHAP over the 768 embedding dimensions (model-internal view)
    def shap_top(self, emb, k=10):
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.clf)
        sv = np.asarray(self._explainer.shap_values(emb)).reshape(-1)
        idx = np.argsort(-np.abs(sv))[:k]
        return pd.DataFrame({"embedding_dim": idx, "shap_value": sv[idx]})
