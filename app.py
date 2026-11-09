import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core import ART, FASTA, WINDOW_HALF, Predictor, chr7_record, compose, open_genome, split_window

st.set_page_config(page_title="CFTR Variant Classifier", layout="wide")


@st.cache_resource(show_spinner="Loading DNABERT2 + classifier (first time takes a while)...")
def load():
    return Predictor(), chr7_record(open_genome(FASTA))


pred, rec = load()

st.title("CFTR Variant Pathogenicity Classifier")
st.caption("DNABERT2 (frozen) + XGBoost, GRCh38 chr7. Research prototype for high-recall "
           "pre-screening. Not a clinical diagnostic tool.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                    help="0.5 = paper default (84.2% pathogenic recall). "
                         "0.3 = high-recall mode (96.2% recall, but very few benign calls).")
    half = st.number_input("Window half-width", 5, 30, WINDOW_HALF,
                           help="Must match the value used in train_export.py")
    mode = st.radio("Input", ["Held-out test variant", "Manual (chr7, GRCh38)"])

if mode == "Held-out test variant":
    test = pd.read_csv(ART / "test_meta.csv")
    labels = [f"{r.pos}  {r.ref}>{r.alt}   (database label: {'pathogenic' if r.cause else 'benign'})"
              for r in test.itertuples()]
    i = st.selectbox("Pick a variant", range(len(test)), format_func=lambda k: labels[k])
    pos, ref, alt = int(test.pos[i]), test.ref[i], test.alt[i]
else:
    c1, c2, c3 = st.columns(3)
    pos = c1.number_input("Position (GRCh38, 1-based)", value=117587778, step=1)
    ref = c2.text_input("Ref allele", "G").strip().upper()
    alt = c3.text_input("Alt allele", "T").strip().upper()

if not st.button("Analyse variant", type="primary"):
    st.stop()

parts = split_window(rec, pos, ref, alt, int(half))
if parts is None:
    st.error("The ref allele does not match GRCh38 at this position (same check as Eq. 1 in the paper). "
             "Check the position/allele.")
    st.stop()
left, ref, alt, right = parts
seq = compose(*parts)

p, emb = pred.proba([seq])
p = float(p[0])
is_path = p >= thr

# ---------------- result ----------------
a, b = st.columns([1, 2])
a.metric("P(pathogenic)", f"{p:.3f}")
(a.error if is_path else a.success)(
    f"{'PATHOGENIC (flag for expert review)' if is_path else 'BENIGN-like'} at threshold {thr:.2f}")
b.progress(min(max(p, 0.0), 1.0))
b.code(seq, language=None)
b.caption("Input to DNABERT2: flanking sequence with the mutation written as [ref>alt].")

st.divider()
st.subheader("Why this prediction?")
t1, t2, t3, t4 = st.tabs(["Context sensitivity", "Alt-allele scan", "Similar known variants",
                          "Embedding (SHAP)"])

with t1:
    base_p, sens = pred.flank_sensitivity(left, ref, alt, right)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(range(len(sens)), sens["sensitivity"], color="#c0392b")
    ax.set_xticks(range(len(sens)))
    ax.set_xticklabels([f"{b}\n{r:+d}" for b, r in zip(sens["base"], sens["rel_pos"])], fontsize=7)
    ax.set_ylabel("mean |change in P(path)|")
    ax.set_title("How much the score moves if each flanking base is replaced by the other 3")
    st.pyplot(fig)
    top = sens.sort_values("sensitivity", ascending=False).head(3)
    st.write("Most influential flanking positions (relative to the variant): "
             + ", ".join(f"{int(r):+d} ({b})" for r, b in zip(top.rel_pos, top.base)))
    st.caption("Occlusion / in-silico mutagenesis: a model-behaviour explanation, not proof of biological mechanism.")

with t2:
    scan = pred.alt_scan(left, ref, right)
    if scan is None:
        st.info("Alt-allele scan is only available for single-nucleotide variants.")
    else:
        scan["selected"] = scan["alt"].eq(alt).map({True: "<- your variant", False: ""})
        st.dataframe(scan, hide_index=True)
        st.caption("Score of every possible substitution at this position.")

with t3:
    st.dataframe(pred.neighbours(emb), hide_index=True)
    st.caption("Closest training variants in DNABERT2 embedding space (cosine). "
               "Neighbours mostly of one label = the model is relying on a consistent local pattern.")

with t4:
    st.dataframe(pred.shap_top(emb), hide_index=True)
    st.caption("SHAP values of the XGBoost model for the 10 most influential of 768 embedding dimensions. "
               "Dimensions are not individually human-interpretable; use them for a global/technical view.")

st.warning("Metrics in the paper (ROC-AUC 0.698) indicate moderate discrimination. "
           "Explanations describe model behaviour and are not clinical evidence.")
