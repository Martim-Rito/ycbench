"""Shapley Regression feasibility experiment on W26 Dataset v1 (methodological, NOT a result).

Uses the authors' official implementation (local clone of github.com/tomasbrogueira/shapley_regression,
imported read-only; no bytecode is written there) with our own manual CV loop (the official estimator has no
`classes_`, so cross_val_predict is unusable).

Primary question: what changes when ONLY the interaction order k (1, 2, 3) changes?
  - feature sets: S3 (n=3: founders, previous YC, one-liner words) and S5 (S3 + TLD_ai + TLD_other; .com = reference)
  - L2, C=1, lbfgs, same 198 rows, same target_top20, RepeatedStratifiedKFold 5x10 seed 0,
    imputation + MinMax scaling fitted inside each training fold, no class weights, no tuning.
  - real labels vs shuffled labels (10 deterministic permutations of y; stratified CV re-run on each).
Secondary: regularisation (L2 / L1 / unregularised) for k=2,3 on S5, no tuning.

Writes reports/w26_shapley_feasibility_report.md. Optional hand-written interpretation is read from
reports/w26_shapley_feasibility_interpretation.md if it exists.
"""
import itertools
import subprocess
import sys
import warnings
from math import comb
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the (read-only) official repository

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT.parent / "shapley_regression"
sys.path.insert(0, str(OFFICIAL))
from core.models.choquet import choquet_k_additive_shapley  # noqa: E402
from core.models.regression import ChoquisticRegression  # noqa: E402

DATA = ROOT / "data" / "processed" / "w26_dataset_v1.csv"
REPORT = ROOT / "reports" / "w26_shapley_feasibility_report.md"
INTERP = ROOT / "reports" / "w26_shapley_feasibility_interpretation.md"

SEED = 0
N_SPLITS, N_REPEATS = 5, 10
K_TOP = 20
TOL = 1e-6                       # |coef| <= TOL is "numerically zero" (declared before inspecting results)
STAB_REF = 0.70                  # descriptive reference level only (paper's 70%), not a test
SHUFFLE_SEEDS = [1000 + i for i in range(10)]
K_VALUES = (1, 2, 3)

NUM = ["founder_n_founders", "company_one_liner_n_words"]
BIN = ["founder_previous_yc"]
SETS = {"S3": dict(num=NUM, bin=BIN, ohe=[]), "S5": dict(num=NUM, bin=BIN, ohe=["TLD_ai", "TLD_other"]),
        # S4 = EXTRA sensitivity set (not requested): one non-.com dummy instead of two mutually exclusive dummies, see section 2b
        "S4": dict(num=NUM, bin=BIN, ohe=["TLD_not_com"])}
REGS = {  # penalty / solver as passed to the official wrapper (which forwards them to sklearn LogisticRegression)
    "L2": dict(penalty="l2", solver="lbfgs", C=1.0),
    "L1": dict(penalty="l1", solver="saga", C=1.0),
    "UNREG": dict(penalty=None, solver="lbfgs", C=1.0),  # C is ignored when penalty=None
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args]).decode().strip()


# ------------------------------------------------------------------ data
df = pd.read_csv(DATA)
df = df[df["target_defined"] == 1].sort_values("canonical_slug").reset_index(drop=True)
assert len(df) == 198 and int(df["target_top20"].sum()) == 20
y = df["target_top20"].astype(int).to_numpy()
assert df["company_site_tld_group"].isin(["com", "ai", "other"]).all()
df["TLD_ai"] = (df["company_site_tld_group"] == "ai").astype(int)
df["TLD_other"] = (df["company_site_tld_group"] == "other").astype(int)
df["TLD_not_com"] = (df["company_site_tld_group"] != "com").astype(int)
miss_rows = df[df[NUM + BIN].isna().any(axis=1)]
assert len(miss_rows) == 5 and miss_rows["target_top20"].sum() == 0


def feature_order(spec):
    return spec["num"] + spec["bin"] + spec["ohe"]  # = column order produced by the ColumnTransformer below


def coalitions(names, k):
    out = []
    for r in range(1, k + 1):
        out += list(itertools.combinations(range(len(names)), r))
    return out  # same enumeration as official choquet_k_additive_shapley


def make_pipe(spec, k, reg):
    parts = [("num", SimpleImputer(strategy="median"), spec["num"]),
             ("bin", SimpleImputer(strategy="most_frequent"), spec["bin"])]
    if spec["ohe"]:
        parts.append(("ohe", "passthrough", spec["ohe"]))
    return Pipeline([
        ("impute", ColumnTransformer(parts)),
        ("sr", ChoquisticRegression(representation="shapley", k_add=k, scale_data=True, max_iter=1000, random_state=SEED, **REGS[reg])),
    ])


def top_k_hits(p, yy, k):
    """Same rule as run_w26_v1_experiment.py: ties at the boundary broken by row (slug) order."""
    order = np.argsort(-p, kind="stable")
    det = int(yy[order[:k]].sum())
    cut = p[order[k - 1]]
    above, tied = p > cut, p == cut
    exp = yy[above].sum() + (k - above.sum()) * yy[tied].mean()
    return det, float(exp), int(tied.sum())


def run_cv(yy, spec, k, reg):
    X = df[feature_order(spec)]
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X, yy))
    vr, va, tr_, ta, coefs, ints = [], [], [], [], [], []
    oof = np.zeros((N_REPEATS, len(yy)))
    n_conv = 0
    for i, (tr, te) in enumerate(splits):
        pipe = make_pipe(spec, k, reg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit(X.iloc[tr], yy[tr])
        n_conv += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        p_te = pipe.predict_proba(X.iloc[te])[:, 1]
        p_tr = pipe.predict_proba(X.iloc[tr])[:, 1]
        oof[i // N_SPLITS, te] = p_te
        vr.append(roc_auc_score(yy[te], p_te)); va.append(average_precision_score(yy[te], p_te))
        tr_.append(roc_auc_score(yy[tr], p_tr)); ta.append(average_precision_score(yy[tr], p_tr))
        coefs.append(pipe.named_steps["sr"].coef_[0].copy()); ints.append(float(pipe.named_steps["sr"].intercept_[0]))
    top = [top_k_hits(oof[r], yy, K_TOP) for r in range(N_REPEATS)]
    return dict(val_roc=np.array(vr), val_ap=np.array(va), tr_roc=np.array(tr_), tr_ap=np.array(ta),
                oof_roc=np.array([roc_auc_score(yy, oof[r]) for r in range(N_REPEATS)]),
                oof_ap=np.array([average_precision_score(yy, oof[r]) for r in range(N_REPEATS)]),
                top=np.array([t[0] for t in top]), ties=np.array([t[2] for t in top]),
                coefs=np.array(coefs), intercepts=np.array(ints), n_conv=n_conv, n_fits=len(splits))


# ------------------------------------------------------------------ coefficient statistics
def coef_table(res, names, k):
    """Per-coefficient stats across the 50 fitted folds. Sign consistency = max(frac +, frac -) among fits where |coef| > TOL."""
    C = res["coefs"]
    co = coalitions(names, k)
    assert C.shape[1] == len(co) == sum(comb(len(names), r) for r in range(1, k + 1))
    rows = []
    for j, A in enumerate(co):
        c = C[:, j]
        nz = np.abs(c) > TOL
        pos, neg = (c > TOL).sum(), (c < -TOL).sum()
        rows.append(dict(coalition=" × ".join(names[i] for i in A), order=len(A), mean=c.mean(), sd=c.std(ddof=1),
                         mean_abs=np.abs(c).mean(), nonzero_frac=nz.mean(),
                         sign_cons=(max(pos, neg) / nz.sum()) if nz.sum() else np.nan))
    return pd.DataFrame(rows)


def rank_stability(res, names, k):
    """Mean Spearman correlation between the signed interaction (order >= 2) coefficient vectors of different folds
    of the same repeat (10 fold pairs x 10 repeats). Training sets overlap by 60-75%, so this is optimistic."""
    orders = np.array([len(A) for A in coalitions(names, k)])
    sel = orders >= 2
    if k < 2 or sel.sum() < 3:
        return np.nan
    C = res["coefs"][:, sel]
    vals = []
    for r in range(N_REPEATS):
        blk = C[r * N_SPLITS:(r + 1) * N_SPLITS]
        for a, b in itertools.combinations(range(N_SPLITS), 2):
            if np.ptp(blk[a]) > 0 and np.ptp(blk[b]) > 0:
                vals.append(spearmanr(blk[a], blk[b])[0])
    return float(np.nanmean(vals)) if vals else np.nan


def order_summary(tab, res, names, k):
    rows = []
    tot = tab["mean_abs"].sum()
    for o in range(1, k + 1):
        t = tab[tab["order"] == o]
        sc = t["sign_cons"].dropna()
        rows.append(dict(order=o, n_coef=len(t), mean_abs=t["mean_abs"].mean(), mass_share=t["mean_abs"].sum() / tot if tot > 0 else np.nan,
                         mean_sign_cons=sc.mean() if len(sc) else np.nan,
                         frac_ge_ref=(sc >= STAB_REF).mean() if len(sc) else np.nan, mean_nonzero_frac=t["nonzero_frac"].mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ validation (assertions; recorded in the report)
VALID = []


def check(name, ok, detail=""):
    VALID.append((name, bool(ok), detail))
    assert ok, f"validation failed: {name} {detail}"


def independent_phi(Xs, A):
    """Eq. (1) of the paper, evaluated directly for one coalition A (not via the official enumeration)."""
    s = np.zeros(len(Xs))
    for r in range(1, len(A) + 1):
        for C in itertools.combinations(A, r):
            s += (-1) ** (len(A) - r) / (len(A) - r + 1) * Xs[:, list(C)].min(axis=1)
    return s


spec5 = SETS["S5"]
names5 = feature_order(spec5)
imp = ColumnTransformer([("num", SimpleImputer(strategy="median"), spec5["num"]), ("bin", SimpleImputer(strategy="most_frequent"), spec5["bin"]),
                         ("ohe", "passthrough", spec5["ohe"])])
Xs5 = MinMaxScaler().fit_transform(imp.fit_transform(df[names5]))
for k in (1, 2, 3):
    T = choquet_k_additive_shapley(Xs5, k)
    co = coalitions(names5, k)
    check(f"official basis shape k={k}", T.shape == (198, sum(comb(5, r) for r in range(1, k + 1))), str(T.shape))
    worst, unique = 0.0, True
    for j, A in enumerate(co):
        phi = independent_phi(Xs5, A)
        worst = max(worst, np.abs(T[:, j] - phi).max())
        match = [m for m in range(T.shape[1]) if np.allclose(T[:, m], phi, atol=1e-12)]
        unique &= (match == [j])
    check(f"coalition->column mapping and values, independent Eq.(1), k={k}", worst < 1e-12 and unique, f"max|diff|={worst:.1e}, unique match={unique}")
check("k=1 basis equals scaled features", np.allclose(choquet_k_additive_shapley(Xs5, 1), Xs5))
# pairwise closed form: phi_ij = min(xi,xj) - (xi+xj)/2
T2 = choquet_k_additive_shapley(Xs5, 2)
check("pair column (0,1) = min - (xi+xj)/2", np.allclose(T2[:, 5], np.minimum(Xs5[:, 0], Xs5[:, 1]) - (Xs5[:, 0] + Xs5[:, 1]) / 2))
# k=1 official == plain MinMax + LR (same fold, same folds' imputation)
tr, te = next(iter(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED).split(df, y)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = make_pipe(spec5, 1, "L2").fit(df[names5].iloc[tr], y[tr]).predict_proba(df[names5].iloc[te])[:, 1]
    plain = Pipeline([("impute", ColumnTransformer([("num", SimpleImputer(strategy="median"), spec5["num"]), ("bin", SimpleImputer(strategy="most_frequent"), spec5["bin"]),
                                                    ("ohe", "passthrough", spec5["ohe"])])),
                      ("mm", MinMaxScaler()), ("lr", LogisticRegression(random_state=SEED))]).fit(df[names5].iloc[tr], y[tr])
    b = plain.predict_proba(df[names5].iloc[te])[:, 1]
check("k=1 official == MinMax + sklearn LogisticRegression (predict_proba)", np.abs(a - b).max() < 1e-10, f"max|diff|={np.abs(a-b).max():.1e}")
# regularisation settings are really what we think (compare with native sklearn on the same basis)
Z = choquet_k_additive_shapley(Xs5, 2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    off = {r: ChoquisticRegression(k_add=2, scale_data=False, max_iter=1000, random_state=SEED, **REGS[r]).fit(Xs5, y) for r in REGS}
    nat = {"L2": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED).fit(Z, y),
           "L1": LogisticRegression(C=1.0, l1_ratio=1, solver="saga", max_iter=1000, random_state=SEED).fit(Z, y),
           "UNREG": LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000, random_state=SEED).fit(Z, y)}
for r in REGS:
    d = np.abs(off[r].coef_ - nat[r].coef_).max()
    check(f"official {r} == native sklearn ({'C=inf' if r == 'UNREG' else 'l1_ratio=1' if r == 'L1' else 'default L2'})", d < 1e-6, f"max|dcoef|={d:.1e}")
check("L1 is sparse, L2 is dense (k=2, S5, full data)", (np.abs(off['L1'].coef_) > TOL).sum() < Z.shape[1] and (np.abs(off['L2'].coef_) > TOL).all(),
      f"L1 nonzero={(np.abs(off['L1'].coef_) > TOL).sum()}/{Z.shape[1]}")

# basis rank diagnostic. Observed rank (198 rows) vs "structural" rank = rank on a large simulated sample in which every feature
# combination that is logically possible is observed (independent features with the observed marginals; TLD groups mutually exclusive).
# Deficiency that disappears in the simulation is a finite-sample property of this dataset; deficiency that stays is structural.
RANKS = []
_rng = np.random.default_rng(1)
N_SIM = 100_000


def _simulate(spec, Xobs):
    """Xobs = imputed, unscaled observed matrix in feature_order(spec) order."""
    nm = feature_order(spec)
    cols = {}
    for c in spec["num"]:
        cols[c] = _rng.choice(Xobs[:, nm.index(c)], N_SIM)
    for c in spec["bin"]:
        cols[c] = (_rng.random(N_SIM) < Xobs[:, nm.index(c)].mean()).astype(float)
    grp = _rng.choice(["com", "ai", "other"], N_SIM, p=[(df["company_site_tld_group"] == g).mean() for g in ("com", "ai", "other")])
    for c in spec["ohe"]:
        cols[c] = {"TLD_ai": grp == "ai", "TLD_other": grp == "other", "TLD_not_com": grp != "com"}[c].astype(float)
    return np.column_stack([cols[c] for c in nm])


def _rank_sv(T):
    Tc = T - T.mean(0)
    sv = np.linalg.svd(Tc, compute_uv=False)
    return int(np.linalg.matrix_rank(Tc, tol=1e-9)), sv


XOBS = {}
for sname, spec in SETS.items():
    nm = feature_order(spec)
    imp_ = ColumnTransformer([("num", SimpleImputer(strategy="median"), spec["num"]), ("bin", SimpleImputer(strategy="most_frequent"), spec["bin"])]
                             + ([("ohe", "passthrough", spec["ohe"])] if spec["ohe"] else []))
    XOBS[sname] = imp_.fit_transform(df[nm])
    Xs_ = MinMaxScaler().fit_transform(XOBS[sname])
    Xsim = MinMaxScaler().fit_transform(_simulate(spec, XOBS[sname]))
    for k in (1, 2, 3):
        Tk = choquet_k_additive_shapley(Xs_, k)
        rk, sv = _rank_sv(Tk)
        rk_s, _ = _rank_sv(choquet_k_additive_shapley(Xsim, k))
        RANKS.append((sname, k, Tk.shape[1], rk, rk_s, float(sv[rk - 1]), float(sv[0] / sv[rk - 1])))
RK = {(a, b): (c, d, e) for a, b, c, d, e, _, _ in RANKS}
check("S5: pair (TLD_ai, TLD_other) column == -(TLD_ai + TLD_other)/2 exactly (mutually exclusive dummies)",
      np.allclose(choquet_k_additive_shapley(Xs5, 2)[:, [j for j, A in enumerate(coalitions(names5, 2)) if A == (3, 4)][0]], -(Xs5[:, 3] + Xs5[:, 4]) / 2))
check("S3 full column rank at k=1,2,3 on the 198 rows", all(RK[("S3", k)][0] == RK[("S3", k)][1] for k in (1, 2, 3)), str([RK[("S3", k)][:2] for k in (1, 2, 3)]))
check("S4 full column rank at k=1,2; rank-deficient at k=3 (rank 12 of 14); full rank structurally at all k",
      RK[("S4", 1)][0] == 4 and RK[("S4", 2)][0] == 10 and RK[("S4", 3)][:2] == (14, 12) and all(RK[("S4", k)][2] == RK[("S4", k)][0] for k in (1, 2, 3)), str(RK[("S4", 3)]))
# S4 k=3 mechanism: exactly one row has founder_previous_yc = 1 and TLD_not_com = 1; there min(a,p,t) and min(w,p,t) are constant multiples of min(p,t)
X4 = MinMaxScaler().fit_transform(XOBS["S4"])
a4, w4, p4, t4 = X4.T
one = np.where((p4 == 1) & (t4 == 1))[0]
check("S4: founder_previous_yc=1 & TLD_not_com=1 occurs in exactly one row", len(one) == 1, f"row index {one.tolist()}, previous-YC rows = {int(p4.sum())}")
check("S4: min(a,p,t) = a* . min(p,t) and min(w,p,t) = w* . min(p,t) on all 198 rows (a*, w* = values in that single row)",
      np.allclose(np.minimum.reduce([a4, p4, t4]), a4[one[0]] * np.minimum(p4, t4)) and np.allclose(np.minimum.reduce([w4, p4, t4]), w4[one[0]] * np.minimum(p4, t4)),
      f"a*={a4[one[0]]:.3f}, w*={w4[one[0]]:.3f}")

# ------------------------------------------------------------------ run
real, null = {}, {}


def run_condition(sname, k, reg):
    spec = SETS[sname]
    key = (sname, k, reg)
    real[key] = run_cv(y, spec, k, reg)
    null[key] = []
    for s in SHUFFLE_SEEDS:
        ys = np.random.default_rng(s).permutation(y)
        null[key].append(run_cv(ys, spec, k, reg))
    print("done", key, flush=True)


for sname in SETS:
    for k in K_VALUES:
        run_condition(sname, k, "L2")
for sname in ("S3", "S5"):  # secondary regularisation comparison: S3 (full rank, primary) and S5 (original brief, sensitivity)
    for k in (2, 3):
        for reg in ("L1", "UNREG"):
            run_condition(sname, k, reg)

# null calibration (descriptive): many more permutations for the cheapest S3 conditions, same pipeline, same CV
CAL_SEEDS = [2000 + i for i in range(100)]
cal = {}
for k in (1, 3):
    cal[k] = [run_cv(np.random.default_rng(sd).permutation(y), SETS["S3"], k, "L2") for sd in CAL_SEEDS]
    print("done calibration S3 k", k, flush=True)


# ------------------------------------------------------------------ report
def ms(a, d=3):
    a = np.asarray(a, float)
    return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"


def rng_(a):
    return f"[{int(min(a))}-{int(max(a))}]"


def metric_row(label, ncoef, r):
    gap = r["tr_roc"] - r["val_roc"]
    return (f"| {label} | {ncoef} | {ms(r['val_roc'])} | {ms(r['val_ap'])} | {r['oof_roc'].mean():.3f} | {r['oof_ap'].mean():.3f} | "
            f"{r['top'].mean():.2f} ± {r['top'].std(ddof=1):.2f} {rng_(r['top'])} | {r['tr_roc'].mean():.3f} | {r['val_roc'].mean():.3f} | {gap.mean():.3f} | "
            f"{r['tr_ap'].mean():.3f} / {r['val_ap'].mean():.3f} | {r['n_conv']} |")


def null_row(label, ncoef, lst):
    f = lambda key: np.array([x[key].mean() for x in lst])
    top = f("top")
    gap = np.array([(x["tr_roc"] - x["val_roc"]).mean() for x in lst])
    return (f"| {label} | {ncoef} | {ms(f('val_roc'))} | {ms(f('val_ap'))} | {ms(f('oof_roc'))} | {ms(f('oof_ap'))} | "
            f"{top.mean():.2f} ± {top.std(ddof=1):.2f} [{top.min():.1f}-{top.max():.1f}] | {f('tr_roc').mean():.3f} | {f('val_roc').mean():.3f} | {ms(gap)} | "
            f"{f('tr_ap').mean():.3f} / {f('val_ap').mean():.3f} | {sum(x['n_conv'] for x in lst)} |")


HDR = ("| model | #coef | fold ROC-AUC (val) | fold PR-AUC (val) | OOF ROC-AUC | OOF AP | OOF Top-20 mean ± SD [range] | train ROC | val ROC | gap (train−val) | train / val AP | conv. warnings |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|")
NHDR = HDR.replace("OOF ROC-AUC |", "OOF ROC-AUC (mean ± SD over seeds) |").replace("OOF Top-20 mean ± SD [range]", "OOF Top-20: mean ± SD of per-seed means [min-max]")

nco = lambda n, k: sum(comb(n, r) for r in range(1, k + 1))
L = []
head_ycb, head_sr = git(ROOT, "rev-parse", "--short", "HEAD"), git(OFFICIAL, "rev-parse", "--short", "HEAD")
sr_clean = git(OFFICIAL, "status", "--short") == ""
L += ["# W26 Shapley Regression: feasibility experiment", "",
      "_Generated by `scripts/run_w26_shapley_feasibility.py`. **Feasibility / methodological experiment, not a result.** "
      "Nothing here shows that Shapley Regression is a good startup predictor, that k=2 is optimal, that k=3 is inherently worse, or that any interaction is meaningful._", "",
      "## 0. Provenance", "",
      f"- YC Bench repo HEAD `{head_ycb}`; official `shapley_regression` clone at `{head_sr}` (working tree clean after the run: {sr_clean}); scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}.",
      "- Modelling uses the official `ChoquisticRegression(representation='shapley', k_add=k, scale_data=True)` unchanged (MinMax scaling -> Shapley basis -> sklearn `LogisticRegression`). Our code only adds the imputer, the manual CV loop, coalition bookkeeping and statistics.",
      "- The official estimator always forwards `penalty=`, which scikit-learn 1.9.1 flags with a FutureWarning (deprecated in 1.8, removed in 1.10). The warning is expected and was not suppressed in the fits; convergence warnings are counted per condition.",
      "- Dataset: `data/processed/w26_dataset_v1.csv`, 198 `target_defined` rows, 20 positives (`target_top20`), unchanged. Google columns are **not** used.", "",
      "## 1. Design", "",
      f"- Feature sets: **S3** = founder_n_founders, company_one_liner_n_words, founder_previous_yc (n=3). **S5** = S3 + TLD_ai + TLD_other, .com as reference (n=5). TLD is treated as a sensitivity set: its association with the target is plausibly a Google/domain measurement artefact, so TLD coefficients are not interpreted.",
      "- **S4 (extra, not in the brief)** = S3 + one `TLD_not_com` dummy (n=4). Added after the rank diagnostic in section 2b showed that S5's two mutually exclusive TLD dummies create redundant Shapley columns. S4 removes that structural redundancy for k<=2, but section 2c shows S4 k=3 is still rank-deficient for a different, finite-sample reason. **S3 is the only set that is full column rank at k=1,2,3** (k=3 is poorly conditioned), so S3 is the primary set for the k comparison and for the regularisation comparison; S4 and S5 are sensitivity sets.",
      "- Imputation (median for numeric, most frequent for binary; 5 rows, 0 positives) and MinMax scaling are fitted inside each training fold. L2, C=1, lbfgs, max_iter=1000, no class weights, no tuning.",
      f"- `RepeatedStratifiedKFold` {N_SPLITS}x{N_REPEATS}, seed {SEED} (50 fits per condition). Top-20 uses the same tie rule as `run_w26_v1_experiment.py` (OOF predictions per repeat, ties by slug order). Theoretical chance Top-20 = 2.0; the 100-permutation calibration in section 6 shows the pooled OOF Top-20 under shuffled labels is about 1.6 on S3, so both references are given.",
      f"- Shuffled-label null: `y` permuted with `numpy.random.default_rng(s).permutation`, s in {SHUFFLE_SEEDS[0]}..{SHUFFLE_SEEDS[-1]} ({len(SHUFFLE_SEEDS)} permutations, deterministic); stratified CV (seed {SEED}) is re-run on each permuted `y`. Null cells are the mean over permutations (± SD across permutations), so a single lucky shuffle cannot define the null. This is a deliberate deviation from 'a shuffle seed' (one shuffle with 20 positives is too noisy to be a reference).",
      f"- Coefficient statistics: |coef| <= {TOL:g} is numerically zero. **Sign consistency** = max(fraction positive, fraction negative) among the fits in which |coef| > {TOL:g}, over the 50 fits (defined before inspecting results). {STAB_REF:.0%} is a descriptive reference level, not a test. Rank stability = mean Spearman correlation between the signed order>=2 coefficient vectors of different folds of the same repeat; training sets overlap by 60-75%, so all stability numbers are optimistic.",
      "- Coalition names come from our own enumeration (`itertools.combinations`, orders 1..k), verified against the official basis (section 2), not from the official `Feature_i` names.", "",
      "## 2. Validation checks (all asserted at run time)", "", "| check | pass | detail |", "|---|---|---|"]
L += [f"| {n} | {'yes' if ok else 'NO'} | {d} |" for n, ok, d in VALID]
L += ["", "## 2b. Basis rank diagnostic (found while inspecting results)", "",
      "Rank = rank of the centred Shapley basis (all 198 rows, MinMax-scaled). **Structural rank** = the same on a simulated sample of 100,000 rows with the observed marginals, independent features and mutually exclusive TLD groups, i.e. a sample in which every logically possible feature combination is observed (seeded). "
      "Deficiency that is also present in the simulation is **structural** (implied by the encoding); deficiency present only in the observed data is **finite-sample** (some combination is too rare or never observed). Condition number = largest / smallest non-zero singular value of the centred basis (descriptive; a full-rank basis with a large condition number is still weakly identified).", "",
      "| set | k | #columns | rank (198 rows) | structural rank (simulated) | redundant: structural | redundant: finite-sample only | smallest singular value | condition number |", "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {s_} | {k} | {nc} | {rk} | {rks} | {nc - rks} | {rks - rk} | {sv:.3f} | {cn:.0f} |" for s_, k, nc, rk, rks, sv, cn in RANKS]
L += ["",
      "**S5 (structural).** `TLD_ai` and `TLD_other` are mutually exclusive, so `min(TLD_ai, TLD_other) = 0` and the Shapley column for that pair equals −(TLD_ai + TLD_other)/2 exactly (asserted in section 2): a linear function of the two main-effect columns, not an interaction. "
      "Every triple containing both dummies is likewise in the span of lower-order columns. These coefficients are decided by the penalty, not by the data. S5 additionally has finite-sample redundancies because `founder_previous_yc` is 1 in only 5 rows. "
      "**S5 is kept as requested but is a sensitivity analysis only; its TLD_ai × TLD_other and TLD-pair triples are artefacts of dummy coding.**", "",
      "## 2c. Why S4 k=3 is still rank-deficient (mechanism)", "",
      "S4 = {a = n_founders, w = one-liner words, p = founder_previous_yc, t = TLD_not_com}. The structural rank is full at k=1,2,3 (simulated column), so the encoding itself implies no dependency; the two redundant columns come from the data. Findings (asserted in section 2):", "",
      "1. `p` and `t` are binary, so min(p,t) = p·t, and the Shapley column φ(p,t) = pt − (p+t)/2. The cell p=1 & t=1 contains **exactly one row** (row index "
      f"{one.tolist()}; only {int(p4.sum())} rows have p=1, 4 of them with t=0).",
      "2. For binary p,t and any other feature u in [0,1], min(u,p,t) = u·p·t, which is non-zero only in that single row. In that row a and w are fixed numbers a*, w*, so **min(a,p,t) = a*·min(p,t) and min(w,p,t) = w*·min(p,t) on all 198 rows** (asserted, a*=" f"{a4[one[0]]:.3f}, w*={w4[one[0]]:.3f}).",
      "3. Each Shapley column φ(A) is a fixed linear combination of the min-terms min(C), C⊆A, with coefficient 1 on min(A) itself, so the k=3 basis is an invertible (triangular) recombination of the min-term matrix and has the same rank. The min-term matrix contains min(p,t), min(a,p,t), min(w,p,t), which by (2) satisfy two exact linear relations (min(a,p,t) − a*·min(p,t) = 0 and min(w,p,t) − w*·min(p,t) = 0). Hence 2 redundant columns: 14 columns, rank 12. Empirically the null space of the centred basis is 2-dimensional and its sparse basis contains exactly the φ-columns that feed these two relations (e.g. φ(a), φ(a,p), φ(a,t), φ(p,t), φ(a,p,t)).",
      "4. This is **finite-sample**, not structural: the simulation (all combinations observed) is full rank, and a single further row with p=t=1 and different a and w would restore it. It is a rare-support problem of `founder_previous_yc` (5 rows of 198 have p=1, only 1 of them with t=1), not of the encoding of TLD.",
      "5. There is also a near-dependence: the smallest non-zero singular value of S4 k=3 is 0.021 (condition number ≈ 390), carried mainly by φ(a,w,p) and φ(w,p), φ(a,p), φ(a,w): every column containing `p` has its non-linear part supported on the 5 rows where p=1, and " + f"all {int((XOBS['S3'][:, 2] == 1).sum())} of them are negatives ({int(y[XOBS['S3'][:, 2] == 1].sum())} of the 20 positives)" + ". S3 k=3 (smallest singular value 0.033) is full rank but weakly identified for the same reason.", "",
      "**Identifiability of individual coefficients on these 198 rows** (full column rank is necessary, not sufficient; the condition number is in the table above):", "",
      "| set | k=1 | k=2 | k=3 |", "|---|---|---|---|"]
def _ident(sn, k):
    nc, rk, rks = RK[(sn, k)]
    if rks < nc:
        return f"no (structural redundancy {nc - rks})" + (f" + finite-sample {rks - rk}" if rks > rk else "")
    if rk < nc:
        return f"no (finite-sample rank {rk}/{nc})"
    return "yes, ill-conditioned" if k == 3 else "yes"
for sn in SETS:
    L.append(f"| {sn} | " + " | ".join(_ident(sn, k) for k in (1, 2, 3)) + " |")
L += ["", "Consequence: the main k-comparison uses S3. For S4 k=3 the coefficients φ(p,t), φ(a,p,t), φ(w,p,t) are not individually identified; S5 k=2 and k=3 contain structurally non-identified columns. Aggregate behaviour of the penalised model (predictions, validation metrics) is still well defined in every case, because L2/L1 select one solution; it is the individual coefficients that must not be read.", "",
"## 3-4. Main experiment: only k changes (L2, C=1)", "",
      "Columns: fold ROC/PR = mean ± SD over the 50 validation folds (4 positives per fold); OOF = out-of-fold predictions per repeat, averaged over the 10 repeats; train/val ROC and AP are fold means of the same fits; gap = train ROC − validation ROC.", ""]
for sname in SETS:
    n = len(feature_order(SETS[sname]))
    L += [f"### {sname} (n={n}) — real labels", "", HDR] + [metric_row(f"k={k}", nco(n, k), real[(sname, k, 'L2')]) for k in K_VALUES] + [""]
    L += [f"### {sname} (n={n}) — shuffled labels (mean over {len(SHUFFLE_SEEDS)} permutations)", "", NHDR] + [null_row(f"k={k}", nco(n, k), null[(sname, k, 'L2')]) for k in K_VALUES] + [""]
L += ["Number of Shapley-basis coefficients (intercept excluded) = Σ_{r<=k} C(n,r): S3: " + ", ".join(f"k={k}: {nco(3, k)}" for k in K_VALUES) + "; S5: " + ", ".join(f"k={k}: {nco(5, k)}" for k in K_VALUES) + ".", ""]

# stability
L += ["## 5. Interaction stability (L2, C=1)", "",
      "Per-order summary over the 50 fits. `mass share` = share of Σ mean|coef| carried by that order. Real labels vs the mean over the shuffled-label permutations (same pipeline).", ""]
for sname in SETS:
    names = feature_order(SETS[sname])
    for k in (2, 3):
        res = real[(sname, k, "L2")]
        tab = coef_table(res, names, k)
        osr = order_summary(tab, res, names, k)
        nl = []
        for x in null[(sname, k, "L2")]:
            t = coef_table(x, names, k)
            nl.append(order_summary(t, x, names, k))
        nmean = pd.concat(nl).groupby("order").mean()
        rs_real = rank_stability(res, names, k)
        rs_null = np.nanmean([rank_stability(x, names, k) for x in null[(sname, k, "L2")]])
        L += [f"### {sname}, k={k}  (rank stability of order>=2 coefficients: real {rs_real:.2f}, shuffled {rs_null:.2f})", "",
              "| order | #coef | mean abs coef (real) | mean abs coef (shuffled) | mass share (real) | mean sign consistency (real) | (shuffled) | share of coefs with consistency >= 70% (real) | (shuffled) |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in osr.iterrows():
            o = int(r["order"]); nr = nmean.loc[o]
            L.append(f"| {o} | {int(r['n_coef'])} | {r['mean_abs']:.3f} | {nr['mean_abs']:.3f} | {r['mass_share']:.2f} | {r['mean_sign_cons']:.2f} | {nr['mean_sign_cons']:.2f} | {r['frac_ge_ref']:.2f} | {nr['frac_ge_ref']:.2f} |")
        L.append("")
        if RK[(sname, k)][1] < RK[(sname, k)][0]:
            L += [f"> **Identifiability warning ({sname} k={k}):** the basis has rank {RK[(sname, k)][1]} for {RK[(sname, k)][0]} columns (section 2b/2c). The individual coefficients of the redundant columns are decided by the penalty, so the sign-consistency and magnitude figures below are not evidence about those interactions.", ""]
        for o in range(2, k + 1):
            t = tab[tab["order"] == o].copy()
            t["snr"] = t["mean"].abs() / (t["sd"] + 1e-12)
            t = t.sort_values(["sign_cons", "snr"], ascending=False).head(6)
            L += [f"Most sign-stable order-{o} coefficients, {sname} k={k} (ties on consistency broken by |mean|/SD; descriptive only):", "",
                  "| coalition | mean | SD | mean abs | nonzero frac | sign consistency |", "|---|---|---|---|---|---|"]
            L += [f"| {r['coalition']} | {r['mean']:+.3f} | {r['sd']:.3f} | {r['mean_abs']:.3f} | {r['nonzero_frac']:.2f} | {r['sign_cons']:.2f} |" for _, r in t.iterrows()]
            L.append("")
        if sname == "S3":  # full tables are small for S3
            L += [f"All S3 k={k} coefficients (real labels):", "", "| coalition | mean | SD | mean abs | sign consistency |", "|---|---|---|---|---|"]
            L += [f"| {r['coalition']} | {r['mean']:+.3f} | {r['sd']:.3f} | {r['mean_abs']:.3f} | {r['sign_cons']:.2f} |" for _, r in tab.iterrows()]
            L.append("")

# regularisation (S3 = primary, full rank; S5 = original brief, sensitivity)
L += ["## 6. Regularisation comparison (secondary; no tuning, no C grid)", "",
      "- L2: `penalty='l2'`, lbfgs, C=1 (same fits as section 4). L1: `penalty='l1'`, **saga**, C=1, max_iter=1000 (the official wrapper leaves `l1_ratio` at 0.0, which sklearn 1.9.1 warns is inconsistent, but section 2 shows the fit equals native `l1_ratio=1`).",
      "- Effectively unregularised: `penalty=None`, lbfgs, max_iter=1000 (section 2 shows this equals native `C=np.inf`). Separation of the classes is not formally tested; convergence warnings and coefficient magnitudes are reported instead.",
      "- **S3 is the primary regularisation comparison**: it is the only set whose k=2 and k=3 bases are full column rank (section 2c), so coefficient-level statements are at least possible. **S5 (original brief) is a sensitivity analysis**: its basis is structurally rank-deficient, so L1/L2/unregularised behaviour partly reflects how each penalty resolves redundant columns.",
      "- The shuffled-label rows are descriptive references (mean over 10 permutations); they are not a hypothesis test.", ""]
names_of = {"S3": feature_order(SETS["S3"]), "S5": feature_order(SETS["S5"])}


def sparsity_row(label, labs, results, sname, k_):
    names = names_of[sname]
    n = len(names)
    orders = np.array([len(A) for A in coalitions(names, k_)])
    nz, o3, any3, ever, half, sc, mabs, mx, empty = [], [], [], [], [], [], [], [], []
    for r in results:
        tab = coef_table(r, names, k_)
        C = np.abs(r["coefs"]) > TOL
        nz.append(C.sum(1).mean())
        empty.append((C.sum(1) == 0).sum())
        o3.append(np.nanmean([(C[i][orders == 3].sum() / C[i].sum()) if C[i].sum() else np.nan for i in range(len(C))]) if k_ == 3 else np.nan)
        any3.append(C[:, orders == 3].any(1).sum() if k_ == 3 else np.nan)
        ever.append(C.any(0).sum()); half.append((C.mean(0) >= 0.5).sum())
        sc.append(tab["sign_cons"].mean())
        mabs.append([tab.loc[tab["order"] == o, "mean_abs"].mean() if (tab["order"] == o).any() else np.nan for o in (1, 2, 3)])
        mx.append(np.abs(r["coefs"]).max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mabs = np.nanmean(np.array(mabs, float), axis=0)
        o3v = float(np.nanmean(o3)) if k_ == 3 else float("nan")
        any3v = float(np.nanmean(any3)) if k_ == 3 else float("nan")
    f3 = lambda v: "n/a" if not np.isfinite(v) else f"{v:.2f}"
    return (f"| {label} | {labs} | {np.mean(nz):.1f} / {nco(n, k_)} | {np.mean(empty):.1f} / 50 | {f3(o3v)} | {'n/a' if not np.isfinite(any3v) else f'{any3v:.1f} / 50'} | {np.mean(ever):.1f} | {np.mean(half):.1f} | {np.nanmean(sc):.2f} | "
            f"{f3(mabs[0])} / {f3(mabs[1])} / {f3(mabs[2])} | {np.mean(mx):.2f} |")


for sname, role in (("S3", "primary: full-rank basis"), ("S5", "sensitivity: structurally rank-deficient basis, original brief")):
    n = len(names_of[sname])
    L += [f"### {sname} ({role})", "", "#### Real labels", "", HDR]
    for k in (2, 3):
        for reg in ("L2", "L1", "UNREG"):
            L.append(metric_row(f"{sname} k={k} {reg}", nco(n, k), real[(sname, k, reg)]))
    L += ["", f"#### Shuffled labels (mean over {len(SHUFFLE_SEEDS)} permutations; ± is the SD across permutations)", "", NHDR]
    for k in (2, 3):
        for reg in ("L2", "L1", "UNREG"):
            L.append(null_row(f"{sname} k={k} {reg}", nco(n, k), null[(sname, k, reg)]))
    L += ["", f"#### Sparsity, order-3 selection and coefficient size ({sname})", "",
          "`nonzero/fit` = mean number of coefficients with |coef| > tol per fit; `intercept-only fits` = fits in which every coefficient is zero, so all validation scores are tied (ROC-AUC exactly 0.5, PR-AUC = prevalence, Top-20 decided by the slug-order tie rule); `order-3 share` = mean fraction of nonzero terms that are three-way; `fits with any order-3 term` = number of the 50 fits (mean over permutations for shuffled rows) in which at least one three-way coefficient is nonzero; `ever selected` = coefficients nonzero in at least one fit; `>=50% of fits` = coefficients nonzero in at least half of the fits; mean sign consistency over coefficients; mean |coef| per order (order 1 / 2 / 3); max |coef| in any fit (divergence indicator).", "",
          "| condition | labels | nonzero/fit | intercept-only fits (all coefs zero) | order-3 share | fits with any order-3 term | ever selected | selected in >=50% of fits | mean sign cons. | mean abs coef (order1/2/3) | max abs coef |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k_ in (2, 3):
        for reg in ("L2", "L1", "UNREG"):
            L.append(sparsity_row(f"{sname} k={k_} {reg}", "real", [real[(sname, k_, reg)]], sname, k_))
            L.append(sparsity_row(f"{sname} k={k_} {reg}", "shuffled (mean of 10)", null[(sname, k_, reg)], sname, k_))
    L.append("")
    for k_ in (2, 3):
        tab = coef_table(real[(sname, k_, "L1")], names_of[sname], k_)
        t = tab.sort_values("nonzero_frac", ascending=False).head(8)
        L += [f"L1, {sname}, k={k_}: most frequently selected coefficients (real labels; fraction of the 50 fits in which the coefficient is nonzero):", "",
              "| coalition | order | selected in fraction of fits | mean | sign consistency |", "|---|---|---|---|---|"]
        L += [f"| {r['coalition']} | {int(r['order'])} | {r['nonzero_frac']:.2f} | {r['mean']:+.3f} | {r['sign_cons']:.2f} |" for _, r in t.iterrows()]
        L.append("")

# descriptive size of the shuffled-label deviation from the theoretical chance level 0.5
dev = []
for key, lst in null.items():
    m = np.array([x["val_roc"].mean() for x in lst])
    dev.append((key, m.mean() - 0.5, m.std(ddof=1) / np.sqrt(len(m))))
L += ["### Shuffled-label validation ROC relative to the theoretical chance level 0.5 (descriptive)", "",
      "The theoretical chance ROC-AUC is 0.5. The mean over only 10 permutations of 20 positives is a noisy estimate of what this CV procedure returns under no signal. Across all conditions: "
      f"mean validation ROC minus 0.5 ranges from {min(d[1] for d in dev):+.3f} to {max(d[1] for d in dev):+.3f}; the standard error of the mean over the 10 permutations (SD/√10) ranges from {min(d[2] for d in dev):.3f} to {max(d[2] for d in dev):.3f}. "
      f"The sign is negative in {sum(d[1] < 0 for d in dev)} of {len(dev)} conditions and the deviation is within ±2 SE of 0 in {sum(abs(d[1]) <= 2 * d[2] for d in dev)} of {len(dev)} (descriptive counts, not tests; conditions share the same permutations and are not independent). See the 100-permutation calibration below.", ""]

L += ["### Null calibration: 100 additional permutations (S3, L2, C=1; descriptive)", "",
      "Purpose: check how the shuffled-label *cross-validated* metrics behave around their theoretical references (ROC-AUC 0.5, Top-20 = 20·20/198 ≈ 2.0). "
      f"Same pipeline and CV; permutation seeds {CAL_SEEDS[0]}..{CAL_SEEDS[-1]}; each cell is the mean over 100 permutations ± SE (SD/√100). This is a calibration of the control, not a test of any model.", "",
      "| condition | mean fold val ROC | mean OOF ROC | mean OOF AP (prevalence 0.101) | mean OOF Top-20 (reference 2.0) | SD of per-permutation Top-20 |", "|---|---|---|---|---|---|"]
for k in (1, 3):
    f = lambda key: np.array([x[key].mean() for x in cal[k]])
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    L.append(f"| S3 k={k} L2 | {f('val_roc').mean():.3f} ± {se(f('val_roc')):.3f} | {f('oof_roc').mean():.3f} ± {se(f('oof_roc')):.3f} | {f('oof_ap').mean():.3f} ± {se(f('oof_ap')):.3f} | {f('top').mean():.2f} ± {se(f('top')):.2f} | {f('top').std(ddof=1):.2f} |")
L += ["", "Reading: the *theoretical* chance level of an uninformative classifier is 0.5 ROC-AUC / 2.0 Top-20 and is unchanged. The table shows how far a cross-validated estimate on this fixed 198-row dataset lies from it when labels are permuted; a gap that is large relative to its SE would indicate a property of cross-validation on a small, discrete-feature dataset (with a fixed number of positives, positives that fall in a training fold are missing from the validation fold), not a different chance level.", ""]
L += ["## 7. Interpretation", ""]
L += [INTERP.read_text() if INTERP.exists() else "_(interpretation file not written yet)_", "",
      "## 8. Limits of this experiment", "",
      "- 20 positives (4 per validation fold): every metric is wide; SDs across folds/repeats are not confidence intervals (repeats reuse the same 198 rows; training sets of different folds overlap).",
      "- S3/S5 contain almost no signal (the earlier plain-LR Model A had OOF ROC 0.54, Top-20 3.3 vs chance 2.0). Three of the five features are (near-)binary and `founder_previous_yc` is 1 in only 5 of 193 rows, so most interaction columns are almost entirely zero.",
      "- All features come from 2026 snapshots (caveat tier). The target is dominated by 11 article-covered startups plus 9 attention-driven positives; TLD is plausibly tied to Google/domain measurement.",
      "- MinMax is fitted on training folds; validation values can fall outside [0,1] and nothing clips them (official behaviour).",
      "- The shuffled-label control is descriptive, not inferential: 10 permutations, one dataset, overlapping folds. It shows what the metrics look like when labels carry no information; it gives no p-values or confidence intervals. The theoretical chance ROC-AUC is 0.5; deviations of the 10-permutation mean from 0.5 are treated as sampling variability.", ""]
REPORT.write_text("\n".join(L))
print("wrote", REPORT)
assert git(OFFICIAL, "status", "--short") == "", "official repository was modified"
