"""Build the W26 feature audit, the model dataset and the dataset report.

Deterministic and offline: reads only files in this repository (plus the repository's own git history
for the deleted March-2026 GitHub/Hacker News/Reddit artefacts). No network, no external data,
no imputation, no model training.

Outputs
  data/processed/w26_feature_audit.csv      every startup-level feature found, with temporal/leakage verdicts
  data/processed/w26_model_dataset.csv      only features admissible at the operational cutoff + targets
  reports/w26_dataset_report.md             human-readable report (generated; numbers come from the data)

Run from anywhere:  python scripts/build_w26_dataset.py
"""
import ast
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "w26_dataset_report.md"

CUTOFF = "2025-10-31"          # operational cutoff = end of the notebook baseline window (NOT a verified YC deadline)
HISTORY_COMMIT = "8e7ae11"     # initial commit: still contains data/features and data/raw (deleted 3 minutes later)
K = 20                         # Top-K used by the notebook (hard-coded 20 in cells 3, 40, 41)
EPS = 1e-9

# ----------------------------------------------------------------------------------------------
# Notebook constants (extracted from the original notebook, never retyped)
# ----------------------------------------------------------------------------------------------
NB = json.loads((ROOT / "YC_W26_Google_Mentions.ipynb").read_text())
nb_src = lambda i: "".join(NB["cells"][i]["source"])


def nb_raw_mentions(i):
    m = re.search(r'raw_data = """(.*?)"""', nb_src(i), re.S)
    return {l.split(": ")[0].strip(): float(l.split(": ")[1]) for l in m.group(1).strip().splitlines()}


def nb_dict(i, name):
    s = nb_src(i)
    m = re.search(rf"^{name}\s*=\s*\{{", s, re.M)
    start, depth = m.end() - 1, 0
    for j in range(start, len(s)):
        depth += s[j] == "{"
        depth -= s[j] == "}"
        if depth == 0:
            return ast.literal_eval(s[start : j + 1])


# guard: the formulas this script re-implements must still be the ones in the notebook
assert "row[metric] * weight" in nb_src(12 + 1) and "max(weighted_metrics.values())" in nb_src(13)
assert "(1 + 10 * mom)" in nb_src(15) and "max(velocity_multiplier, 0.1)" in nb_src(15)
assert "google_mentions'] * 0.05" in nb_src(20) and "max(axis=1)" in nb_src(20)

DURING_NB = nb_raw_mentions(2)       # 198 hard-coded during-batch counts (cell 2)
PRE_CLEANED_NB = nb_raw_mentions(39)  # 198 hard-coded pre-application counts (cell 39)
SIGNAL_WEIGHTS = nb_dict(12, "vc_weights")
STARTUP_TO_DOMAIN = nb_dict(20, "startup_to_domain")
ATTENTION_WEIGHT, MOM_FACTOR, MOM_FLOOR = 0.05, 10, 0.1

# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def git_show(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{HISTORY_COMMIT}:{path}"], stderr=subprocess.DEVNULL)


def git_ls(prefix):
    out = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", HISTORY_COMMIT]).decode()
    return [p for p in out.splitlines() if p.startswith(prefix)]


def domain_of(url):
    """Same function the collection scripts used (yc_serpapi*.py), so keys match yc_mentions*.csv."""
    return urlparse(url).netloc.replace("www.", "") if url else None


def slug_of(yc_url):
    return yc_url.rstrip("/").split("/")[-1]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def split_domain(d):
    parts = d.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return ".".join(parts[-2:]), parts[-3]
    return parts[-1], (parts[-2] if len(parts) >= 2 else parts[0])


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "\\|").replace("\n", " ") for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# 1. Canonical universe
# ----------------------------------------------------------------------------------------------
csv_rows = list(csv.DictReader(open(ROOT / "yc_w26_startups.csv")))
csv_by_slug = {slug_of(r["yc_url"]): r for r in csv_rows}
all_companies = json.load(open(ROOT / "yc_all_companies.json"))
json_by_slug = {x["slug"]: x for x in all_companies if x["batch"] == "W26"}
assert len(csv_by_slug) == len(csv_rows) and len(json_by_slug) == sum(1 for x in all_companies if x["batch"] == "W26")

csv_only = set(csv_by_slug) - set(json_by_slug)
json_only = set(json_by_slug) - set(csv_by_slug)
renames = {}  # accepted only when the CSV name states "(formerly X)" and exactly one JSON-only record is named X
for c in sorted(csv_only):
    m = re.search(r"\(formerly ([^)]+)\)", csv_by_slug[c]["name"], re.I)
    if not m:
        continue
    cands = [j for j in sorted(json_only) if norm(json_by_slug[j]["name"]) == norm(m.group(1)) or norm(j) == norm(m.group(1))]
    if len(cands) == 1:
        renames[c] = cands[0]

entries, used_json = [], set()
for c in sorted(csv_by_slug):
    if c in json_by_slug:
        entries.append((c, c, c, "slug_match")); used_json.add(c)
    elif c in renames:
        entries.append((c, c, renames[c], "rename_rule(csv name says formerly)")); used_json.add(renames[c])
    else:
        entries.append((c, c, None, "csv_only"))
for j in sorted(set(json_by_slug) - used_json):
    entries.append((j, None, j, "json_only"))

# ----------------------------------------------------------------------------------------------
# 1b. Offline founder-history features (from yc_all_companies.json only)
#     A W26 founder has "previous YC history" when the same LinkedIn profile slug appears among the founders of a company
#     in an earlier batch that clearly ended before the cutoff. The repository has no batch dates, so the calendar below is an
#     ASSUMPTION about YC's schedule: W = Jan-Mar, P (spring; the repo's spring_2026_batch folder holds P26) = Mar-Jun,
#     S = Jun-Sep, F = Oct-Dec. A batch counts only if it ended by 2025-09, i.e. clearly before the 2025-10-31 cutoff.
#     Matching is by LinkedIn profile slug only (no name matching, which would be ambiguous).
# ----------------------------------------------------------------------------------------------
SEASON_END_MONTH = {"W": 3, "P": 6, "X": 6, "S": 9, "F": 12}
HISTORY_BEFORE = (2025, 9)


def linkedin_key(url):
    m = re.search(r"linkedin\.com/in/([^/?#]+)", unquote(url or "").strip().lower())
    return m.group(1) if m else None


def batch_end(code):
    m = re.fullmatch(r"([WXPSF])(\d{2})", code or "")
    return (2000 + int(m.group(2)), SEASON_END_MONTH[m.group(1)]) if m else None


w26_keys = defaultdict(set)  # linkedin slug -> W26 startup slugs
for slug_, x_ in json_by_slug.items():
    for f_ in x_.get("founders") or []:
        k_ = linkedin_key(f_.get("linkedin_url"))
        if k_:
            w26_keys[k_].add(slug_)
history_matches = []  # (linkedin slug, batch, startup slug, batch end)
for x_ in all_companies:
    if x_["batch"] == "W26":
        continue
    for f_ in x_.get("founders") or []:
        k_ = linkedin_key(f_.get("linkedin_url"))
        if k_ in w26_keys:
            history_matches.append((k_, x_["batch"], x_["slug"], batch_end(x_["batch"])))
prev_hist, unknown_hist, later_hist = defaultdict(set), defaultdict(set), defaultdict(set)
for k_, b_, s_, e_ in history_matches:
    if e_ is None:
        unknown_hist[k_].add(s_)
    elif e_ <= HISTORY_BEFORE:
        prev_hist[k_].add((b_, s_))
    else:
        later_hist[k_].add((b_, s_))


def startups_flagged(threshold):
    return len({sl for k_, b_, s_, e_ in history_matches if e_ and e_ <= threshold for sl in w26_keys[k_]})


fh_stats = {
    "w26_founders": sum(len(x_.get("founders") or []) for x_ in json_by_slug.values()),
    "w26_founders_with_key": len(w26_keys),
    "founders_matched_clearly_before": len(prev_hist),
    "startups_flagged": startups_flagged(HISTORY_BEFORE),
    "founders_matched_unknown_batch": len(unknown_hist),
    "founders_matched_not_clearly_before": len(later_hist),
    "previous_batches": Counter(b_ for v in prev_hist.values() for b_, _ in v),
    "founders_in_more_than_one_w26_startup": sum(1 for v in w26_keys.values() if len(v) > 1),
    "sens_without_S25_P25_F24_F25": startups_flagged((2024, 9)),
    "sens_including_F25": startups_flagged((2025, 12)),
}

# ----------------------------------------------------------------------------------------------
# 2. Source tables
# ----------------------------------------------------------------------------------------------
early = pd.read_csv(ROOT / "yc_mentions_early.csv").set_index("domain")["google_mentions"]
during_csv = pd.read_csv(ROOT / "yc_mentions.csv").set_index("domain")["google_mentions"]
traction_csv = pd.read_csv(ROOT / "yc_w26_traction.csv")
predemo_csv = pd.read_csv(ROOT / "yc_w26_pre_demo_scores.csv")
SIGNAL_COLS = list(SIGNAL_WEIGHTS)

# Velocity Score exactly as in the notebook (cells 13 and 15), checked against the committed CSV
traction_csv["base_score"] = (traction_csv[SIGNAL_COLS].mul(pd.Series(SIGNAL_WEIGHTS))).max(axis=1)
traction_csv["multiplier"] = (1 + MOM_FACTOR * traction_csv["mom_growth"]).clip(lower=MOM_FLOOR)
traction_csv["velocity_recomputed"] = traction_csv["base_score"] * traction_csv["multiplier"]
assert ((traction_csv["velocity_recomputed"] - traction_csv["Velocity_Score"]).abs() < EPS).all(), "Velocity mismatch vs yc_w26_traction.csv"
traction_csv["domain"] = traction_csv["Startup"].map(STARTUP_TO_DOMAIN)
assert traction_csv["domain"].notna().all() and traction_csv["domain"].is_unique
traction_by_domain = traction_csv.set_index("domain")

# git-history artefacts (March 2026 pipeline)
try:
    gh_metrics = json.loads(git_show("data/features/github_metrics.json"))
    hn_metrics = json.loads(git_show("data/features/hn_metrics.json"))
    hn_raw = {p.split("/")[-1][:-5]: json.loads(git_show(p))["stories"] for p in git_ls("data/raw/hn/")}
    reddit_log_lines = git_show("data/features/reddit_scrape.log").decode().splitlines()
    GIT_OK = True
except Exception as exc:  # history not available in this clone
    gh_metrics, hn_metrics, hn_raw, reddit_log_lines, GIT_OK = {}, {}, {}, [], False
    print("WARNING: git history unavailable:", exc)
yc_oss_slugs = set(gh_metrics)

# P26 LinkedIn evidence (never W26)
linkedin_p26 = json.load(open(ROOT / "spring_2026_batch" / "yc_p26_dataset_linkedin-profile-scraper_2026-05-17_13-54-11-106.json"))

# ----------------------------------------------------------------------------------------------
# 3. Universe frame (all candidate columns; model columns are selected later)
# ----------------------------------------------------------------------------------------------
rows = []
matched_yc_oss = set()
for canon, c_slug, j_slug, resolution in entries:
    c, j = csv_by_slug.get(c_slug), json_by_slug.get(j_slug)
    r = {
        "canonical_slug": canon, "name": (c or j)["name"], "identity_resolution": resolution,
        "csv_slug": c_slug, "json_slug": j_slug,
        "in_yc_w26_startups_csv": int(c is not None), "in_all_companies_json": int(j is not None),
        "csv_domain": domain_of(c["website"]) if c else None, "json_domain": domain_of(j["website"]) if j else None,
    }
    r["domain_differs"] = int(c is not None and j is not None and r["csv_domain"] != r["json_domain"])
    site = r["csv_domain"] or r["json_domain"]
    r["site_domain_source"] = "csv" if r["csv_domain"] else "json"
    r["x_tld"], label = split_domain(site)
    r["x_site_name_length"] = len(label)
    d = r["csv_domain"]
    # YC-native (JSON snapshot)
    fs = (j or {}).get("founders") or []
    r["founder_n_founders"] = len(fs) if j else None
    r["x_founder_n_linkedin_url"] = sum(1 for f in fs if f.get("linkedin_url")) if j else None
    if j:
        keys_ = [linkedin_key(f.get("linkedin_url")) for f in fs]
        prevs_ = [prev_hist.get(k_, set()) for k_ in keys_ if k_]
        r["n_previous_yc_founders"] = sum(1 for p_ in prevs_ if p_)
        r["n_previous_yc_startups"] = len(set().union(*prevs_)) if prevs_ else 0
        r["founder_previous_yc"] = int(r["n_previous_yc_founders"] > 0)
        r["x_founders_without_url_key"] = sum(1 for k_ in keys_ if k_ is None)
    else:
        r["n_previous_yc_founders"] = r["n_previous_yc_startups"] = r["founder_previous_yc"] = r["x_founders_without_url_key"] = None
    r["x_founder_any_title"] = int(any(f.get("title") for f in fs)) if j else None
    r["x_founder_any_name"] = int(any(f.get("name") for f in fs)) if j else None
    r["company_one_liner"] = (j or {}).get("one_liner") if j else None
    r["company_one_liner_n_words"] = len(j["one_liner"].split()) if j and j.get("one_liner") else None
    tags = (j or {}).get("tags") or []
    r["company_n_tags"] = len(tags) if j else None
    r["company_tags"] = "|".join(tags) if tags else None
    r["x_status"] = (j or {}).get("status") if j else None
    r["x_name_formerly_marker"] = int("formerly" in r["name"].lower())
    # Google mentions
    r["attn_google_mentions_pre_raw"] = early.get(d) if d else None
    r["ref_attn_google_mentions_pre_cleaned_nb"] = PRE_CLEANED_NB.get(d) if d else None
    r["x_mentions_during_csv"] = during_csv.get(d) if d else None
    r["x_mentions_during_nb"] = DURING_NB.get(d) if d else None
    # traction (post-hoc, target side)
    t = traction_by_domain.loc[d] if d in traction_by_domain.index else None
    for col in SIGNAL_COLS + ["mom_growth"]:
        r[f"x_traction_{col}"] = t[col] if t is not None else None
    r["x_velocity_score"] = t["velocity_recomputed"] if t is not None else None
    # GitHub / HN from March-2026 pipeline (yc-oss slug must equal a repo slug; no guessing)
    ys = next((s for s in (c_slug, j_slug) if s and s in yc_oss_slugs), None)
    if ys:
        matched_yc_oss.add(ys)
    r["x_ycoss_slug"] = ys
    r["x_github_repo_found"] = int(bool(gh_metrics[ys].get("repo"))) if ys else None
    r["x_github_fork_growth_2026q1"] = gh_metrics[ys].get("fork_growth") if ys else None
    r["x_hn_score_alltime"] = hn_metrics[ys].get("hn_score") if ys else None
    hits = hn_raw.get(ys, []) if ys else None
    r["x_hn_n_hits_alltime"] = len(hits) if ys else None
    r["x_hn_n_hits_before_cutoff"] = sum(1 for h in hits if h["created_at"][:10] <= CUTOFF) if ys else None
    r["x_hn_n_hits_after_cutoff"] = sum(1 for h in hits if h["created_at"][:10] > CUTOFF) if ys else None
    rows.append(r)

U = pd.DataFrame(rows).sort_values("canonical_slug").reset_index(drop=True)
N_UNIVERSE = len(U)
assert U["canonical_slug"].is_unique
assert all(d in early.index for d in U["csv_domain"].dropna()), "a CSV domain is missing from yc_mentions_early.csv"
unmatched_yc_oss = sorted(yc_oss_slugs - matched_yc_oss)

# ----------------------------------------------------------------------------------------------
# 4. Target: the repository's Pre-Demo Day Score (notebook cells 12-20)
#    defined  = a during-batch measurement exists in notebook cell 2 (value may be 0)
#    score>0  = defined and max(Velocity, 0.05*mentions) > 0   (the rows kept by the notebook / pre_demo CSV)
# ----------------------------------------------------------------------------------------------
U["target_defined"] = U["x_mentions_during_nb"].notna().astype(int)
vel = U["x_velocity_score"].fillna(0.0)
score = pd.concat([vel, ATTENTION_WEIGHT * U["x_mentions_during_nb"]], axis=1).max(axis=1)
U["target_predemo_score"] = score.where(U["target_defined"] == 1)
assert set(traction_csv["domain"]) <= set(U.loc[U["target_defined"] == 1, "csv_domain"]), "a traction startup has no target"

# cross-check against the committed artefact
pos = U[U["target_predemo_score"] > 0]
chk = predemo_csv.merge(pos[["csv_domain", "target_predemo_score"]], left_on="domain", right_on="csv_domain", how="outer", indicator=True)
assert (chk["_merge"] == "both").all(), "score>0 set differs from yc_w26_pre_demo_scores.csv"
assert ((chk["hybrid_score"] - chk["target_predemo_score"]).abs() < EPS).all()


def top_k_flags(values, tiebreak, k):
    """1 for the k highest values (ties broken by `tiebreak` ascending), 0 for other defined rows, NaN for undefined.
    Returns (flags, boundary_tie)."""
    d = pd.DataFrame({"v": values, "t": tiebreak}).dropna(subset=["v"]).sort_values(["v", "t"], ascending=[False, True])
    boundary_tie = bool(len(d) > k and abs(d["v"].iloc[k - 1] - d["v"].iloc[k]) < EPS)
    flags = pd.Series(float("nan"), index=values.index)
    flags.loc[d.index] = 0
    flags.loc[d.index[:k]] = 1
    return flags, boundary_tie


U["target_top20"], TARGET_BOUNDARY_TIE = top_k_flags(U["target_predemo_score"], U["csv_domain"], K)
U["ref_baseline_top20"], BASELINE_BOUNDARY_TIE = top_k_flags(
    U["ref_attn_google_mentions_pre_cleaned_nb"].where(U["ref_attn_google_mentions_pre_cleaned_nb"] > 0), U["csv_domain"], K)
# rows with a cleaned value of 0 are measured-but-inactive: 0, not NaN
U.loc[U["ref_attn_google_mentions_pre_cleaned_nb"].notna() & U["ref_baseline_top20"].isna(), "ref_baseline_top20"] = 0

counts = {
    "universe": N_UNIVERSE,
    "target_defined": int(U["target_defined"].sum()),
    "target_score_gt0": int((U["target_predemo_score"] > 0).sum()),
    "target_score_eq0": int((U["target_predemo_score"] == 0).sum()),
    "target_undefined": int(U["target_predemo_score"].isna().sum()),
    "target_top20": int(U["target_top20"].sum()),
    "top20_traction_driven": int(((U["target_top20"] == 1) & (U["x_velocity_score"] > ATTENTION_WEIGHT * U["x_mentions_during_nb"])).sum()),
}
counts["top20_attention_driven"] = counts["target_top20"] - counts["top20_traction_driven"]
positive_scores = U.loc[U["target_predemo_score"] > 0, "target_predemo_score"]
counts["tied_score_values_gt0"] = int(positive_scores.duplicated(keep=False).sum())
baseline_hits = int(((U["ref_baseline_top20"] == 1) & (U["target_top20"] == 1)).sum())
is_trac = U["x_velocity_score"].notna() & (U["target_predemo_score"] > 0)
TRACTION_ALWAYS_OUTRANKS = bool(U.loc[is_trac, "target_predemo_score"].min() > U.loc[~is_trac, "target_predemo_score"].max())
dup_at_boundary = bool(U.loc[U["target_top20"] == 1, "target_predemo_score"].min() in set(U.loc[U["target_top20"] == 0, "target_predemo_score"]))

# sensitivity: target recomputed from the committed yc_mentions.csv instead of the notebook's hard-coded values
alt_mentions = U["x_mentions_during_csv"]
alt_score = pd.concat([vel, ATTENTION_WEIGHT * alt_mentions.fillna(0)], axis=1).max(axis=1).where(U["csv_domain"].notna())
alt_flags, _ = top_k_flags(alt_score, U["csv_domain"], K)
alt_overlap = int(((alt_flags == 1) & (U["target_top20"] == 1)).sum())
during_diff = U[U["x_mentions_during_csv"].fillna(-1).ne(U["x_mentions_during_nb"].fillna(-1)) & U["csv_domain"].notna()]
n_during_diff = len(during_diff)
n_during_diff_nb_zero = int((during_diff["x_mentions_during_nb"] == 0).sum())

# ----------------------------------------------------------------------------------------------
# 5. Feature audit specification (single source of truth for audit + model selection + report)
# ----------------------------------------------------------------------------------------------
CUT_NOTE = f"operational cutoff {CUTOFF}"
SPECS = []


def spec(feature, family, source, collected, pre, post, ts, leak, tier, model_col, why, data=None, n_source=None, unit="startup",
         units=None, units_nn=None):
    SPECS.append(dict(feature=feature, family=family, source=source, collected=collected, pre=pre, post=post, ts=ts, leak=leak,
                      tier=tier, model_col=model_col, why=why, data=data, n_source=n_source, unit=unit, units=units, units_nn=units_nn))


J_SNAP = "snapshot committed 2026-05-06 (f11c801); collection date and origin of founders not recorded"
# --- identity (meta) ---
spec("slug / name / website (yc_w26_startups.csv)", "identity", "yc_w26_startups.csv", "committed 2026-03-30 (8e7ae11); origin not recorded",
     "conditional", False, True, "none", "meta", None, "Identifier columns; list membership only exists after acceptance.", U["in_yc_w26_startups_csv"].where(U["in_yc_w26_startups_csv"] == 1), 198)
spec("slug / name / website (yc_all_companies.json W26)", "identity", "yc_all_companies.json", J_SNAP,
     "conditional", False, True, "none", "meta", None, "Identifier columns; same membership caveat.",U["in_all_companies_json"].where(U["in_all_companies_json"] == 1), 199)
# --- company characteristics ---
spec("one_liner", "company characteristics", "yc_all_companies.json[W26].one_liner", J_SNAP, "conditional", False, True, "none", "caveat",
     "company_one_liner", "Static descriptor; may have been edited after the cutoff, edit history unknown. Kept for text features (not yet encoded).", U["company_one_liner"], 199)
spec("one_liner word count", "company characteristics", "derived from one_liner", J_SNAP, "conditional", False, True, "none", "caveat",
     "company_one_liner_n_words", "Same caveat as one_liner.", U["company_one_liner_n_words"], 199)
spec("tags", "company characteristics", "yc_all_companies.json[W26].tags", J_SNAP, "conditional", False, True, "none", "caveat",
     "company_tags", "Low coverage (empty list = no tag assigned, kept missing). Tags may be assigned by YC after acceptance.", U["company_tags"], 199)
spec("number of tags", "company characteristics", "derived from tags", J_SNAP, "conditional", False, True, "none", "caveat",
     "company_n_tags", "0 means an empty tag list in the JSON (observed), NaN means no JSON record.", U["company_n_tags"], 199)
spec("website TLD", "company characteristics", "derived from website (csv, else json)", "see website sources", "conditional", False, True, "none", "caveat",
     "company_site_tld", f"Domain differs between csv and json for {int(U['domain_differs'].sum())} startups (possible rebrands); csv domain preferred, json fallback.", U["x_tld"], 204)
spec("website name length", "company characteristics", "derived from website (csv, else json)", "see website sources", "conditional", False, True, "none", "caveat",
     "company_site_name_length", "Length of the second-level label. Same domain caveat.", U["x_site_name_length"], 204)
spec("status", "company characteristics", "yc_all_companies.json[W26].status", J_SNAP, "unknown", False, True, "possible", "excluded", None,
     "Empty for every W26 record (and every other batch); would be an outcome-like field if populated.", U["x_status"], 199)
spec("name contains 'formerly'", "company characteristics", "derived from yc_w26_startups.csv name", "committed 2026-03-30", "unknown", False, True, "possible", "excluded", None,
     "Marks a rename of unknown date; the name snapshot post-dates the cutoff.", U["x_name_formerly_marker"].where(U["in_yc_w26_startups_csv"] == 1), 198)
# --- founders ---
spec("number of founders", "founder / human capital", "yc_all_companies.json[W26].founders (length)", J_SNAP, "conditional", False, True, "none", "caveat",
     "founder_n_founders", "Founder list is a post-acceptance snapshot; team size is unlikely to change but is not timestamped.", U["founder_n_founders"], 199)
FH_WHY = ("Derived offline from yc_all_companies.json by LinkedIn-slug match to founders of earlier batches that ended by 2025-09 "
          "(batch calendar is an assumption, see report). Founders with no URL or only matches in batches of unknown date are not counted, so this is a lower bound.")
spec("any W26 founder previously a YC founder", "founder / human capital", "yc_all_companies.json (earlier batches)", J_SNAP, "yes", False, False, "none", "clean",
     "founder_previous_yc", FH_WHY, U["founder_previous_yc"], 199)
spec("number of W26 founders with previous YC history", "founder / human capital", "yc_all_companies.json (earlier batches)", J_SNAP, "yes", False, False, "none", "clean",
     "n_previous_yc_founders", FH_WHY, U["n_previous_yc_founders"], 199)
spec("number of distinct previous YC startups of the team", "founder / human capital", "yc_all_companies.json (earlier batches)", J_SNAP, "yes", False, False, "none", "clean",
     "n_previous_yc_startups", FH_WHY, U["n_previous_yc_startups"], 199)
spec("founder LinkedIn URL (any per startup)", "founder / human capital", "yc_all_companies.json[W26].founders[].linkedin_url", J_SNAP, "conditional", False, True, "none", "excluded", None,
     "An identifier, not information; near-constant (see founder-level coverage). Profile content is not in the repository for W26.",
     U["x_founder_n_linkedin_url"].where(U["x_founder_n_linkedin_url"] > 0), 199, "startup",
     )
spec("founder LinkedIn URL (founder level)", "founder / human capital", "yc_all_companies.json[W26].founders[].linkedin_url", J_SNAP, "conditional", False, True, "none", "excluded", None,
     "Same as above, founder-level coverage.", None, 199, "founder",
     units=sum(len(x.get("founders") or []) for x in json_by_slug.values()),
     units_nn=sum(1 for x in json_by_slug.values() for f in (x.get("founders") or []) if f.get("linkedin_url")))
spec("founder title", "founder / human capital", "yc_all_companies.json[W26].founders[].title", J_SNAP, "conditional", False, True, "none", "excluded", None,
     "Almost always the literal 'Founder'; formatting, not role information.", None, 199, "founder",
     units=sum(len(x.get("founders") or []) for x in json_by_slug.values()),
     units_nn=sum(1 for x in json_by_slug.values() for f in (x.get("founders") or []) if f.get("title")))
spec("founder name", "founder / human capital", "yc_all_companies.json[W26].founders[].name", J_SNAP, "conditional", False, True, "none", "excluded", None,
     "Personal data with no predictive content here; deliberately kept out of the model table. No demographic inference from names.", None, 199, "founder",
     units=sum(len(x.get("founders") or []) for x in json_by_slug.values()),
     units_nn=sum(1 for x in json_by_slug.values() for f in (x.get("founders") or []) if f.get("name")))
spec("LinkedIn profile content (experience, education, followers, current position)", "founder / human capital",
     "spring_2026_batch/yc_p26_dataset_linkedin-profile-scraper_2026-05-17_13-54-11-106.json (P26 only)", "2026-05-17 (file name), P26 founders only",
     "no", True, False, "possible", "excluded", None,
     f"No W26 profile content exists in the repository ({len(linkedin_p26)} profiles, all P26). If collected later, current position/followers are post-cutoff snapshots.",
     pd.Series([None] * N_UNIVERSE, dtype="float64"), 0)
# --- Google mention variables ---
spec("Google mentions, pre-application (raw)", "traction / attention", "yc_mentions_early.csv (yc_serpapi_early.py)", "SerpAPI window 2025-08-17..2025-10-31; run date unknown (<2026-03-30)",
     "conditional", False, False, "none", "caveat", "attn_google_mentions_pre_raw",
     "Window ends at the cutoff, but retrieval happened later, Google counts are estimates, and domains are 2026 websites. Raw values kept (heavy tails, not transformed).",
     U["attn_google_mentions_pre_raw"], 198)
spec("Google mentions, pre-application (cleaned, notebook cell 39)", "traction / attention", "YC_W26_Google_Mentions.ipynb cell 39 (hard-coded)", "same window; cleaning date and rule not recorded",
     "conditional", False, False, "possible", "baseline_reference", "ref_attn_google_mentions_pre_cleaned_nb",
     "Researcher-curated with the target visible and an undocumented rule; not an independent predictor. Kept only as a reference column so models can be compared with the published 14/20 baseline.",
     U["ref_attn_google_mentions_pre_cleaned_nb"], 198)
spec("Google mentions, during batch (csv)", "traction / attention", "yc_mentions.csv (yc_serpapi.py)", "SerpAPI cd_min 2026-01-01, no end date; run date unknown (<2026-03-30)",
     "no", True, False, "obvious", "target_side", None, "After the cutoff and a component of the target.", U["x_mentions_during_csv"], 198)
spec("Google mentions, during batch (notebook cell 2)", "traction / attention", "YC_W26_Google_Mentions.ipynb cell 2 (hard-coded)", "same window as above",
     "no", True, False, "obvious", "target_side", None, "Target component (used by the notebook and by yc_w26_pre_demo_scores.csv).", U["x_mentions_during_nb"], 198)
# --- traction (hand-entered, post-hoc) ---
for col in SIGNAL_COLS + ["mom_growth"]:
    spec(f"traction: {col}", "traction / attention", "yc_w26_traction.csv (Lobster Capital article, typed by hand)", "committed 2026-03-31; article date unknown",
         "no", True, True, "obvious", "target_side", None,
         "Post-hoc third-party traction; present only for the 11 startups the article covered (the other 193 are unknown, not zero).", U[f"x_traction_{col}"], 11)
spec("Velocity Score", "traction / attention", "derived (notebook cells 13 and 15); equals yc_w26_traction.csv.Velocity_Score", "committed 2026-03-31",
     "no", True, True, "obvious", "target_side", None, "Target component.", U["x_velocity_score"], 11)
spec("Pre-Demo Day Score", "target", "derived (notebook cell 20); matches yc_w26_pre_demo_scores.csv.hybrid_score", "committed 2026-03-31",
     "no", True, True, "obvious", "target", "target_predemo_score", "The target itself (defined for every startup with a during-batch measurement; > 0 for the rows of yc_w26_pre_demo_scores.csv).", U["target_predemo_score"], 184)
spec("Top-20 label", "target", "derived from the Pre-Demo Day Score", "committed 2026-03-31 (score)",
     "no", True, True, "obvious", "target", "target_top20", f"1 for the {K} highest scores among defined rows, 0 for other defined rows, NaN where the target is undefined.", U["target_top20"], 184)
spec("Baseline Top-20 flag (cleaned pre-application mentions)", "traction / attention", "derived from notebook cell 39 values", "same window as cell 39",
     "conditional", False, False, "possible", "baseline_reference", "ref_baseline_top20", "Reference for comparing models with the published 14/20 baseline; not a predictor.", U["ref_baseline_top20"], 198)
# --- GitHub / HN / Reddit / Twitter / Similarweb / funding ---
spec("GitHub repo found", "developer / community", "git 8e7ae11:data/features/github_metrics.json (extract_github_metrics.py)", "2026-03-16 (HN/Reddit logs of the same run); fork window to 2026-03-24",
     "no", True, False, "possible", "excluded", None,
     "Collected after the cutoff; repo = top search hit by name sorted by forks (found for only a few startups, several visibly unrelated). Absence is not 'no repo'.", U["x_github_repo_found"], 132)
spec("GitHub fork growth 2026-01-01..2026-03-24", "developer / community", "git 8e7ae11:data/features/github_metrics.json", "window 2026-01-01..2026-03-24",
     "no", True, False, "obvious", "excluded", None, "Measured entirely after the cutoff.", U["x_github_fork_growth_2026q1"], 132)
spec("Hacker News score (all-time search)", "network / ecosystem", "git 8e7ae11:data/features/hn_metrics.json (extract_hn_metrics.py)", "2026-03-16 01:46-01:50",
     "unknown", True, True, "possible", "excluded", None,
     "No date window; mixes pre- and post-cutoff hits. Zeros are unreliable: the domain filter compares a URL with 'https://' to the hit URL (126 of 132 runs returned nothing).", U["x_hn_score_alltime"], 132)
spec("HN hits created on/before cutoff", "network / ecosystem", "git 8e7ae11:data/raw/hn/*.json (created_at per hit)", "run 2026-03-16; hits carry created_at",
     "yes", False, False, "none", "excluded", None,
     "Timing is verifiable per hit, but coverage is almost nil and zero vs missing cannot be separated (same search bug); only a few startups have any raw file.", U["x_hn_n_hits_before_cutoff"], 132)
spec("HN hits created after cutoff", "network / ecosystem", "git 8e7ae11:data/raw/hn/*.json", "run 2026-03-16", "no", True, False, "possible", "excluded", None,
     "Post-cutoff attention.", U["x_hn_n_hits_after_cutoff"], 132)
spec("Reddit metrics", "network / ecosystem", "git 8e7ae11:data/features/reddit_scrape.log (extract_reddit_metrics.py)", "2026-03-16 09:34",
     "unknown", False, True, "none", "excluded", None,
     f"No data: the run stopped after {len(reddit_log_lines)} log lines with HTTP 403 errors; reddit_metrics.json does not exist.", pd.Series([None] * N_UNIVERSE, dtype="float64"), 0)
spec("Tweets / ARR / funding extracted by LLM", "funding / investor", "scripts/scrape/extract_tweets_apify.py + scripts/processing/process_twitter_llm.py", "never run in the repository",
     "unknown", False, True, "possible", "excluded", None,
     "Scripts only; no output files. The scraper has no date filter, so it would not be point-in-time anyway.", pd.Series([None] * N_UNIVERSE, dtype="float64"), 0)
spec("Similarweb monthly visits", "traction / attention", "yc_traffic.py ... yc_traffic4.py", "never saved (no yc_w26_traffic.csv)",
     "unknown", False, True, "none", "excluded", None, "Scripts only; no output files.", pd.Series([None] * N_UNIVERSE, dtype="float64"), 0)
spec("Funding / investor signals", "funding / investor", "none", "n/a", "unknown", False, True, "none", "excluded", None,
     "No funding or investor field exists anywhere in the repository for W26.", pd.Series([None] * N_UNIVERSE, dtype="float64"), 0)


def build_audit():
    out = []
    for s in SPECS:
        if s["unit"] == "founder":
            n_units, n_nn, nz = s["units"], s["units_nn"], None
        else:
            ser = s["data"]
            n_units, n_nn = N_UNIVERSE, int(ser.notna().sum())
            nz = int((pd.to_numeric(ser, errors="coerce").fillna(0) != 0).sum()) if pd.api.types.is_numeric_dtype(ser) else None
        out.append({
            "feature": s["feature"], "family": s["family"], "tier": s["tier"], "in_model_dataset": int(s["tier"] in ("clean", "caveat", "target", "baseline_reference")),
            "model_column": s["model_col"], "source": s["source"], "collected_or_observed": s["collected"], "unit": s["unit"],
            "n_source_rows": s["n_source"], "n_units": n_units, "n_non_missing": n_nn, "n_missing": n_units - n_nn,
            "pct_non_missing": round(100 * n_nn / n_units, 1), "n_nonzero": nz,
            "available_pre_cutoff": s["pre"], "post_cutoff": s["post"], "timestamp_unestablished": s["ts"],
            "target_leakage": s["leak"], "rationale": s["why"],
        })
    return pd.DataFrame(out)


audit = build_audit()

# ----------------------------------------------------------------------------------------------
# 6. Model dataset
# ----------------------------------------------------------------------------------------------
META = ["canonical_slug", "name", "identity_resolution", "csv_slug", "json_slug", "in_yc_w26_startups_csv", "in_all_companies_json",
        "csv_domain", "json_domain", "domain_differs", "site_domain_source", "target_defined"]
FAMILIES = {
    "founder / human capital": ["founder_n_founders", "founder_previous_yc", "n_previous_yc_founders", "n_previous_yc_startups"],
    "company characteristics": ["company_one_liner", "company_one_liner_n_words", "company_tags", "company_n_tags", "company_site_tld", "company_site_name_length"],
    "funding / investor signals": [],
    "network / ecosystem signals": [],
    "traction / attention signals": ["attn_google_mentions_pre_raw"],
}
PREDICTORS = [c for cols in FAMILIES.values() for c in cols]
REFERENCE = ["ref_attn_google_mentions_pre_cleaned_nb", "ref_baseline_top20"]
TARGETS = ["target_predemo_score", "target_top20"]

M = U.rename(columns={"x_tld": "company_site_tld", "x_site_name_length": "company_site_name_length"})[META + PREDICTORS + REFERENCE + TARGETS].copy()
for col in ["founder_n_founders", "founder_previous_yc", "n_previous_yc_founders", "n_previous_yc_startups", "company_one_liner_n_words", "company_n_tags", "company_site_name_length", "target_defined",
            "domain_differs", "in_yc_w26_startups_csv", "in_all_companies_json", "ref_baseline_top20", "target_top20"]:
    M[col] = M[col].astype("Int64")
M["attn_google_mentions_pre_raw"] = pd.to_numeric(M["attn_google_mentions_pre_raw"])
assert list(M.columns[-2:]) == TARGETS

# every column that carries information must be traceable in the audit
model_cols_in_audit = set(audit.loc[audit["in_model_dataset"] == 1, "model_column"].dropna())
assert set(PREDICTORS + REFERENCE + TARGETS) == model_cols_in_audit, (set(PREDICTORS + REFERENCE + TARGETS) ^ model_cols_in_audit)

# Dataset v1 (November checkpoint): a deliberately small view of the canonical table. Nothing is imputed or encoded beyond the TLD grouping.
V1_PREDICTORS = ["founder_n_founders", "founder_previous_yc", "company_one_liner_n_words", "company_site_tld_group", "attn_google_mentions_pre_raw"]
V1_DEFERRED = ["company_one_liner", "company_tags", "company_n_tags", "company_site_tld", "company_site_name_length", "n_previous_yc_founders", "n_previous_yc_startups"]
V1 = M.copy()
V1["company_site_tld_group"] = V1["company_site_tld"].map(lambda t: t if t in ("com", "ai") else "other")
V1.loc[V1["company_site_tld"].isna(), "company_site_tld_group"] = None
V1 = V1[META + V1_PREDICTORS + REFERENCE + TARGETS]
assert list(V1.columns[-2:]) == TARGETS and set(V1_PREDICTORS).isdisjoint(META + REFERENCE + TARGETS)
V1_ROLE_NOTES = {
    "founder_n_founders": ("x", "caveat", "Founder list is a 2026 snapshot; team size is unlikely to have changed."),
    "founder_previous_yc": ("x", "clean", "Earlier-batch founders matched by LinkedIn slug; lower bound; 6 startups are 1."),
    "company_one_liner_n_words": ("x", "caveat", "From the 2026 JSON snapshot; the one-liner may have been edited after the cutoff. Not a clean point-in-time feature."),
    "company_site_tld_group": ("x", "caveat", "com / ai / other, from the CSV domain (a 2026 website; csv and json TLDs differ for 4 of the 6 domain conflicts). Not a clean point-in-time feature."),
    "attn_google_mentions_pre_raw": ("x", "caveat", "Raw-attention modelling predictor (apply log1p when modelling), NOT the published YC Bench baseline. Same Google channel and domain as the target, "
                                                    "retrieved after the cutoff, and nonzero for some domains registered after it. Compare models with and without it."),
}
v1_rows = []
for c in V1.columns:
    if c in V1_PREDICTORS:
        role, tier, note = V1_ROLE_NOTES[c]
    elif c in TARGETS:
        role, tier, note = "target", "target", "Unchanged definition (YC Bench Pre-Demo Score)."
    elif c in REFERENCE:
        role, tier, note = "reference_baseline", "baseline_reference", "Published/reproduced YC Bench baseline (cleaned notebook values). Not X."
    else:
        role, tier, note = "audit", "meta", "Identifier / identity / audit field. Not X."
    v1_rows.append({"column": c, "role": role, "tier": tier, "note": note})
V1_MANIFEST = pd.DataFrame(v1_rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUT_DIR / "w26_feature_audit.csv", index=False, lineterminator="\n")
M.to_csv(OUT_DIR / "w26_model_dataset.csv", index=False, lineterminator="\n")
V1.to_csv(OUT_DIR / "w26_dataset_v1.csv", index=False, lineterminator="\n")
V1_MANIFEST.to_csv(OUT_DIR / "w26_dataset_v1_columns.csv", index=False, lineterminator="\n")

# ----------------------------------------------------------------------------------------------
# 7. Report
# ----------------------------------------------------------------------------------------------
fam_of = {c: f for f, cols in FAMILIES.items() for c in cols}
aud_by_col = audit.dropna(subset=["model_column"]).set_index("model_column")
pred_rows = []
for c in PREDICTORS:
    a = aud_by_col.loc[c]
    pred_rows.append({"column": c, "family": fam_of[c], "source": a["source"], "non-missing": f"{int(M[c].notna().sum())}/{N_UNIVERSE} ({100 * M[c].notna().mean():.1f}%)",
                      "temporal verdict": f"{a['available_pre_cutoff']} (tier: {a['tier']})", "reasoning": a["rationale"]})
pred_tbl = pd.DataFrame(pred_rows)
excl = audit[~audit["tier"].isin(["clean", "caveat", "target", "baseline_reference"])][["feature", "family", "tier", "n_non_missing", "n_nonzero", "n_units", "available_pre_cutoff", "target_leakage", "rationale"]]
excl = excl.rename(columns={"n_non_missing": "non-missing", "n_nonzero": "non-zero", "n_units": "of", "target_leakage": "leakage"})
unres_csv = sorted(r["canonical_slug"] for r in rows if r["identity_resolution"] == "csv_only")
unres_json = sorted(r["canonical_slug"] for r in rows if r["identity_resolution"] == "json_only")
renamed = [(r["csv_slug"], r["json_slug"]) for r in rows if r["identity_resolution"].startswith("rename_rule")]
dom_diff = U.loc[U["domain_differs"] == 1, ["canonical_slug", "csv_domain", "json_domain"]]
ties_note = ("**TIE AT THE TOP-20 BOUNDARY - positives are not uniquely defined.**" if TARGET_BOUNDARY_TIE else
             "No tie at the rank 20/21 boundary (checked on the score values).")
fh_lines = [
    "### Founder-history features (offline, from `yc_all_companies.json`)", "",
    "`founder_previous_yc` (any W26 founder was a founder in an earlier YC batch), `n_previous_yc_founders` (how many), `n_previous_yc_startups` (distinct earlier YC startups of the team). "
    f"Matching is by LinkedIn profile slug only. A batch counts only if it ended by {HISTORY_BEFORE[0]}-{HISTORY_BEFORE[1]:02d}; "
    "**[I]** the repository has no batch dates, so this uses an assumed calendar (W Jan-Mar, P spring Mar-Jun, S Jun-Sep, F Oct-Dec). F25 and later batches are never counted.", "",
    md_table(pd.DataFrame([
        ["W26 founders (JSON)", fh_stats["w26_founders"]],
        ["W26 founders with a LinkedIn slug (matchable)", fh_stats["w26_founders_with_key"]],
        ["matched to a founder of a batch clearly before the cutoff", fh_stats["founders_matched_clearly_before"]],
        ["startups with founder_previous_yc = 1", fh_stats["startups_flagged"]],
        ["founders matched only to records with no batch code (not counted)", fh_stats["founders_matched_unknown_batch"]],
        ["founders matched only to later batches such as F25/P26 (not counted)", fh_stats["founders_matched_not_clearly_before"]],
        ["founders appearing in more than one W26 startup (same URL; not 'previous')", fh_stats["founders_in_more_than_one_w26_startup"]],
        ["startups flagged if S25/P25/F24/F25 are excluded", fh_stats["sens_without_S25_P25_F24_F25"]],
        ["startups flagged if F25 were included", fh_stats["sens_including_F25"]],
    ], columns=["quantity", "count"])), "",
    f"Earlier batches of the matches: {dict(sorted(fh_stats['previous_batches'].items()))}. **[F]** The result does not depend on the assumed 2024-2025 calendar (see the two sensitivity rows).",
    "Lower bound: 104 records in the JSON have no batch code and founders matched only there are not counted; founders without a URL cannot be matched; the JSON is a 2026 snapshot. "
    "Startups without a JSON record (5) are missing, not 0.", "",
]
L = []
L += ["# W26 dataset report", "",
      "_Generated by `scripts/build_w26_dataset.py`; every number below is computed from the repository files. Statements are marked **[F]** (repository fact) or **[I]** (inference)._", "",
      "## 0. Operational cutoff", "",
      f"The operational benchmark cutoff is **{CUTOFF}**. **[F]** This is the end of the pre-application baseline window in `YC_W26_Google_Mentions.ipynb` "
      "(cell 38 markdown, `yc_serpapi_early.py:26`). It is **not** a verified official YC application deadline. **[I]** All verdicts below are relative to this operational date.", "",
      "Temporal tiers used: `clean` = timestamp or batch ordering shows it predates the cutoff; `caveat` = admissible but not strictly timestamped (static descriptor, or window valid but retrieved later); "
      "`excluded` = post-cutoff, target-side, unknown or unusable; `baseline_reference` = curated column kept only to compare with the published baseline; `meta` = identifiers.", "",
      "## 1. Counts", "",
      md_table(pd.DataFrame([
          ["canonical universe (startups)", counts["universe"], "union of yc_w26_startups.csv and yc_all_companies.json W26 after identity resolution"],
          ["target defined (a during-batch measurement exists; score may be 0)", counts["target_defined"], "notebook cell 2 holds one value per `yc_w26_startups.csv` domain"],
          ["target score > 0", counts["target_score_gt0"], "= rows of `yc_w26_pre_demo_scores.csv`, verified row by row"],
          ["target score = 0 (defined, no traction, 0 mentions in the notebook)", counts["target_score_eq0"], "dropped by the notebook (`dropna` on log10), kept here with score 0"],
          ["target undefined (no during-batch measurement)", counts["target_undefined"], "JSON-only startups: no CSV domain, never measured; kept as NaN, **not** 0"],
          ["target_top20 = 1", counts["target_top20"], f"{counts['top20_traction_driven']} traction-driven + {counts['top20_attention_driven']} attention-driven"],
      ], columns=["quantity", "count", "note"])), "",
      f"Model-eligible rows (`target_defined = 1`): **{counts['target_defined']}**. The three quantities (defined / score>0 / undefined) are different and are reported separately on purpose. "
      "**[F]** `yc_w26_pre_demo_scores.csv` contains only the score>0 rows, which is why it has fewer rows than the universe.", "",
      "## 2. Canonical universe and identity", "",
      f"- `yc_w26_startups.csv`: {len(csv_by_slug)} startups; `yc_all_companies.json` (batch W26): {len(json_by_slug)} startups. **[F]**",
      f"- Matched by identical slug: {sum(1 for r in rows if r['identity_resolution'] == 'slug_match')}. **[F]**",
      f"- Matched by rule (CSV name states `(formerly X)` and exactly one JSON-only record is named X): {renamed}. **[I]** This is the only non-slug match accepted.",
      f"- **Unresolved, CSV only ({len(unres_csv)}):** {unres_csv}", f"- **Unresolved, JSON only ({len(unres_json)}):** {unres_json}",
      "- `ishikilabs` (CSV: 'Fern (Formerly Ishiki Labs)', getfern.ai) and `fern-bot` (JSON: 'Fern', fern.bot) share a name but have different domains; not merged. **[F]** names/domains, **[D]** whether they are the same company.",
      f"- Domain differs between CSV and JSON for {len(dom_diff)} matched startups: " + ", ".join(f"{a} ({b} vs {c})" for a, b, c in dom_diff.itertuples(index=False)) + ". **[F]**",
      f"- yc-oss slugs from the March-2026 run that match no universe slug (not guessed): {unmatched_yc_oss}. **[F]**", "",
      "## 3. Predictor columns in `w26_model_dataset.csv`", "",
      "Columns before the predictors are metadata (identifiers, `target_defined`, `site_domain_source`); they are not predictors. "
      "Missing values are left as missing; nothing is imputed or transformed.", "",
      md_table(pred_tbl), "",
      "Families with **no usable columns**: funding / investor signals, network / ecosystem signals. **[F]** nothing for them exists in the repository.", "",
      "### Reference columns (not predictors)", "",
      "- `ref_attn_google_mentions_pre_cleaned_nb`: notebook cell 39 values (the ones that produce the published baseline). Researcher-curated, rule undocumented. "
      f"{int(M['ref_attn_google_mentions_pre_cleaned_nb'].notna().sum())} values, {int((M['ref_attn_google_mentions_pre_cleaned_nb'] > 0).sum())} above 0.",
      f"- `ref_baseline_top20`: 1 for the top {K} of that column (ties broken by domain; {'TIE AT BOUNDARY' if BASELINE_BOUNDARY_TIE else 'no tie at the boundary'}). "
      f"Intersection with `target_top20`: **{baseline_hits}/{K}**; the notebook stores 14/20 (output of its cell 41). **[F]**", "",
      *fh_lines,
      "## 4. Excluded features", "", md_table(excl), "",
      "## 5. Target construction", "",
      "`target_predemo_score` re-implements the notebook (cells 12-20), not `scripts/scoring/yc_predemo_score.py`: **[F]** the script is a different method "
      "(robust z-scores over ARR/funding/GitHub/HN) and cannot run because `twitter_metrics.json` is absent from the repository. **[I]** The notebook is the version tied to the published result.", "",
      "1. Base traction score = max(signal value x signal weight) over seven signal types; weights are read from notebook cell 12.",
      f"2. Velocity Score = base score x max(1 + {MOM_FACTOR} x MoM, {MOM_FLOOR}); the floor belongs to the notebook implementation. Checked equal to `yc_w26_traction.csv.Velocity_Score` for all {len(traction_csv)} rows.",
      f"3. Attention score = {ATTENTION_WEIGHT} x during-batch Google mentions (notebook cell 2 values).",
      "4. `target_predemo_score` = max(Velocity Score, attention score); Velocity is 0 for the 193 startups without traction data. "
      "**[F]** their traction is *unobserved*, not zero: traction is unobserved for most startups, so the computed Pre-Demo Score may underestimate the score that would be obtained with complete traction coverage.",
      f"5. `target_top20` = 1 for the {K} highest scores among `target_defined = 1` rows, 0 for other defined rows, NaN for undefined rows. Ties broken by CSV domain. {ties_note}",
      f"6. Score values duplicated among the {counts['target_score_gt0']} positive scores: {counts['tied_score_values_gt0']} rows (the lowest top-20 score is {'' if dup_at_boundary else 'not '}shared with a row outside the top 20); "
      f"{counts['target_score_eq0']} rows tie at 0.", "",
      f"**Composition of the positives.** {counts['top20_traction_driven']} come from the hand-entered Lobster Capital traction and {counts['top20_attention_driven']} from during-batch Google mentions. "
      f"**[F]** Every traction startup outranks every attention-only startup: {TRACTION_ALWAYS_OUTRANKS} (lowest traction score vs highest attention-only score, computed). **[I]** The label therefore mostly encodes 'was covered by the article' plus 'Google visibility after 2026-01-01'.", "",
      "**Sensitivity of the target to its input.** The committed `yc_mentions.csv` differs from the notebook's hard-coded during-batch values for "
      f"{n_during_diff} domains ({n_during_diff_nb_zero} are 0 in the notebook and positive in the CSV). Recomputing the target from `yc_mentions.csv` "
      f"(NaN as 0) keeps only {alt_overlap}/{K} of the positives. **[F]** This script follows the notebook, as it reproduces the committed `yc_w26_pre_demo_scores.csv`.", "",
      "## 6. Dataset v1 (November checkpoint)", "",
      f"`data/processed/w26_dataset_v1.csv` ({V1.shape[0]} rows x {V1.shape[1]} columns; column roles in `w26_dataset_v1_columns.csv`) is a deliberately small view of the canonical table. "
      "Nothing is imputed; missing values stay missing. The target and reference columns are unchanged and stay outside X.", "",
      "**X (5 predictors):** " + ", ".join(f"`{c}`" for c in V1_PREDICTORS) + ".", "",
      md_table(pd.DataFrame([[c, V1_ROLE_NOTES[c][1], f"{int(V1[c].notna().sum())}/{N_UNIVERSE}", V1_ROLE_NOTES[c][2]] for c in V1_PREDICTORS],
                            columns=["predictor", "tier", "non-missing", "caveat"])), "",
      "- `company_one_liner_n_words` and `company_site_tld_group` are caveat-level predictors: they come from later (2026) snapshots, not from point-in-time data.",
      "- `attn_google_mentions_pre_raw` is a raw-attention modelling predictor. It is **not** the published YC Bench baseline, which is `ref_attn_google_mentions_pre_cleaned_nb` / `ref_baseline_top20`. "
      "Models will be compared with and without it.",
      "- Not in v1 (kept in `w26_model_dataset.csv` and the audit): " + ", ".join(f"`{c}`" for c in V1_DEFERRED) + ". `company_n_tags` is dropped because W26 tags are mostly not yet assigned (27/199 vs 127/168 for W25), "
      "so 0 means untagged, not an observed zero.",
      "- Domain age is **not** in v1 (see section 7).",
      "- Audit-only columns (never in X): " + ", ".join(f"`{c}`" for c in META) + ". Reference: " + ", ".join(f"`{c}`" for c in REFERENCE) + ". Targets: " + ", ".join(f"`{c}`" for c in TARGETS) + ".", "",
      "## 7. Future feature directions", "",
      "Investigated in `scripts/pilot_w26_external_sources.py` (30-startup pilot, `reports/w26_pilot_acquisition_report.md`) and deliberately not used in v1. Cutoff 2025-10-31.", "",
      md_table(pd.DataFrame([
          ["LinkedIn founder education / employment / previous founder experience", "Most promising human-capital direction. P26 proxy (not W26): 205/229 founder profiles scraped (89.5%); education dated for 70-78% of entries; 54% of profiles show another founder role; "
           "YC entries (about 20% of education entries) must be masked.", "Deferred: collection needs legal / ethics clearance, and point-in-time validation of entries."],
          ["Funding / investors", "No funding or investor field exists anywhere in the repository for W26.", "Future enrichment."],
          ["Network / ecosystem", "Largely absent (Reddit run failed, no outputs).", "Future enrichment."],
          ["Domain age (RDAP)", "Retrieved 29/30. 10 of 29 CSV domains were registered after the cutoff (34%), so age is structurally absent for them; the domain is a 2026 website, "
           "registration after the cutoff is partly a consequence of acceptance, and it is mechanically tied to a target measured on that domain. 7 startups have different CSV/JSON domains "
           "(6 after normalising `try.wideframe.com`); old domains can carry a previous owner's history.", "Deferred: needs a historical-identity rule, and a post-cutoff state cannot be imputed."],
          ["Certificate Transparency", "29/30 retrieved; 17/28 first certificates fall within 3 days of registration, 4/29 predate it (previous owner); about 10% transient failures.", "Mostly redundant with domain age."],
          ["Wayback", "Any capture by the cutoff 11/30; 5 of those 11 first captures predate the current registration (another owner).", "Low / ambiguous coverage."],
          ["Hacker News", "1/30 stories and 1/30 comments matched.", "Too sparse."],
          ["GitHub", "Company-level repos by the cutoff 1/30; identity matching is weak.", "Too sparse / identity issues."],
          ["Richer one-liner semantics / embeddings", "Only the word count is used.", "Deferred."],
          ["Additional historical web / domain reconstruction", "See domain age, Wayback and CT.", "Deferred."],
      ], columns=["direction", "what the pilot showed", "status"])), "",
      "The earlier pilot recommendation to keep the earliest-registered of the CSV and JSON domains is **superseded**: it selects on the feature's own value and would make startups with two domains look older.", "",
      "## 8. Unresolved ambiguities", "",
      "- Origin and date of `yc_w26_startups.csv` and of the founder lists in `yc_all_companies.json` [D].",
      "- Identity of the 5 CSV-only and 6 JSON-only startups, including Fern / Ishiki Labs [D].",
      "- Rule and date for the notebook's cleaned values (cell 39: 68 zeros vs 12 in the raw CSV) and for the 14 zero values in cell 2 [D].",
      "- Dates of the Google collections and of the Lobster Capital article [D].",
      "- Whether the paper uses the cleaned or the raw baseline, and its definition of Precision@20 / Recall@11 / lift [D].",
      f"- `{REPORT_PATH.relative_to(ROOT)}` lives outside `data/` because `.gitignore` ignores `data/`; the CSVs under `data/processed/` are therefore untracked.", "",
      "## 9. Shapes", "",
      md_table(pd.DataFrame([["data/processed/w26_feature_audit.csv", audit.shape[0], audit.shape[1]], ["data/processed/w26_model_dataset.csv", M.shape[0], M.shape[1]],
                             ["data/processed/w26_dataset_v1.csv", V1.shape[0], V1.shape[1]], ["data/processed/w26_dataset_v1_columns.csv", V1_MANIFEST.shape[0], V1_MANIFEST.shape[1]]],
                            columns=["file", "rows", "columns"])), "",
      f"Model dataset columns: {len(META)} metadata + {len(PREDICTORS)} predictors + {len(REFERENCE)} reference + {len(TARGETS)} targets (the last two columns).", ""]
REPORT_PATH.write_text("\n".join(L))

print("universe", N_UNIVERSE, "| defined", counts["target_defined"], "| score>0", counts["target_score_gt0"], "| score=0", counts["target_score_eq0"],
      "| undefined", counts["target_undefined"], "| top20", counts["target_top20"])
print("audit", audit.shape, "| model", M.shape, "| baseline ∩ target", baseline_hits, "| boundary ties", TARGET_BOUNDARY_TIE, BASELINE_BOUNDARY_TIE)
