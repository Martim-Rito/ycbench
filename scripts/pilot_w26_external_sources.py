"""Pilot acquisition for candidate external W26 features (about 30 startups, no scaling).

For each source it records what was retrieved, with the exact temporal rule applied, so that coverage, missingness and
effort can be assessed BEFORE any full collection. Nothing here modifies the model dataset or the target.

Sample: the first N target-defined W26 startups ordered by sha256("<SEED>:<canonical_slug>"). The ordering does not use
target_top20, traction, the Pre-Demo score or any other outcome-related column.

Sources (all public, read-only, unauthenticated, low request rate):
  RDAP (IANA bootstrap)   domain registration date                          static, reconstructible
  crt.sh                  certificate transparency, first certificate       dated records
  Wayback CDX             captures of the homepage up to the cutoff         dated records
  HN Algolia              stories/comments mentioning the domain <= cutoff  dated records
  GitHub REST             owner/repos linked from the company homepage      repo created_at <= cutoff only
  LinkedIn                NOT queried. Offline proxy on the repository's own P26 profiles (date completeness, leakage markers).

Responses are cached in data/pilot/cache, so a re-run is offline and deterministic. Run: python scripts/pilot_w26_external_sources.py
"""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data" / "pilot"
CACHE_DIR = PILOT_DIR / "cache"
SEED, N_SAMPLE = "w26-pilot-v1", 30
CUTOFF = "2025-10-31"
CUT_TS = int(datetime(2025, 10, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
CUT_CDX = "20251031235959"
UA = "ycbench-thesis-pilot/0.1 (academic research, low request rate)"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMINGS = {}  # source -> list of seconds per live request


class Resp:
    def __init__(self, status=None, text="", headers=None, error=None, cached=False):
        self.status, self.text, self.headers, self.error, self.cached = status, text, headers or {}, error, cached

    def json(self):
        return json.loads(self.text)


def http_get(source, url, params=None, headers=None, timeout=30, retries=2, pause=1.0):
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{source}_{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return Resp(d["status"], d["text"], d["headers"], None, True)
    h = {"User-Agent": UA, **(headers or {})}
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            TIMINGS.setdefault(source, []).append(time.time() - t0)
            keep = {k: v for k, v in r.headers.items() if k.lower() in ("link", "x-ratelimit-remaining", "x-ratelimit-limit")}
            if r.status_code in (200, 404):  # 404 is a meaningful answer for RDAP/GitHub; transient codes are not cached
                path.write_text(json.dumps({"status": r.status_code, "text": r.text, "headers": keep,
                                            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": r.url}))
            time.sleep(pause)
            return Resp(r.status_code, r.text, keep)
        except requests.RequestException as exc:
            TIMINGS.setdefault(source, []).append(time.time() - t0)
            last = type(exc).__name__
            time.sleep(pause * (attempt + 1))
    return Resp(None, "", {}, last)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def apex(domain):
    parts = domain.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain


# ---------------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------------
model = pd.read_csv(ROOT / "data" / "processed" / "w26_model_dataset.csv")
pool = model[model["target_defined"] == 1].copy()
pool["_h"] = [hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest() for s in pool["canonical_slug"]]
sample = pool.sort_values("_h").head(N_SAMPLE).drop(columns="_h").reset_index(drop=True)
sample["lookup_domain"] = sample["csv_domain"].map(apex)
sample[["canonical_slug", "name", "csv_domain", "json_domain", "domain_differs", "lookup_domain"]].to_csv(PILOT_DIR / "w26_pilot_sample.csv", index=False)


# ---------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------
def rdap(domain, bootstrap):
    tld = domain.rsplit(".", 1)[-1]
    base = next((svc[1][0] for svc in bootstrap["services"] if tld in svc[0]), None)
    if not base:
        return {"rdap_status": "no_rdap_for_tld"}
    r = http_get("rdap", base.rstrip("/") + "/domain/" + domain, timeout=30, pause=0.5)
    if r.error:
        return {"rdap_status": f"error:{r.error}"}
    if r.status == 404:
        return {"rdap_status": "not_found"}
    if r.status != 200:
        return {"rdap_status": f"error:http_{r.status}"}
    ev = [e for e in r.json().get("events", []) if e.get("eventAction") == "registration"]
    if not ev:
        return {"rdap_status": "no_registration_event"}
    date = ev[0]["eventDate"][:10]
    out = {"rdap_status": "ok", "rdap_registration_date": date, "rdap_registered_after_cutoff": int(date > CUTOFF)}
    if date <= CUTOFF:
        out["rdap_age_days_at_cutoff"] = (datetime.fromisoformat(CUTOFF) - datetime.fromisoformat(date)).days
    return out


def crtsh(domain):
    r = http_get("crtsh", "https://crt.sh/", {"q": domain, "output": "json"}, timeout=60, retries=2, pause=2.0)
    if r.error or r.status != 200:
        return {"ct_status": f"error:{r.error or 'http_' + str(r.status)}"}
    try:
        certs = r.json()
    except ValueError:
        return {"ct_status": "error:bad_json"}
    if not certs:
        return {"ct_status": "empty"}
    nb = sorted(c["not_before"][:10] for c in certs)
    before = [c for c in certs if c["not_before"][:10] <= CUTOFF]
    return {"ct_status": "ok", "ct_first_cert_date": nb[0], "ct_first_cert_after_cutoff": int(nb[0] > CUTOFF),
            "ct_n_certs_before_cutoff": len({c["serial_number"] for c in before}), "ct_n_certs_all": len({c["serial_number"] for c in certs})}


def wayback(domain, reg_date=None):
    base = {"url": domain, "output": "json", "fl": "timestamp,statuscode", "filter": "statuscode:200", "collapse": "timestamp:8"}
    r = http_get("wayback", "https://web.archive.org/cdx/search/cdx", {**base, "to": CUT_CDX}, timeout=60, retries=2, pause=1.0)
    if r.error or r.status != 200:
        return {"wb_status": f"error:{r.error or 'http_' + str(r.status)}"}
    rows = r.json()[1:] if r.text.strip() else []
    r2 = http_get("wayback", "https://web.archive.org/cdx/search/cdx", {**base, "from": "20251101000000"}, timeout=60, retries=2, pause=1.0)
    after = (len(r2.json()[1:]) if (not r2.error and r2.status == 200 and r2.text.strip()) else None)
    out = {"wb_status": "ok" if rows else "empty", "wb_n_capture_days_before_cutoff": len(rows),
           "wb_first_capture": rows[0][0][:8] if rows else None, "wb_n_capture_days_after_cutoff_diagnostic": after}
    anyq = {k: v for k, v in base.items() if k != "filter"}
    r3 = http_get("wayback", "https://web.archive.org/cdx/search/cdx", {**anyq, "to": CUT_CDX}, timeout=60, retries=2, pause=1.0)
    out["wb_n_capture_days_before_cutoff_any_status"] = len(r3.json()[1:]) if (not r3.error and r3.status == 200 and r3.text.strip()) else None
    if reg_date and reg_date <= CUTOFF:
        r4 = http_get("wayback", "https://web.archive.org/cdx/search/cdx", {**base, "from": reg_date.replace("-", "") + "000000", "to": CUT_CDX}, timeout=60, retries=2, pause=1.0)
        out["wb_n_capture_days_since_registration_before_cutoff"] = len(r4.json()[1:]) if (not r4.error and r4.status == 200 and r4.text.strip()) else None
    return out


def host_matches(url, domain):
    host = (urlparse(url or "").hostname or "").lower()
    return host == domain or host.endswith("." + domain)


def hacker_news(domain):
    base = "https://hn.algolia.com/api/v1/search_by_date"
    nf = f"created_at_i<={CUT_TS}"
    s = http_get("hn", base, {"query": domain, "tags": "story", "restrictSearchableAttributes": "url", "numericFilters": nf, "hitsPerPage": 1000}, pause=0.5)
    c = http_get("hn", base, {"query": f'"{domain}"', "tags": "comment", "restrictSearchableAttributes": "comment_text", "numericFilters": nf, "hitsPerPage": 1000}, pause=0.5)
    if s.error or s.status != 200:
        return {"hn_status": f"error:{s.error or 'http_' + str(s.status)}"}
    sj = s.json()
    exact = [h for h in sj["hits"] if host_matches(h.get("url"), domain)]
    out = {"hn_status": "ok", "hn_n_stories_raw_hits": sj["nbHits"], "hn_n_stories_host_match": len(exact),
           "hn_first_story_date": min((h["created_at"][:10] for h in exact), default=None)}
    if not c.error and c.status == 200:
        out["hn_n_comments_text_match"] = sum(1 for h in c.json()["hits"] if domain in (h.get("comment_text") or "").lower())
    return out


SKIP_OWNERS = {"features", "topics", "sponsors", "marketplace", "about", "pricing", "login", "join", "orgs", "apps", "collections", "settings",
               "site", "contact", "security", "enterprise", "readme", "explore", "trending", "notifications", "pulls", "issues", "codespaces",
               "customer-stories", "solutions", "resources", "team", "organizations", "new", "search", "github", "login", "signup"}
GH_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
GH_STATE = {"remaining": None}


def gh_get(path, params=None):
    if GH_STATE["remaining"] is not None and GH_STATE["remaining"] <= 4:
        return Resp(None, "", {}, "rate_limit_guard")
    r = http_get("github", "https://api.github.com" + path, params, headers=GH_HEADERS, timeout=30, retries=1, pause=0.4)
    if r.headers.get("X-RateLimit-Remaining") is not None:
        GH_STATE["remaining"] = int(r.headers["X-RateLimit-Remaining"])
    return r


def github(row):
    domain = row["csv_domain"]
    page = http_get("homepage", f"https://{domain}/", timeout=20, retries=1, pause=0.8)
    if page.error or page.status is None or page.status >= 400:
        return {"gh_status": f"homepage_error:{page.error or 'http_' + str(page.status)}"}
    owners = {}
    for m in re.finditer(r"(?<![\w.-])github\.com/([A-Za-z0-9][A-Za-z0-9_.-]*)(?:/([A-Za-z0-9_.-]+))?", page.text):
        o = m.group(1)
        if o.lower() in SKIP_OWNERS:
            continue
        owners[o] = owners.get(o, 0) + 1
    if not owners:
        return {"gh_status": "no_github_link_on_homepage"}
    names = {norm(row["canonical_slug"]), norm(row["name"]), norm(row["lookup_domain"].split(".")[0])}
    plausible = {o: any(len(a) >= 4 and (a in norm(o) or norm(o) in a) for a in names) for o in owners}
    ranked = sorted(owners, key=lambda o: (-plausible[o], -owners[o], o))
    todo = [o for o in ranked if plausible[o]][:2] or (ranked[:1] if len(ranked) == 1 else [])
    out = {"gh_status": "link_found", "gh_owners_linked": "|".join(ranked), "gh_n_owners_linked": len(ranked), "gh_n_plausible_owners": sum(plausible.values())}
    if not todo:
        out["gh_status"] = "link_found_not_queried_ambiguous_owner"
        return out
    o = todo[0]
    u = gh_get(f"/users/{o}")
    if u.error or u.status != 200:
        out["gh_status"] = f"api_error:{u.error or 'http_' + str(u.status)}"
        return out
    uj = u.json()
    rp = gh_get(f"/users/{o}/repos", {"per_page": 100, "type": "owner"})
    if rp.error or rp.status != 200:
        out["gh_status"] = f"api_error_repos:{rp.error or 'http_' + str(rp.status)}"
        return out
    repos = rp.json()
    before = [x for x in repos if x["created_at"][:10] <= CUTOFF and not x["fork"]]
    out.update({"gh_status": "ok", "gh_owner_queried": o, "gh_owner_plausible": int(plausible[o]), "gh_owner_type": uj["type"],
                "gh_owner_created_at": uj["created_at"][:10], "gh_owner_created_before_cutoff": int(uj["created_at"][:10] <= CUTOFF),
                "gh_n_repos_listed": len(repos), "gh_repos_truncated_at_100": int(len(repos) >= 100), "gh_n_own_repos_created_before_cutoff": len(before)})
    if before:
        first = sorted(before, key=lambda x: x["created_at"])[0]
        out["gh_first_repo_created"] = first["created_at"][:10]
        cm = gh_get(f"/repos/{o}/{first['name']}/commits", {"until": f"{CUTOFF}T23:59:59Z", "per_page": 1})
        if not cm.error and cm.status == 200:
            link = cm.headers.get("Link", "")
            m = re.search(r'[?&]page=(\d+)>; rel="last"', link)
            out["gh_commits_before_cutoff_first_repo"] = int(m.group(1)) if m else len(cm.json())
    return out


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------
boot = http_get("rdap", "https://data.iana.org/rdap/dns.json", timeout=30)
bootstrap = boot.json()
rows = []
for _, r in sample.iterrows():
    d = r["lookup_domain"]
    rec = {"canonical_slug": r["canonical_slug"], "csv_domain": r["csv_domain"], "lookup_domain": d, "domain_differs": int(r["domain_differs"])}
    rec.update(rdap(d, bootstrap))
    rec.update(crtsh(d))
    rec.update(wayback(d, rec.get("rdap_registration_date")))
    rec.update(hacker_news(d))
    rec.update(github(r))
    reg = rec.get("rdap_registration_date")
    rec["ct_first_cert_before_registration"] = int(bool(reg and rec.get("ct_first_cert_date") and rec["ct_first_cert_date"] < reg))
    rec["wb_first_capture_before_registration"] = int(bool(reg and rec.get("wb_first_capture") and f"{rec['wb_first_capture'][:4]}-{rec['wb_first_capture'][4:6]}-{rec['wb_first_capture'][6:8]}" < reg))
    rows.append(rec)
    print(r["canonical_slug"], rec.get("rdap_status"), rec.get("ct_status"), rec.get("wb_status"), rec.get("hn_status"), rec.get("gh_status"))
res = pd.DataFrame(rows)
res.to_csv(PILOT_DIR / "w26_pilot_results.csv", index=False)

# ---------------------------------------------------------------------------------------------
# LinkedIn: offline proxy on the repository's own P26 profiles (no LinkedIn request is made)
# ---------------------------------------------------------------------------------------------
MONTH = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
profiles = json.load(open(ROOT / "spring_2026_batch" / "yc_p26_dataset_linkedin-profile-scraper_2026-05-17_13-54-11-106.json"))
p26 = json.load(open(ROOT / "spring_2026_batch" / "yc_spring2026_withlinkedin.json"))


def lkey(u):
    m = re.search(r"linkedin\.com/in/([^/?#]+)", unquote(u or "").strip().lower())
    return m.group(1) if m else None


def ym(d):
    if not d or not d.get("year"):
        return None
    return (int(d["year"]), MONTH.get(d.get("month"), 1))


founder_company = {lkey(f.get("linkedin_url")): x["name"] for x in p26 for f in (x.get("founders") or []) if lkey(f.get("linkedin_url"))}
founder_urls = set(founder_company)
profile_keys = {lkey(p.get("linkedinUrl")) for p in profiles}
YC = re.compile(r"\bYC\b|Y Combinator", re.I)
FOUNDER = re.compile(r"\b(co-?founder|founder|founding)\b", re.I)
cut = (2025, 10)
exp = [(p, e) for p in profiles for e in (p.get("experience") or [])]
edu = [(p, e) for p in profiles for e in (p.get("education") or [])]
matched = [p for p in profiles if lkey(p.get("linkedinUrl")) in founder_company]
cur_ident, prev_founder = 0, 0
for p in matched:
    cname = norm(founder_company[lkey(p["linkedinUrl"])])
    ents = p.get("experience") or []
    is_cur = [len(cname) >= 3 and (cname in norm(e.get("companyName")) or norm(e.get("companyName")) in cname) and bool(norm(e.get("companyName"))) for e in ents]
    cur_ident += int(any(is_cur))
    prev_founder += int(any(FOUNDER.search(e.get("position") or "") and not c and (ym(e.get("startDate")) or (9999, 1)) <= cut for e, c in zip(ents, is_cur)))
li = {
    "scope": "P26 profiles already in the repository (scraped 2026-05-17); proxy only, no W26 profile was requested",
    "profiles": len(profiles), "founder_urls_in_p26_json": len(founder_urls), "founder_urls_with_a_scraped_profile": len(founder_urls & profile_keys),
    "experience_entries": len(exp), "experience_with_start_year": sum(1 for _, e in exp if ym(e.get("startDate"))),
    "experience_entries_started_after_cutoff": sum(1 for _, e in exp if (ym(e.get("startDate")) or (0, 0)) > cut),
    "experience_entries_naming_YC": sum(1 for _, e in exp if YC.search(e.get("companyName") or "")),
    "profiles_with_experience_naming_YC": sum(1 for p in profiles if any(YC.search(e.get("companyName") or "") for e in (p.get("experience") or []))),
    "education_entries": len(edu), "education_with_start_year": sum(1 for _, e in edu if ym(e.get("startDate"))),
    "education_with_end_year": sum(1 for _, e in edu if ym(e.get("endDate"))), "education_with_degree": sum(1 for _, e in edu if e.get("degree")),
    "education_entries_Y_Combinator": sum(1 for _, e in edu if YC.search(e.get("schoolName") or "")),
    "profiles_with_headline_naming_YC": sum(1 for p in profiles if YC.search(p.get("headline") or "")),
    "profiles_matched_to_p26_startup": len(matched), "matched_profiles_where_current_startup_entry_is_identified_by_name": cur_ident,
    "matched_profiles_with_other_founder_role_started_by_cutoff": prev_founder,
}
(PILOT_DIR / "w26_pilot_linkedin_p26_proxy.json").write_text(json.dumps(li, indent=2))
(PILOT_DIR / "w26_pilot_timings.json").write_text(json.dumps({k: {"n": len(v), "mean_s": round(sum(v) / len(v), 2), "max_s": round(max(v), 2)} for k, v in TIMINGS.items()}, indent=2))
print("done; github rate-limit remaining:", GH_STATE["remaining"])
