# W26 pilot acquisition report

Status: pilot only. No full collection was run, the target definition is unchanged, and the model dataset gained only the three offline founder-history columns.
Statements are marked **[F]** (observed in the repository or in a pilot response), **[I]** (inference) or **[D]** (unknown).
Operational cutoff: **2025-10-31** (end of the notebook baseline window; not a verified YC deadline).

## 1. Offline founder-history features (implemented)

Added to `w26_model_dataset.csv` by `scripts/build_w26_dataset.py`: `founder_previous_yc`, `n_previous_yc_founders`, `n_previous_yc_startups`.

- Rule: a W26 founder counts when the same LinkedIn profile slug is a founder of a company in an earlier batch that clearly ended before the cutoff (batch ended by 2025-09). Name matching is not used. The batch calendar (W Jan-Mar, P spring Mar-Jun, S Jun-Sep, F Oct-Dec) is an assumption **[I]**, because the repository has no batch dates. F25 and later batches are never counted.
- Result **[F]**: 8 of 422 matchable founders, 6 of 199 startups (3.0%) have previous YC history; the earlier batches are S19 (3), S23 (3), W12 (1), W21 (1). Excluding S25/P25/F24/F25 or including F25 does not change the 6.
- Missing: 5 startups without a JSON record are NaN, not 0. The 104 JSON records without a batch code match none of the W26 founders.
- Shapes: model dataset 204 x 27 (was 24), audit 44 rows (was 41). Target counts are unchanged (198 defined, 184 > 0, 20 positives). Re-running the builder reproduces identical files.
- The feature is very sparse, so it carries little information on its own **[I]**.

## 2. Pilot design

- Sample: 30 target-defined startups, taken as the first 30 by `sha256("w26-pilot-v1:" + canonical_slug)`. The ordering does not use the target, traction or score. Composition, reported only afterwards: 28 have a JSON record, 25 have score > 0, 5 are in the Top-20 (so ~17%, close to the universe's 10%).
- Sources are public, unauthenticated and queried at a low rate. Responses are cached in `data/pilot/cache`, so the run can be repeated offline. Raw results: `data/pilot/w26_pilot_results.csv` (git-ignored). Script: `scripts/pilot_w26_external_sources.py`.
- **LinkedIn was not queried.** Scraping it conflicts with its terms and involves personal data, which needs your and your university's decision. Instead, the repository's own 229 P26 profiles (scraped 2026-05-17) were analysed offline as a proxy. Those numbers describe P26, not W26.
- No GitHub token was used (60 requests/hour). No present-day snapshot (followers, headline, current role, location, stars) was used as a feature.

## 3. Results per source

| Candidate | Source and exact temporal rule | Pilot coverage (N=30) | Missingness / what absence means | Identity reliability | Leakage risk | Effort to scale (204 startups) |
|---|---|---|---|---|---|---|
| Domain creation age | RDAP via the IANA bootstrap; registration date; age = 2025-10-31 minus date, only if registered by the cutoff | Retrieved 29/30 (1 TLD has no RDAP). **10/29 domains were registered after the cutoff**, so usable age at cutoff exists for 19/30 (median 202 days) | Absence of age = "domain did not exist yet" (10) or "no RDAP" (1); these must be coded separately | Keyed on the 2026 CSV domain. Creation date resets when a domain is re-registered | Low (static) | Low: ~0.4 s per call, ~2 min; WHOIS fallback for TLDs without RDAP |
| First HTTPS certificate | crt.sh certificate transparency; earliest `not_before`; only certs with `not_before <= cutoff` count | 29/30 retrieved after retries (first pass: 3 x HTTP 502 and 2 x HTTP 404; 1 x HTTP 404 persists); first cert by the cutoff in 21, after it in 8 | Empty/404 = unavailable, not "no site" | Same domain issue; 4/29 first certs predate the domain's registration (previous owner) | Low | Low-medium: ~8.6 s per call (~30 min), ~10% transient failures; 17/28 first certs fall within 3 days of the registration date, so it mostly duplicates domain age |
| Wayback snapshots before cutoff | CDX API for the homepage, status 200, one count per capture day, `to=20251031235959` | Any 200 capture by the cutoff: 11/30 (median 4 capture days). Of 15 domains with a complete check, 8 have captures since their current registration | No capture means "not archived", not "no website" | **5 of the 11 have a first capture earlier than the current registration, i.e. another owner's site** | Low if only captures <= cutoff and since registration are used | Medium: 2-4 calls per domain, rate-limited (5 diagnostic calls failed), ~20-40 min |
| HN mentions | Algolia `search_by_date`, stories whose URL host equals the domain, `created_at_i <= cutoff`; comments by text | **Story match 1/30**, comment match 1/30 (`openspec.dev`, 2025-10-22) | For link submissions a 0 is a true "none" (the API covers all items); text-only posts and fuzzy comment hits are not covered | Exact host match is reliable. Raw hits are not: 6/30 startups had raw hits, and 5 of them disappeared after the host filter | Low (use counts of items <= cutoff; points keep growing after posting) | Low: 2 calls, ~0.6 s each, ~5 min. Value is low because it is almost always 0 |
| GitHub evidence | Homepage links to GitHub (present-day page, used only to find the owner), then `created_at <= cutoff` for owner and repos | Homepage fetched 29/30 (1 x 404); any GitHub link 5/30; company-level repos created by the cutoff **1/30** (`Fission-AI`, linked from `openspec.dev`: 2 repos, 355 commits) | No link means unknown. An owner with 0 repos by the cutoff is a true none | Weak: 1 linked account is a personal user account with 22 repos (probably a founder's **[I]**), 3 are organisations created after or with no repos by the cutoff, and name matching flagged the real OpenSpec org as implausible | Low if restricted by creation date. Current stars and `pushed_at` were not used | Medium-high: unauthenticated limit 60/h; ~40 owners x 3 calls need a token. JS-rendered pages hide links, so coverage is understated **[I]** |
| Founder education (LinkedIn, P26 proxy) | Entries with dates <= cutoff; YC entries removed | P26: 205 of 229 founder URLs have a scraped profile (89.5%); education has a start year in 440/633 entries (70%), end year 496 (78%), degree 504 (80%) | A missing profile is unavailable, not "no education" | Join by profile slug is reliable; the scraper's yield is the limit | Medium: 127/633 education entries are "Y Combinator" and must be dropped; entries can be edited later | High: scrape, parse, rank institutions, and settle the legal question first. W26 URLs: 423/424 |
| Founder previous employment (P26 proxy) | Experience entries with start date <= cutoff | 1600 entries (7.0 per profile), all have a start date; 169 (10.6%) start after the equivalent cutoff and would be dropped; 103 entries (6.4%) name YC | Same as above | Company names are free text, so employer categories need a lookup table | Medium: 163/229 headlines name YC (so headline is unusable), 99 profiles list a YC role | High |
| Previous founder/startup experience (P26 proxy) | Founder-titled entries starting <= cutoff, excluding the current startup | 111/205 matched profiles (54%) have another founder role by the cutoff | Derived from titles, so uncertain | The current-startup entry is identified by name in 197/205 profiles (96%); the other 8 would be mistaken for "previous" | Medium | High (same scrape) |

## 4. Cross-cutting findings

1. **The CSV domain is a 2026 website.** 10 of 29 (34%) sample domains were registered after the cutoff, so these startups have no pre-cutoff history on that domain. That absence is structural, not missing data.
2. **The domain choice matters.** For `moda`, the JSON domain `moda.dev` was registered 2023-08-02 (before the cutoff) while the CSV domain `usemoda.ai` was registered 2026-01-28. For `constellation-space`, both domains post-date the cutoff. Using only one domain can make the pre-cutoff history look absent.
3. **Re-registered domains carry another owner's history**: 5/11 Wayback hits and 4/29 first certificates predate the current registration. Any domain-history feature must be restricted to the period since the current registration.
4. **Coverage is the main problem.** HN (3%) and GitHub (3% company-level) are nearly constant in W26; domain age and Wayback give information mostly through the "did not exist yet" indicator, not through age.
5. **Present-day snapshots were avoided**, but the GitHub owner discovery uses a current homepage. This is identity only, not a historical value **[I]**.

## 5. Ranking

Criteria in your order: temporal validity, coverage, taxonomy relevance, collection reliability, implementation effort. Grades are relative (H high, M medium, L low); for effort, L is best.

| Rank | Candidate | Temporal validity | Coverage | Taxonomy relevance | Reliability | Effort |
|---|---|---|---|---|---|---|
| 1 | Domain creation age | H | H (97% retrieved; 63% usable) | M (company characteristics) | H | L |
| 2 | First HTTPS certificate | H | H (97%) | L-M (mostly duplicates age) | M | L-M |
| 3 | Wayback snapshots (since registration) | H | M-L (37% any; 27% since registration) | M (traction / maturity) | M | M |
| 4 | HN mentions | H | L (3%) | M (attention) | H | L |
| 5 | GitHub evidence | H | L (3% company-level) | M (traction / ecosystem) | L | M-H |
| 6 | Founder education | M-H | H (90% profile yield; 70-78% dated) | **H (founder family)** | L-M | H |
| 7 | Founder previous employment | M | H | **H** | L-M | H |
| 8 | Previous founder experience | M-L | M (54%) | **H** | L | H |

By your strict order, the dated technical sources (ranks 1-5) come first. **My recommendation differs on one point**: the three LinkedIn-derived features are the only way to populate the founder / human-capital family beyond team size and repeat-YC, so they deserve a separate go/no-go once the legal question is settled, with YC markers masked and entries after the cutoff dropped.

## 6. Recommendations before scaling

1. ~~Resolve the **domain identity** first: use both the CSV and JSON domains and keep the one with the earliest registration; add a `registered_after_cutoff` indicator and a `domain_other_owner_history` flag.~~ **Superseded:** the earliest-registered rule selects on the feature's own value, and domain age is deferred from Dataset v1 (see `reports/w26_dataset_report.md`, sections 6-7).
2. Scale **domain age** and **Wayback (since registration)**; the certificate feature adds little. Treat HN and GitHub as low-yield (collect only if cheap, with a GitHub token).
3. Decide on LinkedIn (terms of use, GDPR, ethics approval) before any founder-level collection.
4. Keep missing, "did not exist yet" and "not archived" as separate states; do not code them as 0.

## 7. Not done

No LinkedIn request, no full-universe collection, no token use, and no change to the target or to the model dataset beyond section 1. The pilot sample (30 of 198) is too small to give precise coverage rates; intervals are wide **[I]**.
