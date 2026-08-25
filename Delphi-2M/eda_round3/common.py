"""
common.py -- shared loading / registry / bookkeeping for ROUND 3: the ordinal-scale
variable audit (《序数量表变量专项 EDA 规格书》).

Rounds 1 (Delphi-2M/eda/eda_radc.py) and 2 (Delphi-2M/eda_round2/) are untouched. This round
is a third, parallel pass with a single question: **is "this variable is an ordered scale"
actually true?** Ordinal smoothing regularisation and anchor interpolation both rest on that
assumption; imposing them where it fails actively damages the model.

TWO datasets, deliberately kept in separate panels because they share no rows:

  nacc_visit   investigator_nacc72.csv        207455 visits x 55268 people, 2005-2025
  radc_visit   longitudinal_data_gk.xlsx      36138 visits x 4428 people (annual cycles)
  radc_person  cross-sectional + ROSMAP_clinical, person level

Every `note` in VARS is transcribed from the NACC UDS Data Element Dictionary / RDD or the
RADC codebook (data/RADC/RADC_codebook_data_set_1736_08-13-2026.pdf). Nothing is inferred
from a variable NAME -- ceradsc and r_stroke are reverse coded, and the spec itself warns
that CERAD / NIA-Reagan directions may be flipped.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))          # Delphi-2M/eda_round3/
DELPHI = os.path.dirname(HERE)                             # Delphi-2M/
ROOT = os.path.dirname(DELPHI)
RADC_DIR = os.path.join(ROOT, "data", "RADC")

OUT = os.path.join(HERE, "eda_out", "scales")
FIGS = os.path.join(OUT, "figs")
TABLES = os.path.join(OUT, "tables")
CACHE = os.path.join(OUT, ".cache")
REPORT = os.path.join(OUT, "REPORT.md")
SUMMARY = os.path.join(OUT, "summary.json")
SCALE_SUMMARY = os.path.join(OUT, "SCALE_SUMMARY.csv")
for _d in (OUT, FIGS, TABLES, CACHE):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, DELPHI)
# Re-exported for the section modules, which import these from `common` so that every
# module has exactly one place to import plotting helpers from.
from figure2.plotting_style import (setup_style, save_fig, save_data,  # noqa: E402,F401
                                   REF_GREY)

# ---------------------------------------------------------------- palette
# Same fixed categorical order as round 2 (Okabe-Ito subset, validated light-surface).
CAT = ["#0072B2", "#D55E00", "#E69F00", "#009E73"]
C1, C2, C3, C4 = CAT
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"

# ---------------------------------------------------------------- NACC location
# The raw NACC export is PHI and lives OUTSIDE this repo (.gitignore blocks
# **/investigator_nacc*.csv anyway). Resolution order, first hit wins.
NACC_ENV = "NACC_CSV"
NACC_CANDIDATES = [
    os.path.join(ROOT, "data", "NACC", "investigator_nacc*.csv"),
    os.path.join(ROOT, "data", "nacc", "investigator_nacc*.csv"),
    os.path.expanduser("~/cc_test/EDA_NACC/investigator_nacc*.csv"),
]


def nacc_path():
    """Absolute path to the NACC investigator CSV, or None if this machine has no copy."""
    p = os.environ.get(NACC_ENV)
    if p and os.path.exists(p):
        return p
    for pat in NACC_CANDIDATES:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]            # highest-numbered release
    return None


# ---------------------------------------------------------------- expected-value grammar
def levels_int(lo, hi):
    return [float(x) for x in range(lo, hi + 1)]


def levels_half(lo, hi):
    """Every 0.5 multiple in [lo, hi] -- the NOMINAL CDRSUM grid, most of which is
    unreachable because the six box scores cannot sum to it (see s3 3.3)."""
    return [lo + 0.5 * i for i in range(int((hi - lo) / 0.5) + 1)]


CONTINUOUS = "continuous"

# FAQ item columns, UDS form B7, in the official order. 0-3 each, so a complete total is 0-30.
FAQ_ITEMS = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE",
             "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]
# 8 = not applicable (never did the activity), 9 = unknown, -4 = not in this form version.
FAQ_SENTINELS = {-4.0, 8.0, 9.0}

CDR_BOXES = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]


class Var:
    """One candidate scale variable, as the spec's §1 tables describe it.

    `spec_name`  the name the spec uses (what we look for)
    `candidates` column names to probe, in order; the FIRST that exists wins
    `expected`   list of legal levels, or CONTINUOUS
    `direction`  EXPECTED direction per spec §1; §2.7 measures it and flags mismatch
    `sentinels`  codes that are missing-data markers, not values
    `dichotomy`  (op, threshold) used only for the §4.2 leakage matrix / §4.3 AUCs
    `role`       spec-declared starting role, revised by the §5 verdict
    """

    def __init__(self, spec_name, panel, candidates, expected, direction, note,
                 sentinels=(), dichotomy=None, role="candidate", group="scale",
                 offgrid=None):
        self.spec_name = spec_name
        self.panel = panel
        self.candidates = list(candidates)
        self.expected = expected
        self.direction = direction
        self.note = note
        self.sentinels = set(float(s) for s in sentinels)
        self.dichotomy = dichotomy
        self.role = role
        self.group = group
        # offgrid: a stated REASON why values may legally sit between declared levels.
        # When set, legality is judged on the [min, max] interval instead of the grid, and
        # the off-grid fraction is reported rather than treated as an encoding error.
        self.offgrid = offgrid
        self.column = None            # filled in by s1_detect
        self.status = "unprobed"

    @property
    def key(self):
        return f"{self.panel.split('_')[0]}:{self.spec_name}"

    @property
    def dataset(self):
        return self.panel.split("_")[0]

    @property
    def grain(self):
        return self.panel.split("_")[1]

    def expected_str(self):
        if self.expected is CONTINUOUS:
            return "continuous"
        e = self.expected
        if len(e) > 8:
            step = e[1] - e[0]
            return f"{_num(e[0])}-{_num(e[-1])} step {_num(step)} ({len(e)} levels)"
        return "/".join(_num(x) for x in e)

    def clean(self, s):
        """Numeric values with sentinel codes turned into NaN."""
        x = pd.to_numeric(s, errors="coerce")
        if self.sentinels:
            x = x.mask(x.isin(list(self.sentinels)))
        return x

    def illegal(self, s):
        """Values that survive sentinel removal but are still not legal.

        "Legal" means ON the declared grid, unless `offgrid` states a reason values may sit
        between levels -- then it means inside the declared [min, max] interval.
        """
        x = self.clean(s).dropna()
        if self.expected is CONTINUOUS:
            return pd.Series(dtype=float)
        grid = np.asarray(self.expected, float)
        if self.offgrid:
            ok = (x.to_numpy() >= grid.min() - 1e-9) & (x.to_numpy() <= grid.max() + 1e-9)
        else:
            ok = np.isclose(x.to_numpy()[:, None], grid[None, :]).any(axis=1)
        return x[~ok]

    def pct_offgrid(self, s):
        """Fraction of observed values that are NOT on the declared grid."""
        x = self.clean(s).dropna().to_numpy(float)
        if self.expected is CONTINUOUS or len(x) == 0:
            return 0.0
        grid = np.asarray(self.expected, float)
        on = np.isclose(x[:, None], grid[None, :]).any(axis=1)
        return float(100.0 * (~on).mean())


def _num(x):
    x = float(x)
    return str(int(x)) if x == int(x) else str(x)


# ---------------------------------------------------------------- the registry (spec §1)
VARS = [
    # ---- §1.1 backbone / auxiliary candidates, NACC ------------------------------------
    Var("NACCMMSE", "nacc_visit", ["NACCMMSE"], levels_int(0, 30), "higher_better",
        "MMSE total, UDS form C1. Retired from the UDS in 2015 in favour of MoCA.",
        sentinels=[-4, 88, 95, 96, 97, 98], dichotomy=("<", 24)),
    Var("MOCATOTS", "nacc_visit", ["MOCATOTS"], levels_int(0, 30), "higher_better",
        "MoCA total score, UDS form C2. Did not exist before the 2015 UDS v3 rollout.",
        sentinels=[-4, 88], dichotomy=("<", 23)),
    Var("CDRSUM", "nacc_visit", ["CDRSUM"], levels_half(0, 18), "higher_worse",
        "CDR sum of boxes = sum of the six domain scores. The 0.5 grid is NOT uniformly "
        "reachable: five domains take {0,.5,1,2,3} but PERSCARE has no 0.5 level.",
        sentinels=[99], dichotomy=(">=", 1)),
    Var("CDRGLOB", "nacc_visit", ["CDRGLOB"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR global score from the Washington University scoring algorithm. Unequal spacing: "
        "the 0->0.5 step is not the same amount of disease as the 2->3 step.",
        sentinels=[99], dichotomy=(">=", 0.5)),
    Var("FAQTOTAL", "nacc_visit", ["FAQTOTAL"], levels_int(0, 30), "higher_worse",
        "DERIVED here: sum of the 10 UDS form B7 FAQ items (0-3 each). The release ships no "
        "FAQ total column. CO-PARTICIPANT reported, so its missingness tracks social support.",
        dichotomy=(">=", 1)),
    Var("NACCGDS", "nacc_visit", ["NACCGDS"], levels_int(0, 15), "higher_worse",
        "Geriatric Depression Scale SHORT form total (0-15). Measures depression, not "
        "cognition. 88 = could not be calculated (too many items missing).",
        sentinels=[-4, 88], dichotomy=(">=", 5)),
    # ---- §1.2 status / diagnosis --------------------------------------------------------
    Var("NACCUDSD", "nacc_visit", ["NACCUDSD"], [1, 2, 3, 4], "higher_worse",
        "UDS cognitive status: 1 normal / 2 impaired-not-MCI / 3 MCI / 4 dementia. Category 2 "
        "is the spec's suspected break in the ordering.", dichotomy=("==", 4)),
    # ---- §3.3 the six CDR box scores ----------------------------------------------------
    Var("MEMORY", "nacc_visit", ["MEMORY"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR box: memory.", sentinels=[99], dichotomy=(">=", 0.5), group="cdr_box"),
    Var("ORIENT", "nacc_visit", ["ORIENT"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR box: orientation.", sentinels=[99], dichotomy=(">=", 0.5), group="cdr_box"),
    Var("JUDGMENT", "nacc_visit", ["JUDGMENT"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR box: judgment and problem solving.", sentinels=[99], dichotomy=(">=", 0.5),
        group="cdr_box"),
    Var("COMMUN", "nacc_visit", ["COMMUN"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR box: community affairs.", sentinels=[99], dichotomy=(">=", 0.5), group="cdr_box"),
    Var("HOMEHOBB", "nacc_visit", ["HOMEHOBB"], [0, 0.5, 1, 2, 3], "higher_worse",
        "CDR box: home and hobbies.", sentinels=[99], dichotomy=(">=", 0.5), group="cdr_box"),
    Var("PERSCARE", "nacc_visit", ["PERSCARE"], [0, 1, 2, 3], "higher_worse",
        "CDR box: personal care. FOUR levels -- this domain has NO 0.5 rating, which is why "
        "the CDRSUM grid is lumpy.", sentinels=[99], dichotomy=(">=", 1), group="cdr_box"),
    # ---- §1.1 RADC ---------------------------------------------------------------------
    Var("cts_mmse30", "radc_visit", ["cts_mmse30", "cts_estmmse30"], levels_int(0, 30),
        "higher_better",
        "RADC MMSE. The release ships cts_estmmse30: 'either the MMSE score or the full MMSE "
        "score ESTIMATED from the MoCA' (codebook p.230) -- i.e. already partly interpolated.",
        dichotomy=("<", 24),
        offgrid="codebook p.230: the column is the MMSE score OR an MMSE estimated from the "
                "MoCA, so fractional scores are real data, not an encoding error. The spec's "
                "'0-30 integer' expectation is what is wrong -- see s3 3.1."),
    Var("cogn_global", "radc_visit", ["cogn_global"], CONTINUOUS, "higher_better",
        "Global cognition z score, mean of 19 tests, standardised on the BASELINE-cycle "
        "cohort mean/sd. Continuous -- must be binned before any token treatment.",
        dichotomy=("<", -1.0)),
    # ---- §1.2 RADC ---------------------------------------------------------------------
    Var("dcfdx", "radc_person", ["dcfdx", "dcfdx_lv"], levels_int(1, 6), "higher_worse",
        "Clinical cognitive dx: 1 NCI / 2 MCI / 3 MCI+other / 4 AD / 5 AD+other / 6 other "
        "dementia. Level 6 is a DIFFERENT DISEASE, not a further step -- non-ordinal as-is.",
        dichotomy=(">=", 4)),
    Var("cogdx", "radc_person", ["cogdx"], levels_int(1, 6), "higher_worse",
        "FINAL consensus cognitive dx, same 1-6 coding as dcfdx. Post-mortem informed for "
        "autopsied participants, so it partly ENCODES the pathology outcomes.",
        dichotomy=(">=", 4)),
    # ---- §1.3 leakage blacklist: outcome only ------------------------------------------
    Var("braaksc", "radc_person", ["braaksc"], levels_int(0, 6), "higher_worse",
        "Braak NFT stage 0-6, ascending severity. AUTOPSY ONLY.",
        dichotomy=(">=", 4), role="outcome_only"),
    Var("ceradsc", "radc_person", ["ceradsc"], levels_int(1, 4), "higher_better",
        "CERAD: 1 definite AD / 2 probable / 3 possible / 4 no AD -- REVERSE coded, so higher "
        "= LESS pathology. AUTOPSY ONLY.", dichotomy=("<=", 2), role="outcome_only"),
    Var("niareagansc", "radc_person", ["niareagansc", "nia_reagan", "niaregansc"],
        levels_int(1, 4), "higher_better",
        "NIA-Reagan: 1 high likelihood AD ... 4 no AD -- reverse coded, opposite to Braak. "
        "AUTOPSY ONLY.", dichotomy=("<=", 2), role="outcome_only"),
]

VAR_BY_KEY = {v.key: v for v in VARS}


def vars_in(panel, group=None):
    return [v for v in VARS if v.panel == panel and (group is None or v.group == group)]


# ---------------------------------------------------------------- panels
class Panel:
    """A rectangle of rows plus the column names the generic §2 component needs.

    outcome_col is the REFERENCE OUTCOME for §2.6/§2.7: a 0/1 column. Which one is
    appropriate differs by panel and is stated in the report, never assumed.
    """

    def __init__(self, name, df, person, order, outcome, at_risk, mildest,
                 age, year=None, version=None, outcome_label="", incident=None,
                 at_risk_incident=None):
        self.name = name
        self.df = df
        self.person = person
        self.order = order
        self.outcome = outcome
        self.at_risk = at_risk
        self.mildest = mildest
        self.age = age
        self.year = year
        self.version = version
        self.outcome_label = outcome_label
        self.incident = incident                  # stricter, at-risk-only variant
        self.at_risk_incident = at_risk_incident

    @property
    def grain(self):
        return self.name.split("_")[1]


# ---------------------------------------------------------------- NACC loading
NACC_KEEP = (["NACCID", "NACCVNUM", "PACKET", "FORMVER", "VISITYR", "NACCAGE", "NACCAGEB",
              "SEX", "EDUC", "NACCUDSD", "NACCALZD", "NACCTMCI", "CDRSUM", "CDRGLOB",
              "NACCMMSE", "MOCATOTS", "NACCGDS", "NOGDS", "INRELTO", "INRELY", "NACCNE4S"]
             + CDR_BOXES + FAQ_ITEMS)
_NACC_CACHE = os.path.join(CACHE, "nacc_subset.parquet")


def load_nacc(force=False):
    """Visit-level NACC frame, or None if no export is present on this machine.

    NACCID is factorised to an integer `pid` immediately and the raw id is dropped, so the
    cache and every downstream source-data CSV carry no NACC identifier.
    """
    p = nacc_path()
    if p is None:
        return None
    if os.path.exists(_NACC_CACHE) and not force:
        return pd.read_parquet(_NACC_CACHE)
    have = set(pd.read_csv(p, nrows=0).columns)
    use = [c for c in NACC_KEEP if c in have]
    d = pd.read_csv(p, usecols=use, low_memory=False)
    d = d[d["NACCID"].notna()].copy()                  # the export ends with a blank row
    d["pid"] = pd.factorize(d["NACCID"])[0]
    d = d.drop(columns=["NACCID"])
    for c in d.columns:
        if c != "pid":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["pid", "NACCVNUM"]).reset_index(drop=True)
    d.to_parquet(_NACC_CACHE, index=False)
    return d


def derive_faq(d):
    """FAQ total from the 10 items. 8 (not applicable) / 9 (unknown) / -4 are NOT zeros.

    Two columns: FAQTOTAL is STRICT (all ten items scored), FAQTOTAL_PRORATED accepts >=8
    items and scales up. §3.5 reports how much coverage the strict rule costs.
    """
    items = [c for c in FAQ_ITEMS if c in d.columns]
    if not items:
        d["FAQTOTAL"] = np.nan
        d["FAQ_N_ITEMS"] = 0
        d["FAQTOTAL_PRORATED"] = np.nan
        return d
    m = d[items].apply(pd.to_numeric, errors="coerce")
    m = m.mask(m.isin(list(FAQ_SENTINELS)))
    n = m.notna().sum(axis=1)
    s = m.sum(axis=1, min_count=1)
    d["FAQ_N_ITEMS"] = n
    d["FAQTOTAL"] = np.where(n == len(items), s, np.nan)
    d["FAQTOTAL_PRORATED"] = np.where(n >= 8, np.round(s * len(items) / n.replace(0, np.nan)), np.nan)
    return d


def nacc_panel(force=False):
    d = load_nacc(force=force)
    if d is None:
        return None
    d = derive_faq(d.copy())
    d["dem_now"] = (d["NACCUDSD"] == 4).astype(float)
    g = d.groupby("pid", sort=False)
    d["visit_idx"] = g.cumcount()
    d["n_visits"] = g["pid"].transform("size")
    d["next_udsd"] = g["NACCUDSD"].shift(-1)
    d["next_year"] = g["VISITYR"].shift(-1)
    d["next_age"] = g["NACCAGE"].shift(-1)
    d["has_next"] = d["next_udsd"].notna()
    d["next_dem"] = (d["next_udsd"] == 4).astype(float).mask(~d["has_next"])
    # incident variant: only rows NOT already demented are at risk of BECOMING demented
    d["at_risk_incident"] = d["has_next"] & (d["dem_now"] == 0)
    # noise floor: both this and the next visit are UDS-normal (NACCUDSD == 1)
    d["mildest_pair"] = (d["NACCUDSD"] == 1) & (d["next_udsd"] == 1)
    return Panel("nacc_visit", d, person="pid", order="NACCVNUM", outcome="next_dem",
                 at_risk="has_next", mildest="mildest_pair", age="NACCAGE",
                 year="VISITYR", version="FORMVER",
                 outcome_label="NACCUDSD = 4 (dementia) at the NEXT visit",
                 incident="next_dem", at_risk_incident="at_risk_incident")


# ---------------------------------------------------------------- RADC loading
RADC_FILES = {"cs": "cross-sectional-data-gk.xlsx",
              "lo": "longitudinal_data_gk.xlsx",
              "cl": "ROSMAP_clinical.csv"}
_RADC_CACHE = os.path.join(CACHE, "radc.pkl")


def load_radc(force=False):
    if os.path.exists(_RADC_CACHE) and not force:
        return pd.read_pickle(_RADC_CACHE)
    cs = pd.read_excel(os.path.join(RADC_DIR, RADC_FILES["cs"]))
    lo = pd.read_excel(os.path.join(RADC_DIR, RADC_FILES["lo"]))
    cl = pd.read_csv(os.path.join(RADC_DIR, RADC_FILES["cl"]))
    for f, c in ((cs, "study"), (lo, "study"), (cl, "Study")):
        f[c] = f[c].astype(str).str.strip()
    out = (cs, lo, cl)
    pd.to_pickle(out, _RADC_CACHE)
    return out


def radc_panels(force=False):
    """(visit panel, person panel).

    NOTE on the age axis: this release has no per-visit age and no visit date, so
    age_at_visit = age_bl + fu_year, exactly as round 2 established (its §4 verified the
    annual grid). Everything time-related here inherits that reconstruction.
    """
    cs, lo, cl = load_radc(force=force)
    v = lo.merge(cs[["projid", "age_bl", "age_first_ad_dx", "died", "msex", "educ"]],
                 on="projid", how="left", validate="many_to_one")
    v["age_at_visit"] = v["age_bl"] + v["fu_year"]
    v = v.sort_values(["projid", "fu_year"]).reset_index(drop=True)
    g = v.groupby("projid", sort=False)
    v["visit_idx"] = g.cumcount()
    v["n_visits"] = g["projid"].transform("size")
    v["ad_now"] = (v["age_first_ad_dx"].notna()
                   & (v["age_first_ad_dx"] <= v["age_at_visit"] + 1e-9)).astype(float)
    v["next_age"] = g["age_at_visit"].shift(-1)
    v["has_next"] = v["next_age"].notna()
    v["next_ad"] = ((v["age_first_ad_dx"].notna())
                    & (v["age_first_ad_dx"] <= v["next_age"] + 1e-9)).astype(float).mask(~v["has_next"])
    # The codebook says age_first_ad_dx is simply NOT RECORDED for participants who were
    # already demented at baseline. Left in, those people look permanently at-risk with a
    # permanent 0 label, and because they sit at the bottom of every cognitive scale they
    # flatten the whole dose-response curve. Round 2's labels.py drops them for the same
    # reason; the screening cut is the conventional MMSE < 24.
    bl = v[v["visit_idx"] == 0].set_index("projid")
    prevalent = bl.index[bl["age_first_ad_dx"].isna() & (bl["cts_estmmse30"] < 24)]
    v["suspected_prevalent_dementia"] = v["projid"].isin(prevalent)
    v["analysable"] = ~v["suspected_prevalent_dementia"]
    v["has_next"] = v["has_next"] & v["analysable"]
    v["at_risk_incident"] = v["has_next"] & (v["ad_now"] == 0)
    v["mildest_pair"] = v["has_next"] & (v["ad_now"] == 0) & (v["next_ad"] == 0)

    # ---- person level. Reference outcomes are chosen to avoid circularity:
    #      clinical dx variables are judged against PATHOLOGY, pathology against CLINICAL dx.
    per = cs.merge(cl[["projid", "dcfdx_lv", "cogdx", "cts_mmse30_lv"]],
                   on="projid", how="left", validate="one_to_one")
    per["braak_ad"] = (pd.to_numeric(per["braaksc"], errors="coerce") >= 4).astype(float) \
        .mask(per["braaksc"].isna())
    per["has_autopsy"] = per["braaksc"].notna()
    per["dem_lv"] = (pd.to_numeric(per["dcfdx_lv"], errors="coerce") >= 4).astype(float) \
        .mask(per["dcfdx_lv"].isna())
    per["has_dx"] = per["dcfdx_lv"].notna()

    vp = Panel("radc_visit", v, person="projid", order="fu_year", outcome="next_ad",
               at_risk="has_next", mildest="mildest_pair", age="age_at_visit",
               year=None, version="study",
               outcome_label="AD dementia by the NEXT observed cycle (proxy rebuilt from "
                             "person-level age_first_ad_dx; suspected baseline-prevalent "
                             "dementia excluded)",
               incident="next_ad", at_risk_incident="at_risk_incident")
    pp = Panel("radc_person", per, person="projid", order=None, outcome="braak_ad",
               at_risk="has_autopsy", mildest=None, age="age_bl",
               year=None, version="study",
               outcome_label="Braak stage >= 4 at autopsy (moderate-severe NFT pathology)")
    return vp, pp


# per-variable reference-outcome overrides (avoids braaksc predicting braaksc>=4)
PERSON_OUTCOME_OVERRIDE = {
    "radc:braaksc": ("dem_lv", "has_dx", "clinical dx dcfdx_lv >= 4 (AD dementia) at the last valid cycle"),
    "radc:ceradsc": ("dem_lv", "has_dx", "clinical dx dcfdx_lv >= 4 (AD dementia) at the last valid cycle"),
    "radc:niareagansc": ("dem_lv", "has_dx", "clinical dx dcfdx_lv >= 4 (AD dementia) at the last valid cycle"),
}


# ---------------------------------------------------------------- stats helpers
def wilson(k, n, z=1.959964):
    """Wilson score interval -- behaves at p near 0/1 where the normal interval does not."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def shannon(counts):
    c = np.asarray([x for x in counts if x > 0], float)
    if c.sum() == 0 or len(c) < 2:
        return 0.0, 0.0
    p = c / c.sum()
    h = float(-(p * np.log(p)).sum())
    return h, h / np.log(len(c))


def icc1(values, groups):
    """One-way random-effects ICC(1) = between-person share of total variance.

    Returned CLIPPED to [0,1] plus the raw value: a negative raw ICC means within-person
    variance exceeds between-person variance, i.e. a measurement-noise problem.
    """
    df = pd.DataFrame({"y": pd.to_numeric(values, errors="coerce"), "g": np.asarray(groups)}).dropna()
    k = df["g"].nunique()
    N = len(df)
    if k < 2 or N <= k:
        return np.nan, k, N, np.nan
    grand = df["y"].mean()
    gs = df.groupby("g")["y"]
    n_i = gs.size().to_numpy(float)
    m_i = gs.mean().to_numpy(float)
    ssb = float((n_i * (m_i - grand) ** 2).sum())
    ssw = float(((df["y"] - df["g"].map(gs.mean())) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if not np.isfinite(denom) or denom == 0:
        return np.nan, k, N, np.nan
    raw = (msb - msw) / denom
    return float(np.clip(raw, 0, 1)), k, N, float(raw)


def auc_single(score, y):
    """Rank AUC of one continuous score against a 0/1 label. NaNs dropped pairwise."""
    s = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    if len(y) < 20 or y.min() == y.max():
        return np.nan, int(len(y))
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = float(y.sum()), float((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), int(len(y))


def dichotomise(x, spec):
    """Apply a Var.dichotomy to a numeric series -> float 0/1 with NaN preserved."""
    if spec is None:
        return None
    op, thr = spec
    x = pd.to_numeric(x, errors="coerce")
    f = {"<": x < thr, "<=": x <= thr, ">": x > thr, ">=": x >= thr,
         "==": x == thr}[op].astype(float)
    return f.mask(x.isna())


def dichotomy_str(spec):
    return "n/a" if spec is None else f"{spec[0]} {_num(spec[1])}"


# ---------------------------------------------------------------- bookkeeping
def vardir(v, *sub):
    """eda_out/scales/<varname>/... -- one directory per variable, per the spec."""
    d = os.path.join(OUT, f"{v.dataset}_{v.spec_name}", *sub)
    os.makedirs(d, exist_ok=True)
    return d


def report(text):
    with open(REPORT, "a") as f:
        f.write(text.rstrip() + "\n\n")


def reset_report():
    for p in (REPORT, SUMMARY):
        if os.path.exists(p):
            os.remove(p)


def summary_put(**kw):
    d = {}
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            d = json.load(f)
    for k, val in kw.items():
        d[k] = jsonable(val)
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)


def jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x) for x in v]
    if isinstance(v, pd.Series):
        return jsonable(v.to_dict())
    return v


def save_table(df, stem, subdir=None, index=False):
    d = subdir or TABLES
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{stem}.csv")
    df.to_csv(p, index=index)
    return p


def gate(passed, verdict):
    """`passed` True = the permissive branch (spec's plan survives as written)."""
    return f"- **决策门 [{'PASS' if passed else 'ACTION'}]** {verdict}"


def md_table(df, floatfmt="{:.3f}", max_rows=None):
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)
    def cell(x):
        if isinstance(x, float) and np.isfinite(x):
            return floatfmt.format(x)
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return str(x)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| … {len(df) - max_rows} more rows in the CSV " + "| " * (len(cols) - 1) + "|")
    return "\n".join(lines)
