"""
common.py -- shared loading / bookkeeping for the ROUND 2 RADC EDA.

Round 1 (Delphi-2M/eda/eda_radc.py -> Delphi-2M/out-radc-eda/) is untouched. This round is a
separate, spec-driven pass whose job is to answer seven build-or-not questions before any
model code is written (see eda_out/REPORT.md, section 10).

Everything here is derived from THREE release tables under data/RADC/:
  cross-sectional-data-gk.xlsx  person-level, 4428 x 24
  longitudinal_data_gk.xlsx     visit-level,  36138 x 29
  ROSMAP_clinical.csv           person-level, 3584 x 18   (ROS+MAP only, ages de-identified)

Variable semantics below are transcribed from the RADC codebook
(data/RADC/RADC_codebook_data_set_1736_08-13-2026.pdf, decoded to codebook_decoded.txt).
Nothing is inferred from the variable NAME alone -- several are reverse coded.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))              # Delphi-2M/eda_round2/
DELPHI = os.path.dirname(HERE)                                 # Delphi-2M/
ROOT = os.path.dirname(DELPHI)
RADC = os.path.join(ROOT, "data", "RADC")

OUT = os.path.join(HERE, "eda_out")
FIGS = os.path.join(OUT, "figs")
TABLES = os.path.join(OUT, "tables")
REPORT = os.path.join(OUT, "REPORT.md")
SUMMARY = os.path.join(OUT, "summary.json")
for _d in (OUT, FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, DELPHI)
from figure2.plotting_style import setup_style, save_fig, save_data, REF_GREY  # noqa: E402

# ---------------------------------------------------------------- palette
# Categorical hues, FIXED order, never cycled. Validated with the dataviz validator
# (light surface #fcfcfb): lightness band PASS / chroma floor PASS / CVD separation PASS
# (worst adjacent dE 11.4 protan) / normal-vision floor PASS (worst 15.6).
# One WARN: #E69F00 sits at 2.19:1 vs surface -- discharged everywhere by an always-on
# legend + direct labels + the per-panel source-data CSV.
CAT = ["#0072B2", "#D55E00", "#E69F00", "#009E73"]
C1, C2, C3, C4 = CAT
# Sequential = ONE hue, light -> dark (magnitude: missingness rates, occupancy shares).
SEQ = ["#deebf7", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
SEQ_CMAP = "Blues"
# Diverging = two poles + a NEUTRAL grey midpoint (polarity: change since last visit).
DIV = ["#0072B2", "#9ecae1", "#d9d9d9", "#F4A582", "#D55E00"]

FILES = {
    "cross_sectional": "cross-sectional-data-gk.xlsx",
    "longitudinal": "longitudinal_data_gk.xlsx",
    "rosmap_clinical": "ROSMAP_clinical.csv",
}

# ---------------------------------------------------------------- codebook-confirmed metadata
# type:  cont | ordinal | nominal | binary | id | time
# grain: person | visit
# Every `note` below is a transcription of the codebook entry, not a guess.
VAR_META = {
    # ---- cross-sectional-data-gk.xlsx
    "projid":            ("id",      "person", "RADC participant id; join key across all three tables"),
    "study":             ("nominal", "person", "ROS / MAP / LATC cohort label"),
    "apoe_genotype":     ("nominal", "person", "22/23/24/33/34/44; NOT ordinal -- e2 is protective, e4 risk"),
    "age_first_ad_dx":   ("cont",    "person", "age at FIRST cycle with dcfdx in {4,5}; missing if demented at baseline"),
    "cogng_demog_slope": ("cont",    "person", "random slope of cogn_global from an LME over the WHOLE follow-up"),
    "age_bl":            ("cont",    "person", "age at baseline assessment, (visit date - dob)/365.25; NOT capped"),
    "died":              ("binary",  "person", "1 died / 0 alive as of the data freeze"),
    "educ":              ("cont",    "person", "years of regular school, reported at baseline"),
    "msex":              ("binary",  "person", "1 male / 0 female"),
    "ldai_bl":           ("cont",    "person", "lifetime daily alcohol intake at baseline, drinks/day"),
    "smoking_bl":        ("ordinal", "person", "0 never / 1 former / 2 current, at baseline"),
    "braaksc":           ("ordinal", "person", "Braak NFT stage 0-6, ascending severity; AUTOPSY ONLY"),
    "ceradsc":           ("ordinal", "person", "CERAD 1 definite / 2 probable / 3 possible / 4 no AD -- REVERSE coded; AUTOPSY ONLY"),
    "gpath":             ("cont",    "person", "global AD pathology burden, 3 measures x 5 regions; AUTOPSY ONLY"),
    "pmi":               ("cont",    "person", "post-mortem interval (hours); AUTOPSY ONLY -- presence alone implies death"),
    "amylsqrt_est_8reg": ("cont",    "person", "sqrt amyloid density, 8 regions; AUTOPSY ONLY"),
    "lewydx_st4":        ("ordinal", "person", "Lewy body disease stage; AUTOPSY ONLY"),
    "tangsqrt_est_8reg": ("cont",    "person", "sqrt tangle density, 8 regions; AUTOPSY ONLY"),
    "tdp_st4":           ("ordinal", "person", "TDP-43 stage; AUTOPSY ONLY"),
    "arteriol_scler":    ("ordinal", "person", "arteriolosclerosis grade; AUTOPSY ONLY"),
    "caa_4gp":           ("ordinal", "person", "cerebral amyloid angiopathy, 4 groups; AUTOPSY ONLY"),
    "cvda_4gp2":         ("ordinal", "person", "cerebral atherosclerosis, 4 groups; AUTOPSY ONLY"),
    "ci_num2_mct":       ("binary",  "person", ">=1 chronic microinfarct; AUTOPSY ONLY"),
    "ci_num2_tct":       ("binary",  "person", ">=1 chronic infarct; AUTOPSY ONLY"),
    # ---- longitudinal_data_gk.xlsx
    "fu_year":           ("time",    "visit",  "follow-up cycle index; 0 = baseline. THE ONLY time field in the release"),
    "log_hcrp":          ("cont",    "visit",  "log C-reactive protein"),
    "log_hil6":          ("cont",    "visit",  "log interleukin-6"),
    "log_htnfa":         ("cont",    "visit",  "log TNF-alpha"),
    "gfr_mdrs":          ("cont",    "visit",  "estimated glomerular filtration rate (MDRD)"),
    "glucose":           ("cont",    "visit",  "blood glucose"),
    "hba1c":             ("cont",    "visit",  "hemoglobin A1c"),
    "hdlchlstrl":        ("cont",    "visit",  "HDL cholesterol"),
    "ldlchlstrl":        ("cont",    "visit",  "LDL cholesterol"),
    "r_stroke":          ("ordinal", "visit",  "clinician stroke dx: 1 highly probable / 2 probable / 3 possible / 4 NOT present -- REVERSE coded"),
    "cogn_global":       ("cont",    "visit",  "global cognition z, mean of 19 tests; z uses the BASELINE-cycle cohort mean/sd"),
    "cts_estmmse30":     ("cont",    "visit",  "estimated MMSE, 0-30 scale"),
    "r_depres":          ("ordinal", "visit",  "clinician major depression dx: 1 highly probable ... 4 NOT present -- REVERSE coded"),
    "bmi":               ("cont",    "visit",  "body mass index, measured at each visit"),
    "dbp_avg":           ("cont",    "visit",  "diastolic blood pressure, averaged"),
    "hypertension_cum":  ("binary",  "visit",  "self-reported hypertension EVER UP TO this cycle -- cumulative, monotone"),
    "sbp_avg":           ("cont",    "visit",  "systolic blood pressure, averaged"),
    "dm_cum":            ("binary",  "visit",  "self-reported diabetes ever up to this cycle -- cumulative, monotone"),
    "chf_cum":           ("binary",  "visit",  "congestive heart failure ever up to this cycle -- cumulative, monotone"),
    "claudication_cum":  ("binary",  "visit",  "claudication ever up to this cycle -- cumulative, monotone"),
    "heart_cum":         ("binary",  "visit",  "heart condition ever up to this cycle -- cumulative, monotone"),
    "antihyp_rx":        ("binary",  "visit",  "antihypertensive taken in the 2 weeks before THIS visit -- point-in-time"),
    "diabetes_rx":       ("binary",  "visit",  "diabetes medication at this visit -- point-in-time"),
    "statin_rx":         ("binary",  "visit",  "statin at this visit -- point-in-time"),
    "ad_rx":             ("binary",  "visit",  "AD medication at this visit -- point-in-time"),
    "berlin_risk_class": ("binary",  "visit",  "Berlin questionnaire high risk of sleep apnea"),
    "psqi_sum":          ("cont",    "visit",  "Pittsburgh sleep quality index summary"),
    # ---- ROSMAP_clinical.csv
    "Study":             ("nominal", "person", "cohort label in the ROSMAP release"),
    "race":              ("nominal", "person", "self-reported race"),
    "spanish":           ("nominal", "person", "Spanish/Hispanic ethnicity, 1 yes / 2 no"),
    "age_at_visit_max":  ("cont",    "person", "age at last visit -- DE-IDENTIFIED, top coded as the string '90+'"),
    "age_death":         ("cont",    "person", "age at death -- DE-IDENTIFIED, top coded as the string '90+'"),
    "cts_mmse30_first_ad_dx": ("cont", "person", "MMSE at the first AD dx cycle"),
    "cts_mmse30_lv":     ("cont",    "person", "MMSE at the LAST valid cycle"),
    "cogdx":             ("ordinal", "person", "FINAL consensus cognitive dx (post-mortem informed for autopsied)"),
    "dcfdx_lv":          ("ordinal", "person", "clinical dx at the LAST valid cycle -- 1 NCI / 2 MCI / 3 MCI+ / 4 AD / 5 AD+ / 6 other dementia"),
    "individualID":      ("id",      "person", "Synapse/ROSMAP identifier"),
}

# The per-visit modelling candidates: everything in the longitudinal table that is a
# measurement rather than a key.
LONG_CANDIDATES = [c for c, (t, g, _) in VAR_META.items()
                   if g == "visit" and t in ("cont", "ordinal", "binary")]

MMSE_BINS = [-0.001, 17, 23, 26, 30]
MMSE_NAMES = ["severe 0-17", "mild-mod 18-23", "borderline 24-26", "normal 27-30"]


# ---------------------------------------------------------------- loading
def _strip(s):
    return s.astype(str).str.strip()


def parse_capped_age(s):
    """ROSMAP_clinical stores top-coded ages as the literal string '90+'.

    Returns (numeric_age, is_capped). Capped rows get 90.0 so they are usable for
    'at least this old' logic, but is_capped must gate anything that treats age as exact.
    """
    raw = s.astype(str).str.strip()
    capped = raw.str.endswith("+")
    num = pd.to_numeric(raw.str.rstrip("+"), errors="coerce")
    return num, (capped.fillna(False) & num.notna()).astype(bool)


def load_raw():
    cs = pd.read_excel(os.path.join(RADC, FILES["cross_sectional"]))
    lo = pd.read_excel(os.path.join(RADC, FILES["longitudinal"]))
    cl = pd.read_csv(os.path.join(RADC, FILES["rosmap_clinical"]))
    cs["study"] = _strip(cs["study"])
    lo["study"] = _strip(lo["study"])
    cl["Study"] = _strip(cl["Study"])
    return cs, lo, cl


def build_visits(cs, lo):
    """Visit-level frame with the derived age axis.

    THE RELEASE HAS NO PER-VISIT AGE AND NO VISIT DATE. age_at_visit is reconstructed as
    age_bl + fu_year, which is exact only if every cycle lands on its anniversary. Section 4
    tests that assumption against ROSMAP_clinical.age_at_visit_max and reports the residual.
    """
    v = lo.merge(
        cs[["projid", "age_bl", "study", "msex", "educ", "apoe_genotype",
            "age_first_ad_dx", "died", "smoking_bl", "ldai_bl"]].rename(columns={"study": "study_cs"}),
        on="projid", how="left", validate="many_to_one",
    )
    v["age_at_visit"] = v["age_bl"] + v["fu_year"]
    v = v.sort_values(["projid", "fu_year"]).reset_index(drop=True)
    g = v.groupby("projid", sort=False)
    v["d_fu"] = g["fu_year"].diff()                 # cycles since previous OBSERVED visit
    v["d_age"] = v["d_fu"]                          # identical by construction -- see section 4
    v["visit_idx"] = g.cumcount()
    v["n_visits"] = g["projid"].transform("size")
    v["is_last"] = v["visit_idx"] == (v["n_visits"] - 1)
    v["age_last"] = g["age_at_visit"].transform("max")
    # AD status AT this visit, from the person-level age at first dx.
    v["ad_now"] = (v["age_first_ad_dx"].notna() & (v["age_first_ad_dx"] <= v["age_at_visit"] + 1e-9))
    v["mmse_bin"] = pd.cut(v["cts_estmmse30"], MMSE_BINS, labels=MMSE_NAMES)
    return v


def load():
    cs, lo, cl = load_raw()
    return cs, lo, cl, build_visits(cs, lo)


# ---------------------------------------------------------------- bookkeeping
def summary_put(**kw):
    d = {}
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            d = json.load(f)
    for k, v in kw.items():
        d[k] = _jsonable(v)
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def report(text):
    with open(REPORT, "a") as f:
        f.write(text.rstrip() + "\n\n")


def reset_report():
    for p in (REPORT, SUMMARY):
        if os.path.exists(p):
            os.remove(p)


def save_table(df, stem, index=False):
    p = os.path.join(TABLES, f"{stem}.csv")
    df.to_csv(p, index=index)
    return p


def fig(stem):
    return os.path.join(FIGS, stem)


def gate(passed, verdict):
    """Format a decision-gate line. `passed` True = the good/permissive branch."""
    return f"- **决策门 [{'PASS' if passed else 'ACTION'}]** {verdict}"
