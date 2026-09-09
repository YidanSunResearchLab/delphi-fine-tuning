"""
tokenize_nacc_ad.py -- AD variant of the NACC -> Delphi tokenizer.

Based on Delphi-AD/tokenize_nacc.py; identical except for two deliberate changes
agreed for AD (everything else -- baseline, diseases, death, off-by-one, smoking,
death timing -- uses byte-for-byte the same logic):

  1. ALL COGNITIVE LABELS ARE KEPT (not NACCUDSD-only). The ordinal scales emitted are
     NACCUDSD (idx 106-109), the six CDR box scores (111-139), the nine FAQ domains
     (140-175), the twelve NPI-Q symptoms (176-223), GDS (224-227) and MoCA (29-32).
     Each is a longitudinal ordinal STATE, evaluated as per-scale staging downstream.

     EVERY COMPOSITE IS TOKENIZED PER ITEM, NOT AS ITS TOTAL. Three sum-scores have been
     replaced by the items they are summed from -- CDRSUM (was 24-28), the FAQ total (was
     33-36) and the NPI-Q total (was 37-40). All three totals are now RETIRED dead slots,
     kept only so every other token id stays where it was. The reasoning is the same in
     each case and is spelled out for CDR below; in short, a total is a deterministic
     function of its items, so feeding both is redundant AND is same-visit leakage, and
     summing destroys which DOMAIN moved. Domain-level trajectories are themselves
     prediction targets here, not just features.

     GDS (224-227) is NEW -- the 15-item Geriatric Depression Scale total was not in the
     vocabulary at all. It is emitted as a 4-bin TOTAL, not per item: the 15 items are
     binary and flip constantly (30 tokens, ~991k events for very little signal), whereas
     the total is a well-established ordinal at 86.7% coverage for 4 tokens.

     MoCA (29-32) is deliberately LEFT as a total. Its ~19 sub-items exist, but MoCA is
     only administered in UDS v3 -- 36% visit coverage -- so 49 tokens would buy a scale
     that is absent from two thirds of visits.

     CDR, THE WORKED EXAMPLE. The CDRSUM sum-of-boxes tokens
     (24-28) are RETIRED. In their place each of the six domains -- MEMORY, ORIENT,
     JUDGMENT, COMMUN, HOMEHOBB, PERSCARE -- gets its own token block and its own
     keep-transitions run, so domain-specific decline survives instead of being summed
     away. Three reasons (eda_round3 s3.3):
       (a) CDRSUM is the deterministic sum of the six boxes (verified 100% equal on
           every row where both are present), so feeding both is redundant AND is
           same-visit leakage.
       (b) each box is a clean 5-level ordinal {0, 0.5, 1, 2, 3}, whereas the CDRSUM
           0.5 grid is lumpy -- several nominal levels are structurally unreachable.
       (c) the six domains have different ICCs and different change rates; the total
           collapses them into one number.
     PERSCARE HAS NO 0.5 LEVEL -- four tokens, not five. That asymmetry is exactly why
     the CDRSUM grid was lumpy in the first place. eda_round3 s2 judged PERSCARE `drop`
     (83% of visits pile at 0); it is kept here by explicit decision, so expect it to be
     a near-constant token in most sequences.

  2. EVERY COGNITIVE SCALE USES KEEP-TRANSITIONS, per scale (not keep-first).
     Run-length-encode within each scale so every bin CHANGE survives -- decline
     (Normal->MCI->Dementia) AND recovery (MCI->Normal) -- collapsing only
     consecutive same-bin repeats (keep-first deletes ~44% of recoveries; see
     Delphi-AD/README). Recovery-tracking is exactly why all scales are kept.
     All NON-cognitive tokens keep the original keep-first convention: one token
     at age-of-first-onset.

Emits bin_token = spec_index - 1 so Delphi's get_batch (+1) realigns to model space
(disk token range 1-226; model space 0=Padding, 1=No event, 2..227 content);
labels.csv is written in MODEL space.

Output: Delphi-2M/out_ad/{nacc_all.bin, labels.csv, nacc_pidmap.csv}
Run:  python -u tokenize_nacc_ad.py   (normally invoked via make_dataset.py)
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.dirname(os.path.abspath(__file__))                # data_prep/
CSV  = os.path.join(HERE, "investigator_nacc72.csv")
SPEC = os.path.join(SCRIPTS, "Vocabulary NACC.xlsx")   # ships with the tokenizer, not the data
OUT  = os.path.join(HERE, "out_ad"); os.makedirs(OUT, exist_ok=True)

# ----------------------------- DECISIONS (flags) -----------------------------
# Candidate refinements to these defaults (smoking status/pack-years, diabetes-split,
# vision/hearing, scope of the ~45 extra tokens) are documented in candidate_improvements.md.
CFG = dict(
    SMOKING_HIGH_MIN_YEARS = 20,        # SMOKYRS >= this -> high
    SMOKING_UNKNOWN_YEARS_TO = "mid",   # TOBAC100=1 but SMOKYRS missing -> this bin
    # OBSOLETE since FAQ became nine independent per-domain scales: each item now stands
    # or falls on its own, so one missing item no longer discards the other eight. Kept
    # only to document that the old all-or-nothing rule threw away 3,773 visits (1.8%)
    # that had at least one valid item. Not read anywhere.
    FAST_REQUIRE_COMPLETE = None,
)
MISSING = {-4, 8, 9, 88, 99, 888, 999, 888.8}  # generic NACC missing codes

# ----------------------------- load raw --------------------------------------
MED = {41:"NACCAHTN",42:"NACCHTNC",43:"NACCACEI",44:"NACCAAAS",45:"NACCBETA",
       46:"NACCCCBS",47:"NACCDIUR",48:"NACCVASD",49:"NACCANGI",50:"NACCLIPL",
       51:"NACCNSD",52:"NACCADEP",53:"NACCAPSY",54:"NACCAANX",55:"NACCADMD",
       56:"NACCPDMD",57:"NACCEMD",58:"NACCEPMD",59:"NACCDBMD"}
# disease: idx -> (column, valid_values)
DIS = {22:("VISWCORR",[0]),23:("HEARWAID",[0]),
       60:("CVHATT",[1,2]),61:("HATTMULT",[1]),62:("CVAFIB",[1,2]),63:("CVANGIO",[1,2]),
       64:("CVBYPASS",[1,2]),65:("CVPACDEF",[1,2]),66:("CVCHF",[1,2]),67:("CVANGINA",[1,2]),
       68:("CVHVALVE",[1,2]),69:("CVOTHR",[1,2]),70:("CBSTROKE",[1,2]),71:("STROKMUL",[1]),
       72:("CBTIA",[1,2]),73:("TIAMULT",[1]),74:("PD",[1]),75:("PDOTHR",[1]),
       76:("SEIZURES",[1,2]),77:("TBI",[1,2]),78:("NCOTHR",[1,2]),
       82:("HYPERTEN",[1,2]),83:("HYPERCHO",[1,2]),84:("B12DEF",[1,2]),85:("THYROID",[1,2]),
       89:("INCONTU",[1,2]),90:("INCONTF",[1,2]),91:("APNEA",[1,2]),92:("RBD",[1,2]),
       93:("INSOMN",[1,2]),94:("OTHSLEEP",[1,2]),95:("ALCOHOL",[1,2]),96:("ABUSOTHR",[1,2]),
       97:("PTSD",[1,2]),98:("BIPOLAR",[1,2]),99:("SCHIZ",[1,2]),100:("DEP2YRS",[1]),
       101:("DEPOTHR",[1]),102:("ANXIETY",[1,2]),103:("OCD",[1,2]),104:("NPSYDEV",[1,2]),
       105:("PSYCDIS",[1,2])}
# CDR box scores -- six domains, each its own ordinal scale (replaces the CDRSUM total).
# column -> [(raw level, model idx), ...] in low->high severity order. Contiguous 111-139.
# PERSCARE has no 0.5 level, hence 4 tokens where the others have 5.
_LV5 = (0, 0.5, 1, 2, 3)
_LV4 = (0, 1, 2, 3)
CDR_BOX_LEVELS, _i = {}, 111
for _col, _lv in [("MEMORY",_LV5),("ORIENT",_LV5),("JUDGMENT",_LV5),
                  ("COMMUN",_LV5),("HOMEHOBB",_LV5),("PERSCARE",_LV4)]:
    CDR_BOX_LEVELS[_col] = [(v, _i + k) for k, v in enumerate(_lv)]
    _i += len(_lv)
assert _i == 140, _i

FAQ = ["BILLS","TAXES","GAMES","STOVE","MEALPREP","EVENTS","PAYATTN","REMDATES","TRAVEL"]
NPI = ["DELSEV","HALLSEV","AGITSEV","DEPDSEV","ANXSEV","ELATSEV","APASEV","DISNSEV",
       "IRRSEV","MOTSEV","NITESEV","APPSEV"]
NPIP = ["DEL","HALL","AGIT","DEPD","ANX","ELAT","APA","DISN","IRR","MOT","NITE","APP"]

# FAQ: nine domains, each a clean 4-level ordinal 0-3 (Normal / has difficulty / needs
# assistance / dependent). Replaces the summed FAQ total that used to live at 33-36.
FAQ_LEVELS = {c: [(v, 140 + 4*k + v) for v in (0,1,2,3)] for k, c in enumerate(FAQ)}

# NPI-Q: twelve symptoms. Severity is only RECORDED when the symptom is present, so the
# per-item level is severity-if-present, 0-if-administered-and-absent, and nothing at all
# if the item was not administered. That per-item presence flag is strictly better than the
# old global "any presence flag valid" test, which let one administered item speak for all
# twelve. Replaces the summed NPI-Q total that used to live at 37-40.
NPI_LEVELS = {sev: [(v, 176 + 4*k + v) for v in (0,1,2,3)] for k, sev in enumerate(NPI)}

# GDS: NEW scale. 4 bins of the 15-item total (0-15), standard clinical cutpoints.
GDS_BINS = [(0, 4, 224), (5, 8, 225), (9, 11, 226), (12, 15, 227)]
BASE = ["NACCID","BIRTHYR","BIRTHMO","VISITYR","VISITMO","VISITDAY","SEX","NACCAPOE",
        "NACCBMI","EDUC","ALCOCCAS","ALCFREQ","TOBAC100","SMOKYRS","NACCMOCA",
        "NACCUDSD","NACCDIED","NACCYOD","NACCMOD","DIABETES","DIABTYPE","ARTHRIT","ARTHTYPE",
        "NACCGDS"]
NEED = (set(BASE)|set(MED.values())|set(c for c,_ in DIS.values())|set(FAQ)|set(NPI)
        |set(NPIP)|set(CDR_BOX_LEVELS))

print("loading raw CSV ...")
df = pd.read_csv(CSV, usecols=lambda c: c in NEED, low_memory=False)
P0 = df["NACCID"].nunique(); print(f"  {len(df)} visits, {P0} patients")

num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
df["birth"] = pd.to_datetime(dict(year=num("BIRTHYR"), month=num("BIRTHMO"), day=15), errors="coerce")
# a missing/blank VISITDAY falls back to mid-month (15) rather than dropping the whole visit;
# sentinels (-4/99) are bounded into [1,28] by the clip. (VISITYR/VISITMO must still be present.)
df["visit"] = pd.to_datetime(dict(year=num("VISITYR"), month=num("VISITMO"),
                                  day=num("VISITDAY").fillna(15).clip(1,28)), errors="coerce")
df["vage"]  = (df["visit"] - df["birth"]).dt.days

ev = []   # list of (pid_series, age_series, idx)
def push(mask, age, idx):
    m = mask & df["birth"].notna()
    if m.any():
        ev.append(pd.DataFrame({"pid":df.loc[m,"NACCID"].values,
                                "age":np.asarray(age)[m.values] if np.ndim(age) else age,
                                "idx":idx}))

# ---- static (age 0): sex, APOE ----
push(num("SEX")==1, 0, 2); push(num("SEX")==2, 0, 3)
for v,idx in zip(range(1,7), range(16,22)): push(num("NACCAPOE")==v, 0, idx)

# ---- background binned (visit age) ----
bmi=num("NACCBMI"); a=df["vage"].values
push(bmi.between(0,22), a, 4); push((bmi>22)&(bmi<=28), a, 5); push((bmi>28)&(bmi<=100), a, 6)
edu=num("EDUC"); push(edu.between(0,12),a,13); push(edu.between(13,16),a,14); push(edu.between(17,36),a,15)
occ=num("ALCOCCAS"); frq=num("ALCFREQ")
push((occ==0)|(frq==0),a,10); push(frq.isin([1,2]),a,11); push(frq.isin([3,4]),a,12)

# ---- smoking (PER-PATIENT trait; TOBAC100 is monotonic, so resolve once/patient
#      to avoid contradictory low+mid/high tokens from inconsistent per-visit records) ----
# SMOKYRS is a YEAR COUNT, so 8 and 9 are VALID values (unlike categorical NACC vars where
# 8/9 mean "unknown"). Use a numeric-missing set that excludes 8/9 -- the generic MISSING would
# silently drop a genuine "8 years" / "9 years" record.
MISSING_YEARS = {-4, 88, 99, 888, 999, 888.8}
t100=num("TOBAC100"); syr=num("SMOKYRS"); syr_valid=~syr.isin(list(MISSING_YEARS))
syrv = syr.where(syr_valid)
pp = pd.DataFrame({"pid": df["NACCID"], "t100": t100, "syrv": syrv, "vage": df["vage"]}).groupby("pid").agg(
        ever     =("t100", lambda s: (s == 1).any()),   # ever smoked >100 cigs (TOBAC100 monotonic)
        sawzero_t=("t100", lambda s: (s == 0).any()),   # explicitly recorded non-smoker
        sawzero_y=("syrv", lambda s: (s == 0).any()),   # explicitly recorded 0 years
        maxsyr   =("syrv", "max"),
        first_age=("vage", "min"))
ever = pp["ever"].fillna(False)
sawzero = pp["sawzero_t"].fillna(False) | pp["sawzero_y"].fillna(False)  # evidence of NON-smoking
ms = pp["maxsyr"]; msv = ms.notna()
HIGH = CFG["SMOKING_HIGH_MIN_YEARS"]
high = ever & msv & (ms >= HIGH)                             # ever, >=20 yrs
midk = ever & msv & (ms < HIGH)                              # ever, <20 yrs (incl recorded 0)
midu = (ever & ~msv) if CFG["SMOKING_UNKNOWN_YEARS_TO"] == "mid" else pd.Series(False, index=pp.index)
mid  = midk | midu                                          # ever-smoked, short/unknown duration
low  = (~ever) & sawzero                                    # NON-smoker only with evidence (missing != never)
# patients with NO smoking evidence at all (ever False, sawzero False) get NO token
for mask, idx in [(low, 7), (mid, 8), (high, 9)]:           # one smoking token/patient at first visit
    sel = pp[mask & pp["first_age"].notna()]
    if len(sel):
        ev.append(pd.DataFrame({"pid": sel.index.values, "age": sel["first_age"].values, "idx": idx}))

# ---- cognitive scales (visit age) -- CDR boxes, MoCA, FAST, NPI-Q ----
# Re-enabled: ALL cognitive labels are kept (not NACCUDSD-only). Each scale is binned
# into its reserved token ids and, like NACCUDSD, deduped with KEEP-TRANSITIONS below.
# CDR: six per-domain scales, NOT the CDRSUM total (see the module docstring). No binning
# is needed -- each raw box level IS a token, so exact equality replaces .between(), and
# the NACC sentinel 99 matches no level and is dropped without a MISSING lookup.
for col, levels in CDR_BOX_LEVELS.items():
    v = num(col)
    for val, idx in levels:
        push(v == val, a, idx)
moca=num("NACCMOCA")
for lo,hi,idx in [(26,30,29),(18,25,30),(10,17,31),(0,9,32)]:
    push(moca.between(lo,hi), a, idx)
# ---- FAQ: nine per-domain scales, NOT the summed total ----
# NACC FAQ code 8 = "not applicable / never did this activity" -> count as 0 (no
# disease-related functional impairment). Under the OLD all-or-nothing total this mattered
# a lot; now each domain stands alone, so a missing STOVE no longer discards the other
# eight (that rule was throwing away 3,773 visits, 1.8%).
for col, levels in FAQ_LEVELS.items():
    if col not in df: continue
    v = num(col).replace(8, 0)
    for val, idx in levels:
        push(v == val, a, idx)

# ---- NPI-Q: twelve per-symptom scales, NOT the summed total ----
# level = severity 1-3 when the symptom is PRESENT, 0 when the item was administered and
# the symptom is absent, nothing when the item was not administered.
for k, sevcol in enumerate(NPI):
    prescol = NPIP[k]
    if sevcol not in df or prescol not in df: continue
    pres = num(prescol)
    v = num(sevcol).where(lambda x: (x>=1)&(x<=3))
    v = v.where(pres == 1, other=0.0)          # administered & absent -> 0
    v = v.where(pres.isin([0,1]))              # not administered      -> drop
    for val, idx in NPI_LEVELS[sevcol]:
        push(v == val, a, idx)

# ---- GDS: NEW scale, 4 bins of the 15-item total ----
gds = num("NACCGDS").where(lambda x: (x>=0)&(x<=15))   # 88 = not assessed, falls outside
for lo, hi, idx in GDS_BINS:
    push(gds.between(lo, hi), a, idx)

# ---- medications (==1) ----
for idx,col in MED.items():
    if col in df: push(num(col)==1, a, idx)

# ---- diseases / vision / hearing ----
for idx,(col,vals) in DIS.items():
    if col in df: push(num(col).isin(vals), a, idx)

# ---- diabetes / arthritis (composite) ----
dia=num("DIABETES"); dt=num("DIABTYPE")
push((dia.isin([1,2]))&(dt==1),a,79); push((dia.isin([1,2]))&(dt==2),a,80); push((dia.isin([1,2]))&(dt.isin([3,9])),a,81)
art=num("ARTHRIT"); at=num("ARTHTYPE")
push((art.isin([1,2]))&(at==1),a,86); push((art.isin([1,2]))&(at==2),a,87); push((art.isin([1,2]))&(at.isin([3,9])),a,88)

# ---- cognitive (NACCUDSD) ----
for v,idx in zip([1,2,3,4],[106,107,108,109]): push(num("NACCUDSD")==v, a, idx)

# ---- DEATH (NACCYOD/NACCMOD) -- FIX ----
yod=num("NACCYOD"); mod=num("NACCMOD")
dy = yod.where((yod>1900)&(yod<2100)); dm = mod.where(mod.between(1,12)).fillna(6)
death_date = pd.to_datetime(dict(year=dy, month=dm, day=15), errors="coerce")
dage = (death_date - df["birth"]).dt.days
push((num("NACCDIED")==1)&dage.notna(), dage.values, 110)

# ----------------------------- assemble --------------------------------------
E = pd.concat(ev, ignore_index=True)
E = E.dropna(subset=["pid","age","idx"])
E["age"] = E["age"].astype(np.int64); E = E[E["age"]>=0]
E["idx"] = E["idx"].astype(int)

# DEDUP -- two regimes:
#   * ALL cognitive labels (CDR boxes 111-139, FAQ domains 140-175, NPI-Q symptoms
#     176-223, GDS 224-227, MoCA 29-32, NACCUDSD 106-109): KEEP-TRANSITIONS, PER SCALE. Each scale is a
#     longitudinal ordinal state, so run-length-encode it per patient -- keep the
#     first value plus every bin CHANGE (decline AND recovery), collapsing only
#     consecutive same-bin repeats WITHIN that scale. Each of the six CDR domains
#     counts as its OWN scale, so a memory-only decline survives even when the other
#     five boxes sit still -- which is the whole point of dropping the CDRSUM total.
#   * everything else (baseline, diseases, death): KEEP-FIRST = age at first onset.
SCALE_GROUP = {}
for _col, _levels in CDR_BOX_LEVELS.items():
    for _, _idx in _levels:  SCALE_GROUP[_idx] = f"CDR_{_col}"
for _col, _levels in FAQ_LEVELS.items():
    for _, _idx in _levels:  SCALE_GROUP[_idx] = f"FAQ_{_col}"
for _col, _levels in NPI_LEVELS.items():
    for _, _idx in _levels:  SCALE_GROUP[_idx] = f"NPI_{_col}"
for _, _, _idx in GDS_BINS:  SCALE_GROUP[_idx] = "GDS"
for i in range(29, 33):   SCALE_GROUP[i] = "MOCA"
for i in range(106, 110): SCALE_GROUP[i] = "NACCUDSD"

E["grp"] = E["idx"].map(SCALE_GROUP)
is_cog = E["grp"].notna()

non = (E[~is_cog].sort_values(["pid", "age", "idx"])
                 .drop_duplicates(subset=["pid", "idx"], keep="first"))       # keep-first

cog = E[is_cog].sort_values(["pid", "grp", "age"], kind="stable")
cog = cog[cog["idx"].ne(cog.groupby(["pid", "grp"])["idx"].shift())]          # per-scale: first-in-run + every change

E = pd.concat([non, cog.drop(columns="grp")], ignore_index=True)

# OFF-BY-ONE FIX: bin token = spec_index - 1  (get_batch adds +1 back)
E["tok"] = E["idx"] - 1

# stable integer pids by sorted NACCID (one consistent pid space across full + splits)
sorted_pids = sorted(E["pid"].unique())
idmap = {p:i for i,p in enumerate(sorted_pids)}
E["pidi"] = E["pid"].map(idmap).astype(np.uint32)
E = E.sort_values(["pidi","age","tok"])

# FULL dataset = source of truth (re-split this however you like later)
full = E[["pidi","age","tok"]].to_numpy(np.uint32)
full.tofile(os.path.join(OUT, "nacc_all.bin"))
print(f"  nacc_all.bin: {full.shape[0]} events, {E['pid'].nunique()} patients (FULL, unsplit)")
# pid map so the full file is traceable back to real NACCIDs
pd.DataFrame({"pidi": range(len(sorted_pids)), "NACCID": sorted_pids}).to_csv(
    os.path.join(OUT, "nacc_pidmap.csv"), index=False)

# NOTE: the train/val/test splits are produced by make_split_ad.py (by-patient
# 70/10/20 into out_ad/nacc-dedup-s<seed>/), NOT here. This tokenizer only emits the
# full unsplit nacc_all.bin + labels.csv + nacc_pidmap.csv.

# ----------------------------- labels.csv (model space) ----------------------
spec = pd.read_excel(SPEC)   # do NOT filter on name -- some tokens label only via 'description'
spec_lbl = {}
for _, r in spec.iterrows():
    if pd.isna(r["index"]):
        continue
    i = int(r["index"])
    nm = str(r["name"]).strip() if pd.notna(r["name"]) else ""
    de = str(r["description"]).strip() if pd.notna(r["description"]) else ""
    if nm and de:   spec_lbl[i] = f"{nm} ({de})"   # e.g. "I21.9 (Heart attack)"
    elif nm:        spec_lbl[i] = nm                # e.g. "Male", "Smoking low"
    elif de:        spec_lbl[i] = de                # e.g. "Alcohol Abuse" (name cell blank)
MAXIDX = 227   # 110 -> 139 (CDR boxes) -> 227 (FAQ 140-175, NPI-Q 176-223, GDS 224-227)
rows = []
for i in range(MAXIDX+1):
    if i==0: rows.append("Padding")
    elif i==1: rows.append("No event")
    else: rows.append(spec_lbl.get(i, f"Reserved {i}"))
pd.DataFrame({"event_name":rows}).to_csv(os.path.join(OUT,"labels.csv"), index=False)
print(f"  labels.csv: {len(rows)} rows (vocab_size = {MAXIDX+1})")
print("DONE ->", OUT)
