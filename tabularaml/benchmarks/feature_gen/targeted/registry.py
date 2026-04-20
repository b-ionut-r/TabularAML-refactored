"""Fixed dataset registries for targeted benchmark suites.

All lists are hard-coded so the benchmark runs offline after the first
dataset download (OpenML and PMLB both cache locally).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    id: str                  # OpenML task ID (str of int) or PMLB dataset name
    name: str                # Human-readable label
    source: str              # "openml_task" | "pmlb"
    task: str                # "classification" | "regression" | "multiclass"
    suite: str               # "amlb" | "pmlb" | "stress_test" | "ctr23"
    rationale: str = ""      # Why this dataset is in the suite


# ---------------------------------------------------------------------------
# AMLB CC18 — complete OpenML-CC18 classification suite (study id 99, 72 tasks).
# These are the canonical task IDs used by the AutoML Benchmark (Gijsbers et al.
# 2022/2024), TabPFN, and AutoGluon papers. Running the full suite enables
# direct leaderboard comparison. Task types are hints; loader infers from data.
# ---------------------------------------------------------------------------
AMLB_CC18: list[DatasetSpec] = [
    DatasetSpec("3",      "kr-vs-kp",                      "openml_task", "classification", "amlb"),
    DatasetSpec("6",      "letter",                        "openml_task", "multiclass",     "amlb"),
    DatasetSpec("11",     "balance-scale",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("12",     "mfeat-factors",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("14",     "mfeat-fourier",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("15",     "breast-w",                      "openml_task", "classification", "amlb"),
    DatasetSpec("16",     "mfeat-karhunen",                "openml_task", "multiclass",     "amlb"),
    DatasetSpec("18",     "mfeat-morphological",           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("22",     "mfeat-zernike",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("23",     "cmc",                           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("28",     "optdigits",                     "openml_task", "multiclass",     "amlb"),
    DatasetSpec("29",     "credit-approval",               "openml_task", "classification", "amlb"),
    DatasetSpec("31",     "credit-g",                      "openml_task", "classification", "amlb"),
    DatasetSpec("32",     "pendigits",                     "openml_task", "multiclass",     "amlb"),
    DatasetSpec("37",     "diabetes",                      "openml_task", "classification", "amlb"),
    DatasetSpec("43",     "spambase",                      "openml_task", "classification", "amlb"),
    DatasetSpec("45",     "splice",                        "openml_task", "multiclass",     "amlb"),
    DatasetSpec("49",     "tic-tac-toe",                   "openml_task", "classification", "amlb"),
    DatasetSpec("53",     "vehicle",                       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("219",    "electricity",                   "openml_task", "classification", "amlb"),
    DatasetSpec("2074",   "satimage",                      "openml_task", "multiclass",     "amlb"),
    DatasetSpec("2079",   "eucalyptus",                    "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3021",   "sick",                          "openml_task", "classification", "amlb"),
    DatasetSpec("3022",   "vowel",                         "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3481",   "isolet",                        "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3549",   "analcatdata_authorship",        "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3560",   "analcatdata_dmft",              "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3573",   "mnist_784",                     "openml_task", "multiclass",     "amlb"),
    DatasetSpec("3902",   "pc4",                           "openml_task", "classification", "amlb"),
    DatasetSpec("3903",   "pc3",                           "openml_task", "classification", "amlb"),
    DatasetSpec("3904",   "jm1",                           "openml_task", "classification", "amlb"),
    DatasetSpec("3913",   "kc2",                           "openml_task", "classification", "amlb"),
    DatasetSpec("3917",   "kc1",                           "openml_task", "classification", "amlb"),
    DatasetSpec("3918",   "pc1",                           "openml_task", "classification", "amlb"),
    DatasetSpec("7592",   "adult",                         "openml_task", "classification", "amlb"),
    DatasetSpec("9910",   "Bioresponse",                   "openml_task", "classification", "amlb"),
    DatasetSpec("9946",   "wdbc",                          "openml_task", "classification", "amlb"),
    DatasetSpec("9952",   "phoneme",                       "openml_task", "classification", "amlb"),
    DatasetSpec("9957",   "qsar-biodeg",                   "openml_task", "classification", "amlb"),
    DatasetSpec("9960",   "wall-robot-navigation",         "openml_task", "multiclass",     "amlb"),
    DatasetSpec("9964",   "semeion",                       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("9971",   "ilpd",                          "openml_task", "classification", "amlb"),
    DatasetSpec("9976",   "madelon",                       "openml_task", "classification", "amlb"),
    DatasetSpec("9977",   "nomao",                         "openml_task", "classification", "amlb"),
    DatasetSpec("9978",   "ozone-level-8hr",               "openml_task", "classification", "amlb"),
    DatasetSpec("9981",   "cnae-9",                        "openml_task", "multiclass",     "amlb"),
    DatasetSpec("9985",   "first-order-theorem-proving",   "openml_task", "multiclass",     "amlb"),
    DatasetSpec("10093",  "banknote-authentication",       "openml_task", "classification", "amlb"),
    DatasetSpec("10101",  "blood-transfusion",             "openml_task", "classification", "amlb"),
    DatasetSpec("14952",  "PhishingWebsites",              "openml_task", "classification", "amlb"),
    DatasetSpec("14954",  "cylinder-bands",                "openml_task", "classification", "amlb"),
    DatasetSpec("14965",  "bank-marketing",                "openml_task", "classification", "amlb"),
    DatasetSpec("14969",  "GesturePhaseSegmentation",      "openml_task", "multiclass",     "amlb"),
    DatasetSpec("14970",  "har",                           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("125920", "dresses-sales",                 "openml_task", "classification", "amlb"),
    DatasetSpec("125922", "texture",                       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146195", "connect-4",                     "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146800", "MiceProtein",                   "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146817", "steel-plates-fault",            "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146819", "climate-model-crashes",         "openml_task", "classification", "amlb"),
    DatasetSpec("146820", "wilt",                          "openml_task", "classification", "amlb"),
    DatasetSpec("146821", "car",                           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146822", "segment",                       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146824", "mfeat-pixel",                   "openml_task", "multiclass",     "amlb"),
    DatasetSpec("146825", "Fashion-MNIST",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("167119", "jungle_chess",                  "openml_task", "multiclass",     "amlb"),
    DatasetSpec("167120", "numerai28.6",                   "openml_task", "classification", "amlb"),
    DatasetSpec("167121", "Devnagari-Script",              "openml_task", "multiclass",     "amlb"),
    DatasetSpec("167124", "CIFAR_10",                      "openml_task", "multiclass",     "amlb"),
    DatasetSpec("167125", "Internet-Advertisements",       "openml_task", "classification", "amlb"),
    DatasetSpec("167140", "dna",                           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("167141", "churn",                         "openml_task", "classification", "amlb"),
]


# ---------------------------------------------------------------------------
# PMLB Standard — Penn ML Benchmarks, top ~20 most-cited datasets.
# Loaded via `pmlb.fetch_data(name)` — no OpenML dependency.
# Anchors results in familiar territory; task type inferred at load time.
# ---------------------------------------------------------------------------
PMLB_STANDARD: list[DatasetSpec] = [
    DatasetSpec("adult",          "adult",          "pmlb", "classification", "pmlb"),
    DatasetSpec("mushroom",       "mushroom",       "pmlb", "classification", "pmlb"),
    DatasetSpec("spambase",       "spambase",       "pmlb", "classification", "pmlb"),
    DatasetSpec("kr_vs_kp",       "kr-vs-kp",       "pmlb", "classification", "pmlb"),
    DatasetSpec("chess",          "chess",           "pmlb", "classification", "pmlb"),
    DatasetSpec("ionosphere",     "ionosphere",      "pmlb", "classification", "pmlb"),
    DatasetSpec("sonar",          "sonar",           "pmlb", "classification", "pmlb"),
    DatasetSpec("breast_cancer",  "breast-cancer",   "pmlb", "classification", "pmlb"),
    DatasetSpec("titanic",        "titanic",         "pmlb", "classification", "pmlb"),
    DatasetSpec("nursery",        "nursery",         "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("waveform_21",    "waveform-21",     "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("car_evaluation", "car-evaluation",  "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("letter",         "letter",          "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("satimage",       "satimage",        "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("shuttle",        "shuttle",         "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("pendigits",      "pendigits",       "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("optdigits",      "optdigits",       "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("vehicle",        "vehicle",         "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("mfeat_factors",  "mfeat-factors",   "pmlb", "multiclass",     "pmlb"),
    DatasetSpec("monk1",          "monk1",           "pmlb", "classification", "pmlb"),
    # --- Regression ---
    DatasetSpec("529_pollen",        "pollen",       "pmlb", "regression", "pmlb"),
    DatasetSpec("503_wind",          "wind",         "pmlb", "regression", "pmlb"),
    DatasetSpec("1193_BNG_lowbwt",   "BNG-lowbwt",  "pmlb", "regression", "pmlb"),
    DatasetSpec("581_fri_c3_500_25", "Friedman-c3", "pmlb", "regression", "pmlb"),
    DatasetSpec("1028_SWD",          "SWD",          "pmlb", "regression", "pmlb"),
]


# ---------------------------------------------------------------------------
# Stress-Test — 10 datasets targeting specific FE pathologies.
# Mix of OpenML task IDs and PMLB names. Designed to expose failure modes
# that the random sample under-represents.
# ---------------------------------------------------------------------------
STRESS_TEST: list[DatasetSpec] = [
    DatasetSpec(
        "1169", "Airlines", "openml_task", "classification", "stress_test",
        rationale="100k+ rows, high-cardinality categoricals: tests scale & encoding robustness",
    ),
    DatasetSpec(
        "42712", "Bike-Sharing", "openml_task", "regression", "stress_test",
        rationale="Temporal structure: tests whether tools extract day_of_week / lag features",
    ),
    DatasetSpec(
        "1113", "KDD-Cup-98", "openml_task", "regression", "stress_test",
        rationale="Multi-table relational structure: primary test of Featuretools DFS",
    ),
    DatasetSpec(
        "23380", "Mercedes-Greener", "openml_task", "regression", "stress_test",
        rationale="Anonymous feature names (X1, X2, ...): tests tools that rely on column names",
    ),
    DatasetSpec(
        "41207", "Allstate-Claims", "openml_task", "regression", "stress_test",
        rationale="Highly skewed regression target: tests overfitting risk with FE",
    ),
    DatasetSpec(
        "40927", "Sleep", "openml_task", "classification", "stress_test",
        rationale="Wide table (small n, many features): tests regularisation under high-dim FE",
    ),
    DatasetSpec(
        "yeast", "yeast", "pmlb", "multiclass", "stress_test",
        rationale="Multiclass imbalance: tests whether FE degrades minority-class prediction",
    ),
    DatasetSpec(
        "monk2", "monk2", "pmlb", "classification", "stress_test",
        rationale="Non-linear parity concept: signal only emerges from feature interactions",
    ),
    DatasetSpec(
        "connect_4", "connect-4", "pmlb", "classification", "stress_test",
        rationale="High-arity nominal categoricals: stresses categorical encoding pipelines",
    ),
    DatasetSpec(
        "adult", "adult-pmlb", "pmlb", "classification", "stress_test",
        rationale="'?' missing value sentinel: tests missing-value handling robustness",
    ),
]


# ---------------------------------------------------------------------------
# OpenML-CTR23 Regression — complete set of 35 tasks from the Curated Tabular
# Regression benchmark (OpenML study 353, Fischer et al. 2023).
# Enables direct leaderboard comparison for regression FE evaluation.
# ---------------------------------------------------------------------------
CTR23_REGRESSION: list[DatasetSpec] = [
    # --- Tiny / Small (<2k rows) ---
    DatasetSpec("361618", "forest_fires",          "openml_task", "regression", "ctr23",
                rationale="UCI forest fires; tiny (517×12); log-skewed target, notoriously hard"),
    DatasetSpec("361619", "student_performance",   "openml_task", "regression", "ctr23",
                rationale="Portuguese student grades; tiny (649×30); high-dim for sample size"),
    DatasetSpec("361617", "energy_efficiency",     "openml_task", "regression", "ctr23",
                rationale="Building physics; small (768×8); UCI Tsanas & Xifara"),
    DatasetSpec("361621", "QSAR_fish_toxicity",    "openml_task", "regression", "ctr23",
                rationale="Chemical toxicity; small (908×6); molecular descriptors"),
    DatasetSpec("361237", "concrete_strength",     "openml_task", "regression", "ctr23",
                rationale="Concrete compressive strength; small (1k×8); UCI classic"),
    DatasetSpec("361616", "Moneyball",             "openml_task", "regression", "ctr23",
                rationale="Baseball salary prediction; small (1.2k×14); sports/economic domain"),
    DatasetSpec("361269", "health_insurance",      "openml_task", "regression", "ctr23",
                rationale="US health insurance charges; small (1.3k×6); known interaction structure"),
    DatasetSpec("361235", "airfoil_self_noise",    "openml_task", "regression", "ctr23",
                rationale="NASA aeroacoustics; small (1.5k×5); all-numeric"),
    DatasetSpec("361250", "red_wine",              "openml_task", "regression", "ctr23",
                rationale="Red wine quality; small (1.6k×11); ordered integer target"),
    DatasetSpec("361622", "cars",                  "openml_task", "regression", "ctr23",
                rationale="Used car prices; small (1.7k×5); mixed categorical + numeric"),
    DatasetSpec("361236", "auction_verification",  "openml_task", "regression", "ctr23",
                rationale="Online auction verification; small (2.4k×9); mixed feature types"),
    DatasetSpec("361623", "space_ga",              "openml_task", "regression", "ctr23",
                rationale="Spatial statistics; small (3k×6); Friedman-style benchmark variant"),
    # --- Medium (4k–15k rows) ---
    DatasetSpec("361234", "abalone",               "openml_task", "regression", "ctr23",
                rationale="Biological age prediction; medium (4k×8); widely cited classic"),
    DatasetSpec("361249", "white_wine",            "openml_task", "regression", "ctr23",
                rationale="Wine quality scoring; medium (5k×11); ordered integer target"),
    DatasetSpec("361243", "music_origin",          "openml_task", "regression", "ctr23",
                rationale="Geographical origin of music; medium (1k×70); high-dim/small-n"),
    DatasetSpec("361258", "kin8nm",                "openml_task", "regression", "ctr23",
                rationale="Robot arm kinematics (Delve); medium (8k×9); all-numeric"),
    DatasetSpec("361259", "pumadyn32nh",           "openml_task", "regression", "ctr23",
                rationale="Robot arm dynamics (Delve); medium (8k×33); higher-dim variant"),
    DatasetSpec("361256", "cpu_activity",          "openml_task", "regression", "ctr23",
                rationale="CPU utilisation (Delve); medium (8k×22); collinear feature group"),
    DatasetSpec("361264", "socmob",                "openml_task", "regression", "ctr23",
                rationale="Social mobility; small (1k×5); social science domain"),
    DatasetSpec("361267", "brazilian_houses",      "openml_task", "regression", "ctr23",
                rationale="Brazilian rental prices; medium (10k×7); real estate with NaN"),
    DatasetSpec("361247", "naval_propulsion",      "openml_task", "regression", "ctr23",
                rationale="CCPP naval plant; medium (11k×16); near-perfect predictability"),
    DatasetSpec("361260", "miami_housing",         "openml_task", "regression", "ctr23",
                rationale="Miami house prices; medium (13k×14); geospatial features"),
    DatasetSpec("361268", "fps_benchmark",         "openml_task", "regression", "ctr23",
                rationale="CPU/GPU FPS benchmark; medium (19k×15); hardware performance"),
    DatasetSpec("361244", "solar_flare",           "openml_task", "regression", "ctr23",
                rationale="Solar activity counts; small (1.1k×12); count-valued target"),
    DatasetSpec("361254", "sarcos",                "openml_task", "regression", "ctr23",
                rationale="SARCOS robot arm; large (48k×21); robot dynamics"),
    # --- Large (20k+ rows) ---
    DatasetSpec("361255", "california_housing",    "openml_task", "regression", "ctr23",
                rationale="Census housing prices; large (20k×8); used in OpenFE paper"),
    DatasetSpec("361272", "fifa",                  "openml_task", "regression", "ctr23",
                rationale="FIFA player ratings; large (18k×29); mixed numeric + categorical"),
    DatasetSpec("361266", "kings_county",          "openml_task", "regression", "ctr23",
                rationale="King County house sales; large (21k×20); geo features + date"),
    DatasetSpec("361242", "superconductivity",     "openml_task", "regression", "ctr23",
                rationale="Critical temperature; large (21k×81); high-dimensional numeric"),
    DatasetSpec("361261", "cps88wages",            "openml_task", "regression", "ctr23",
                rationale="US wages survey; large (28k×8); social-science, skewed target"),
    DatasetSpec("361241", "physiochemical_protein","openml_task", "regression", "ctr23",
                rationale="Protein structure (CASP); large (45k×9); widely used in DL papers"),
    DatasetSpec("361257", "diamonds",              "openml_task", "regression", "ctr23",
                rationale="Diamond pricing; large (54k×10); numeric + 3 ordinal categoricals"),
    DatasetSpec("361251", "grid_stability",        "openml_task", "regression", "ctr23",
                rationale="Smart grid stability; large (60k×12); synthetic, physically motivated"),
    DatasetSpec("361253", "wave_energy",           "openml_task", "regression", "ctr23",
                rationale="Wave energy converters; very large (72k×49); high-dim simulation"),
    DatasetSpec("361252", "video_transcoding",     "openml_task", "regression", "ctr23",
                rationale="Video transcoding time; very large (68k×19); IT systems performance"),
]


# ---------------------------------------------------------------------------
# Smoke Test — Tiny datasets for fast cloud pipeline verification.
# Contains a mix of all task types (classification, regression, multiclass)
# and sources (openml_task, pmlb) to verify API integrations.
# ---------------------------------------------------------------------------
SMOKE_TEST: list[DatasetSpec] = [
    DatasetSpec("breast_cancer", "breast-cancer", "pmlb", "classification", "smoke_test", 
                rationale="Tests PMLB integration (classification)"),
    DatasetSpec("361618", "forest_fires", "openml_task", "regression", "smoke_test", 
                rationale="Tests OpenML CTR23 integration (regression)"),
    DatasetSpec("23", "cmc", "openml_task", "multiclass", "smoke_test", 
                rationale="Tests OpenML AMLB integration (multiclass)"),
]


SUITE_MAP: dict[str, list[DatasetSpec]] = {
    "amlb":        AMLB_CC18,
    "pmlb":        PMLB_STANDARD,
    "stress_test": STRESS_TEST,
    "ctr23":       CTR23_REGRESSION,
    "smoke_test":  SMOKE_TEST,
}


def get_suite(name: str) -> list[DatasetSpec]:
    if name == "all":
        return AMLB_CC18 + PMLB_STANDARD + STRESS_TEST + CTR23_REGRESSION
    if name not in SUITE_MAP:
        raise ValueError(f"Unknown suite {name!r}. Choose from: {sorted(SUITE_MAP)} or 'all'.")
    return SUITE_MAP[name]
