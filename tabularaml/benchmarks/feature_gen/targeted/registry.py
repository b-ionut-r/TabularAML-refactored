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
    suite: str               # "amlb" | "pmlb" | "stress_test"
    rationale: str = ""      # Why this dataset is in the suite


# ---------------------------------------------------------------------------
# AMLB CC18 — OpenML-CC18 classification suite (study id 99)
# Task IDs are the canonical list from the AutoML Benchmark (2022/2024)
# and the TabPFN paper. Running on these enables direct leaderboard comparison.
# ---------------------------------------------------------------------------
AMLB_CC18: list[DatasetSpec] = [
    DatasetSpec("3",     "kr-vs-kp",           "openml_task", "classification", "amlb"),
    DatasetSpec("11",    "balance-scale",       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("12",    "mfeat-factors",       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("14",    "mfeat-fourier",       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("15",    "breast-w",            "openml_task", "classification", "amlb"),
    DatasetSpec("16",    "mfeat-karhunen",      "openml_task", "multiclass",     "amlb"),
    DatasetSpec("22",    "mfeat-zernike",       "openml_task", "multiclass",     "amlb"),
    DatasetSpec("23",    "cmc",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("29",    "credit-approval",     "openml_task", "classification", "amlb"),
    DatasetSpec("31",    "credit-g",            "openml_task", "classification", "amlb"),
    DatasetSpec("37",    "diabetes",            "openml_task", "classification", "amlb"),
    DatasetSpec("38",    "sick",                "openml_task", "classification", "amlb"),
    DatasetSpec("44",    "spambase",            "openml_task", "classification", "amlb"),
    DatasetSpec("46",    "splice",              "openml_task", "classification", "amlb"),
    DatasetSpec("50",    "tic-tac-toe",         "openml_task", "classification", "amlb"),
    DatasetSpec("151",   "electricity",         "openml_task", "classification", "amlb"),
    DatasetSpec("182",   "satimage",            "openml_task", "multiclass",     "amlb"),
    DatasetSpec("188",   "eucalyptus",          "openml_task", "multiclass",     "amlb"),
    DatasetSpec("300",   "isolet",              "openml_task", "multiclass",     "amlb"),
    DatasetSpec("307",   "volcanoes-a1",        "openml_task", "classification", "amlb"),
    DatasetSpec("458",   "analcatdata_authorship", "openml_task", "multiclass",  "amlb"),
    DatasetSpec("469",   "analcatdata_dmft",    "openml_task", "multiclass",     "amlb"),
    DatasetSpec("554",   "mnist_784",           "openml_task", "multiclass",     "amlb"),
    DatasetSpec("1049",  "pc4",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1050",  "pc3",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1053",  "jm1",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1063",  "kc2",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1067",  "kc1",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1068",  "pc1",                 "openml_task", "classification", "amlb"),
    DatasetSpec("1590",  "adult",               "openml_task", "classification", "amlb"),
    DatasetSpec("4134",  "bioresponse",         "openml_task", "classification", "amlb"),
    DatasetSpec("4534",  "PhishingWebsites",    "openml_task", "classification", "amlb"),
    DatasetSpec("23517", "numerai28.6",         "openml_task", "classification", "amlb"),
    DatasetSpec("40966", "MiceProtein",         "openml_task", "multiclass",     "amlb"),
    DatasetSpec("40975", "car",                 "openml_task", "multiclass",     "amlb"),
    DatasetSpec("40978", "higgs",               "openml_task", "classification", "amlb"),
    DatasetSpec("40979", "MagicTelescope",      "openml_task", "classification", "amlb"),
    DatasetSpec("40981", "Australian",          "openml_task", "classification", "amlb"),
    DatasetSpec("40982", "steel-plates-fault",  "openml_task", "multiclass",     "amlb"),
    DatasetSpec("40983", "wilt",                "openml_task", "classification", "amlb"),
    DatasetSpec("40984", "segment",             "openml_task", "multiclass",     "amlb"),
    DatasetSpec("40993", "Ionosphere",          "openml_task", "classification", "amlb"),
    DatasetSpec("40994", "climate-model-simulation-crashes", "openml_task", "classification", "amlb"),
    DatasetSpec("40996", "Fashion-MNIST",       "openml_task", "multiclass",     "amlb"),
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
    DatasetSpec("529_pollen",       "pollen",        "pmlb", "regression", "pmlb"),
    DatasetSpec("503_wind",         "wind",          "pmlb", "regression", "pmlb"),
    DatasetSpec("1193_BNG_lowbwt",  "BNG-lowbwt",   "pmlb", "regression", "pmlb"),
    DatasetSpec("581_fri_c3_500_25","Friedman-c3",  "pmlb", "regression", "pmlb"),
    DatasetSpec("1028_SWD",         "SWD",           "pmlb", "regression", "pmlb"),
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


SUITE_MAP: dict[str, list[DatasetSpec]] = {
    "amlb":        AMLB_CC18,
    "pmlb":        PMLB_STANDARD,
    "stress_test": STRESS_TEST,
}


def get_suite(name: str) -> list[DatasetSpec]:
    if name == "all":
        return AMLB_CC18 + PMLB_STANDARD + STRESS_TEST
    if name not in SUITE_MAP:
        raise ValueError(f"Unknown suite {name!r}. Choose from: {sorted(SUITE_MAP)} or 'all'.")
    return SUITE_MAP[name]
