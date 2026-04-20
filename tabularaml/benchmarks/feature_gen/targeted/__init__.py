"""Targeted benchmark suites for feature engineering evaluation.

Three curated suites complement the 1,000-dataset random sample:
  - amlb:        OpenML-CC18 classification suite (~44 tasks), matches AMLB leaderboard.
  - pmlb:        Penn ML Benchmarks standard ~20 datasets, sanity-check anchor.
  - stress_test: 10 hand-picked datasets targeting specific FE pathologies.

Entry point::

    python -m tabularaml.benchmarks.feature_gen.targeted --suite amlb --frameworks nofe tabularaml openfe
"""
