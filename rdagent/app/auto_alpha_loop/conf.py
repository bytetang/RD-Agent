"""
Settings for the auto_alpha factor R&D loop.

Configurable via env vars prefixed AUTO_ALPHA_ (Pydantic Settings convention),
or via a .env file in the working directory.
"""
from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class AutoAlphaFactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="AUTO_ALPHA_", protected_namespaces=())

    # 1) Override base class paths to point at the auto_alpha scenario.
    scen: str = "rdagent.scenarios.auto_alpha.experiment.scenario.AutoAlphaScenario"
    hypothesis_gen: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorHypothesisGen"
    )
    hypothesis2experiment: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorHypothesis2Experiment"
    )
    coder: str = "rdagent.scenarios.auto_alpha.developer.coder.AutoAlphaFactorCoder"
    runner: str = "rdagent.scenarios.auto_alpha.developer.runner.AutoAlphaFactorRunner"
    summarizer: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorExperiment2Feedback"
    )

    # 2) auto_alpha-specific settings — new bench-based flow.

    playground_path: str = "/Users/jie/Documents/src/auto-alpha-playground"
    """Absolute path to the auto-alpha-playground checkout."""

    bench_cells: str = "fast"
    """`--cells` preset passed to `python -m bench run`. 'fast' = drop slow
    1s lgbm cells (12 cells remain); 'all' = full 14-cell sweep; or an
    fnmatch glob like 'lgbm_1m_*'."""

    accept_min_improved_cells: int = 6
    """Minimum number of cells with delta_headline > 0 to accept the candidate."""

    accept_max_drop: float = 0.005
    """Maximum allowed drop in headline metric on the worst cell. A candidate
    is rejected if any cell has delta_headline < -accept_max_drop. Headline
    metrics are val_acc (classification) and val_ic_pearson (regression)."""

    subprocess_timeout_seconds: int = 1800
    """Per-bench-subprocess timeout. Fast sweep should finish in ~3 min on
    M-series; we leave generous headroom for I/O hiccups."""

    keep_success: bool = False
    """If True, accepted feature plugins are KEPT in featurizer/features/
    after the round. CAUTION: this shifts the baseline ground truth for
    subsequent rounds. Default False keeps each round independent."""


AUTO_ALPHA_FACTOR_PROP_SETTING = AutoAlphaFactorBasePropSetting()
