"""
Settings for the auto_alpha factor R&D loop.

Configurable via env vars prefixed AUTO_ALPHA_ (Pydantic Settings convention),
or via a .env file in the working directory.
"""
from __future__ import annotations

from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class AutoAlphaFactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="AUTO_ALPHA_", protected_namespaces=())

    # 1) Override base class paths to point at the auto_alpha scenario.
    scen: str = "rdagent.scenarios.auto_alpha.experiment.scenario.AutoAlphaScenario"
    """Scenario class for auto_alpha factor mining."""

    hypothesis_gen: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorHypothesisGen"
    )
    """Hypothesis generation class."""

    hypothesis2experiment: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorHypothesis2Experiment"
    )
    """Hypothesis -> Experiment converter class."""

    coder: str = "rdagent.scenarios.auto_alpha.developer.coder.AutoAlphaFactorCoder"
    """Single-shot LLM coder for feature plugins."""

    runner: str = "rdagent.scenarios.auto_alpha.developer.runner.AutoAlphaFactorRunner"
    """Shells out to playground's scripts/eval_feature.py."""

    summarizer: str = (
        "rdagent.scenarios.auto_alpha.proposal.factor_proposal.AutoAlphaFactorExperiment2Feedback"
    )
    """Translates eval JSON envelope into HypothesisFeedback."""

    # 2) auto_alpha-specific settings.

    playground_path: str = "/Users/jie/Documents/src/auto-alpha-playground"
    """Absolute path to the auto-alpha-playground checkout."""

    eval_config_name: str = "configs/train_baseline_eval_1m.yaml"
    """Path (relative to playground_path) of the frozen evaluation config."""

    confirmation_eval_config_name: str = "configs/train_baseline_eval_1s.yaml"
    """Slower confirmation eval config (1s kline); used for top-K survivors."""

    raw_input_dir: Optional[str] = None
    """Override for raw kline directory; None = use eval config default."""

    featurized_output_dir: Optional[str] = None
    """Override for featurized output directory; None = use eval config default."""

    subprocess_timeout_seconds: int = 1800
    """Per-subprocess timeout passed through to eval_feature.py."""

    keep_success: bool = False
    """If True, accepted feature plugins are kept in featurizer/features/."""


AUTO_ALPHA_FACTOR_PROP_SETTING = AutoAlphaFactorBasePropSetting()
