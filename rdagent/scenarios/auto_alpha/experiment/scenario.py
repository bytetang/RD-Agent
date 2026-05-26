"""
AutoAlphaScenario — describes the auto-alpha-playground task to the LLM.

Background and feature-coding contract are loaded from the playground's
own FEATURE_SPEC.md + README.md so the two sides cannot drift.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from rdagent.app.auto_alpha_loop.conf import AUTO_ALPHA_FACTOR_PROP_SETTING
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T


def _read_playground_file(playground_path: str, relpath: str) -> str:
    p = Path(playground_path) / relpath
    if not p.exists():
        return f"(file missing: {p})"
    return p.read_text(encoding="utf-8")


# Mapping from baseline trainer-feature-column names back to the registered
# feature plugin name (i.e., the @register("...") string). Names on the right
# are the ones the LLM must NOT propose as new factor_name values.
#
# Keep this aligned with playground's featurizer/features/ registrations.
_COLUMN_TO_REGISTERED_NAME = {
    "logret_1": "logret",
    "rsi_14": "rsi",
    "macd_12_26_9": "macd",
    "bollinger_20_2.0": "bollinger",
    "volume_z_20": "volume_z",
    # Round-2 baseline additions (top-5 winners promoted after ensemble validation):
    "tsi": "tsi",
    "trix": "trix",
    "atr": "atr",
    "chop": "chop",
    "cmf_adx_prod": "cmf_adx_prod",
}


def _baseline_registered_names_from_eval_config(eval_cfg_path: Path) -> list[str]:
    """Read the frozen eval config and resolve each baseline feature column
    back to its registered plugin name. Returns sorted unique names."""
    if not eval_cfg_path.exists():
        return []
    cfg = yaml.safe_load(eval_cfg_path.read_text())
    columns = list(cfg.get("features", []) or [])
    names: set[str] = set()
    for col in columns:
        # If we know the mapping, use it; else assume the column name itself
        # is the registered name (param-less plugin convention).
        names.add(_COLUMN_TO_REGISTERED_NAME.get(col, col))
    return sorted(names)


class AutoAlphaScenario(Scenario):
    """
    Scenario for autonomous alpha mining on top of auto-alpha-playground.

    The LLM proposes new pandas-based feature plugins; an external subprocess
    (`scripts/eval_feature.py` in the playground) evaluates each candidate
    against a frozen baseline and returns a JSON metric envelope.
    """

    def __init__(self) -> None:
        super().__init__()
        playground = AUTO_ALPHA_FACTOR_PROP_SETTING.playground_path

        # Authoritative coder contract lives in the playground; we paste the
        # spec verbatim so the prompt never lies about what is/isn't allowed.
        self._feature_spec = _read_playground_file(playground, "FEATURE_SPEC.md")
        self._playground_readme = _read_playground_file(playground, "README.md")

        # Resolve baseline feature column list -> registered plugin names.
        # The LLM must avoid proposing these names (it caused round-1 collision
        # in the first flash session: LLM proposed `rsi` against baseline `rsi_14`).
        eval_cfg_path = Path(playground) / AUTO_ALPHA_FACTOR_PROP_SETTING.eval_config_name
        self._baseline_feature_names = _baseline_registered_names_from_eval_config(eval_cfg_path)
        self._baseline_feature_columns = list(
            (yaml.safe_load(eval_cfg_path.read_text()).get("features", []) or [])
            if eval_cfg_path.exists()
            else []
        )

        self._background = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_background").r())
        self._source_data = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_source_data").r())
        self._output_format = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_output_format").r())
        self._interface = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_interface").r(feature_spec=self._feature_spec)
        )
        self._simulator = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_simulator").r(
                eval_config=AUTO_ALPHA_FACTOR_PROP_SETTING.eval_config_name,
                playground_path=playground,
            )
        )
        self._rich_style_description = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_rich_style_description").r()
        )
        self._experiment_setting = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_experiment_setting").r(
                eval_config=AUTO_ALPHA_FACTOR_PROP_SETTING.eval_config_name,
                playground_path=playground,
                baseline_feature_columns=self._baseline_feature_columns,
                baseline_feature_names=self._baseline_feature_names,
            )
        )

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    @property
    def feature_spec(self) -> str:
        """Raw FEATURE_SPEC.md text — used by the coder to seed prompts."""
        return self._feature_spec

    @property
    def baseline_feature_names(self) -> list[str]:
        """Registered plugin names already in baseline; the LLM must not reuse them."""
        return list(self._baseline_feature_names)

    @property
    def baseline_feature_columns(self) -> list[str]:
        """Trainer-config feature column names (e.g. 'rsi_14') already in baseline."""
        return list(self._baseline_feature_columns)

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        if simple_background:
            return f"Background of the scenario:\n{self.background}"
        return (
            f"Background of the scenario:\n{self.background}\n"
            f"The source data you can use:\n{self.get_source_data_desc(task)}\n"
            f"The interface you should follow to write the runnable code:\n{self.interface}\n"
            f"The output of your code should be in the format:\n{self.output_format}\n"
            f"The simulator user can use to test your factor:\n{self.simulator}\n"
        )

    def get_runtime_environment(self) -> str:
        return (
            "Python 3.10+, pandas 2.2+, numpy 1.26+, lightgbm 4.0+. "
            "Generated feature plugins run inside the playground virtualenv "
            "via subprocess; no network, no filesystem access permitted."
        )
