"""
AutoAlphaScenario — describes the auto-alpha-playground task to the LLM.

Background and feature-coding contract are loaded from the playground's
own FEATURE_SPEC.md + README.md so the two sides cannot drift.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

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
# feature plugin name (i.e., the @register("...") string).
_COLUMN_TO_REGISTERED_NAME = {
    "logret_1": "logret",
    "rsi_14": "rsi",
    "macd_12_26_9": "macd",
    "bollinger_20_2.0": "bollinger",
    "volume_z_20": "volume_z",
}


def _baseline_columns_from_bench(playground_path: str) -> list[str]:
    """Source of truth: `BASELINE_FEATURES` in playground's bench/runner.py."""
    p = Path(playground_path) / "bench" / "runner.py"
    if not p.exists():
        return []
    cols: list[str] = []
    in_block = False
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if line.startswith("BASELINE_FEATURES"):
            in_block = True
            continue
        if in_block:
            if line.startswith("]"):
                break
            if line.startswith('"') or line.startswith("'"):
                cols.append(line.strip(",").strip('"').strip("'"))
    return cols


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

        # Resolve baseline feature columns from playground's bench/runner.py.
        self._baseline_feature_columns = _baseline_columns_from_bench(playground)
        self._baseline_feature_names = sorted({
            _COLUMN_TO_REGISTERED_NAME.get(c, c) for c in self._baseline_feature_columns
        })

        self._background = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_background").r())
        self._source_data = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_source_data").r())
        self._output_format = deepcopy(T("scenarios.auto_alpha.prompts:auto_alpha_output_format").r())
        self._interface = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_interface").r(feature_spec=self._feature_spec)
        )
        self._simulator = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_simulator").r(
                bench_cells=AUTO_ALPHA_FACTOR_PROP_SETTING.bench_cells,
                playground_path=playground,
            )
        )
        self._rich_style_description = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_rich_style_description").r()
        )
        self._experiment_setting = deepcopy(
            T("scenarios.auto_alpha.prompts:auto_alpha_experiment_setting").r(
                bench_cells=AUTO_ALPHA_FACTOR_PROP_SETTING.bench_cells,
                playground_path=playground,
                baseline_feature_columns=self._baseline_feature_columns,
                baseline_feature_names=self._baseline_feature_names,
                accept_min_improved_cells=AUTO_ALPHA_FACTOR_PROP_SETTING.accept_min_improved_cells,
                accept_max_drop=AUTO_ALPHA_FACTOR_PROP_SETTING.accept_max_drop,
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
