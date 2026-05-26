"""
AutoAlphaFactorExperiment — a single-feature R&D experiment.

Each experiment holds exactly one AutoAlphaFactorTask (one feature proposal)
and one AutoAlphaFBWorkspace whose `file_dict` carries the generated .py.
The Runner is responsible for shelling out to the playground's
`scripts/eval_feature.py` and stashing the JSON envelope on `exp.result`.
"""
from __future__ import annotations

from typing import Any

from rdagent.core.experiment import Experiment, FBWorkspace, Task


class AutoAlphaFactorTask(Task):
    """One proposed feature.

    Attributes:
        factor_name: lowercase snake_case; matches @register('name') in the file.
        factor_description: human-readable summary of what the feature computes.
        factor_formulation: math/formula description (used to seed the coder prompt).
        variables: free-form dict of param defaults (e.g. {"window": 14}).
    """

    def __init__(
        self,
        factor_name: str,
        factor_description: str = "",
        factor_formulation: str = "",
        variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=factor_name, **kwargs)
        self.factor_name: str = factor_name
        self.factor_description: str = factor_description
        self.factor_formulation: str = factor_formulation
        self.variables: dict[str, Any] = variables or {}

    def get_task_information(self) -> str:
        return (
            f"factor_name: {self.factor_name}\n"
            f"description: {self.factor_description}\n"
            f"formulation: {self.factor_formulation}\n"
            f"variables: {self.variables}"
        )


class AutoAlphaFBWorkspace(FBWorkspace):
    """File-backed workspace for one feature candidate.

    The coder writes the generated source code into `file_dict["<name>.py"]`.
    The Runner reads it back, writes it to a tempfile, and passes that path
    to `scripts/eval_feature.py`. We deliberately do NOT auto-promote into
    the playground's `featurizer/features/` folder — that's the eval script's job.
    """

    def execute(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        # Execution is delegated to the Runner (which knows about the playground
        # checkout and the eval CLI). This is a pure-storage workspace.
        return None


class AutoAlphaFactorExperiment(Experiment[AutoAlphaFactorTask, AutoAlphaFBWorkspace, AutoAlphaFBWorkspace]):
    """One round of `propose -> code -> eval`. Single sub_task by design."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # `result` holds the JSON envelope dict returned by eval_feature.py
        # (status / metrics / baseline_metrics / delta / run_dir / ...).
        self.result: dict[str, Any] | None = None
