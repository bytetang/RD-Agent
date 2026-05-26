"""
AutoAlphaFactorCoder — generates one pandas feature plugin per task.

Design choices vs the qlib FactorCoSTEER pattern:
- Single-shot LLM call (no evolutionary code-evolution sub-loop). The
  playground's `scripts/lint_feature.py` gives a fast static signal, and
  the outer RD loop is the natural retry mechanism.
- Validation is delegated to the playground's lint script via subprocess so
  the lint contract has a single source of truth across both repos.
- On lint failure we still attach the generated code (for the next-round
  hypothesis to learn from) but raise CoderError so the loop records the
  failure as feedback rather than running a broken plugin through eval.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rdagent.app.auto_alpha_loop.conf import AUTO_ALPHA_FACTOR_PROP_SETTING
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.auto_alpha.experiment.factor_experiment import (
    AutoAlphaFactorExperiment,
    AutoAlphaFactorTask,
    AutoAlphaFBWorkspace,
)
from rdagent.scenarios.auto_alpha.experiment.scenario import AutoAlphaScenario
from rdagent.utils.agent.tpl import T


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Pull the first python code block out, or return the whole string."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _lint_via_playground(playground_path: Path, code: str) -> tuple[bool, list[dict[str, Any]]]:
    """Run the playground's lint_feature.py as a subprocess.
    Returns (ok, findings)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        tmp_path = tf.name
    try:
        r = subprocess.run(
            [sys.executable, "scripts/lint_feature.py", tmp_path],
            cwd=str(playground_path),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        # lint_feature.py emits JSON to stdout and exits 0/1 based on findings.
        try:
            payload = json.loads(r.stdout)
        except json.JSONDecodeError:
            return False, [
                {
                    "rule": "lint_invocation_error",
                    "message": f"lint stdout not parseable; stderr tail: {r.stderr[-500:]}",
                    "lineno": 0,
                    "col_offset": 0,
                }
            ]
        return bool(payload.get("ok")), list(payload.get("findings") or [])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class AutoAlphaFactorCoder(Developer[AutoAlphaFactorExperiment]):
    """Single-shot LLM coder for auto-alpha feature plugins."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        if not isinstance(scen, AutoAlphaScenario):
            raise TypeError(
                f"AutoAlphaFactorCoder requires AutoAlphaScenario, got {type(scen).__name__}"
            )
        self.auto_alpha_scen: AutoAlphaScenario = scen
        self.playground_path: Path = Path(
            AUTO_ALPHA_FACTOR_PROP_SETTING.playground_path
        ).expanduser().resolve()

    def develop(self, exp: AutoAlphaFactorExperiment) -> AutoAlphaFactorExperiment:
        if not exp.sub_tasks:
            raise CoderError("AutoAlphaFactorExperiment has no sub_tasks")

        # Single-feature contract: one task per experiment.
        if len(exp.sub_tasks) != 1:
            raise CoderError(
                f"AutoAlphaFactorCoder expects exactly 1 sub_task per experiment "
                f"(got {len(exp.sub_tasks)})"
            )
        task: AutoAlphaFactorTask = exp.sub_tasks[0]
        logger.info(f"coding feature: {task.factor_name}")

        # Ensure the workspace exists.
        if not exp.sub_workspace_list:
            exp.sub_workspace_list = [None]
        if exp.sub_workspace_list[0] is None:
            exp.sub_workspace_list[0] = AutoAlphaFBWorkspace(target_task=task)
        workspace: AutoAlphaFBWorkspace = exp.sub_workspace_list[0]

        # Build prompt and call LLM.
        system_prompt = T(
            "scenarios.auto_alpha.prompts:auto_alpha_coder.system_prompt"
        ).r(
            scenario=self.auto_alpha_scen.get_scenario_all_desc(),
            feature_spec=self.auto_alpha_scen.feature_spec,
        )
        user_prompt = T(
            "scenarios.auto_alpha.prompts:auto_alpha_coder.user_prompt"
        ).r(
            factor_name=task.factor_name,
            factor_description=task.factor_description,
            factor_formulation=task.factor_formulation,
            variables=json.dumps(task.variables, indent=2),
            hypothesis=str(exp.hypothesis) if exp.hypothesis is not None else "(none)",
        )

        raw_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )
        code = _strip_code_fence(raw_response)

        # Stage source code in the workspace under the registered feature name.
        filename = f"{task.factor_name}.py"
        workspace.inject_files(**{filename: code})
        logger.info(f"generated {len(code)} chars of code for {task.factor_name}")

        # Validate via playground lint.
        ok, findings = _lint_via_playground(self.playground_path, code)
        workspace.lint_findings = findings  # type: ignore[attr-defined]
        if not ok:
            messages = "; ".join(f"{f['rule']}: {f['message']}" for f in findings)
            raise CoderError(
                f"playground lint rejected generated code for {task.factor_name}: {messages}"
            )

        return exp
