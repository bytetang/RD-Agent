"""
AutoAlphaFactorRunner — shells out to the playground's eval_feature.py.

The runner is intentionally thin: it writes the generated feature file to a
tempfile, invokes the playground's atomic eval CLI, parses the returned JSON
envelope, and pins it onto `exp.result`. Caching is delegated to CachedRunner
which keys by md5(task_info).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rdagent.app.auto_alpha_loop.conf import AUTO_ALPHA_FACTOR_PROP_SETTING
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.auto_alpha.experiment.factor_experiment import (
    AutoAlphaFactorExperiment,
    AutoAlphaFactorTask,
    AutoAlphaFBWorkspace,
)


def _minimal_env() -> dict[str, str]:
    """Subprocess env passed to eval_feature.py: keep PATH/locale/home only.
    Never forward LLM keys or other secrets to playground subprocesses."""
    import os
    keep = {}
    for k in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME", "TMPDIR"):
        if k in os.environ:
            keep[k] = os.environ[k]
    return keep


class AutoAlphaFactorRunner(CachedRunner[AutoAlphaFactorExperiment]):
    """Calls `scripts/eval_feature.py --feature-file ... --eval-config ...` and
    parses the JSON envelope.

    On any non-`ok` status the runner raises CoderError so the outer loop
    records a failure with the parsed stderr_tail / findings as feedback.
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self.settings = AUTO_ALPHA_FACTOR_PROP_SETTING
        self.playground_path = Path(self.settings.playground_path).expanduser().resolve()

    def get_cache_key(self, exp: Experiment) -> str:
        # Default CachedRunner hashes by task info — append eval_config so a
        # config change invalidates cached results automatically.
        base = super().get_cache_key(exp)
        return f"{base}__{self.settings.eval_config_name}"

    def develop(self, exp: AutoAlphaFactorExperiment) -> AutoAlphaFactorExperiment:
        if not exp.sub_workspace_list or exp.sub_workspace_list[0] is None:
            raise CoderError("AutoAlphaFactorExperiment has no coded sub_workspace")

        workspace: AutoAlphaFBWorkspace = exp.sub_workspace_list[0]
        task: AutoAlphaFactorTask = exp.sub_tasks[0]
        filename = f"{task.factor_name}.py"
        if filename not in workspace.file_dict:
            raise CoderError(
                f"AutoAlphaFactorRunner expected workspace to carry {filename!r}; "
                f"got {list(workspace.file_dict)}"
            )
        code = workspace.file_dict[filename]

        with tempfile.NamedTemporaryFile(
            "w", suffix=f"__{task.factor_name}.py", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(code)
            feature_path = tf.name

        try:
            cmd = [
                sys.executable,
                "scripts/eval_feature.py",
                "--feature-file", feature_path,
                "--eval-config", self.settings.eval_config_name,
                "--timeout", str(self.settings.subprocess_timeout_seconds),
            ]
            if self.settings.raw_input_dir:
                cmd += ["--raw-input-dir", self.settings.raw_input_dir]
            if self.settings.featurized_output_dir:
                cmd += ["--featurized-output-dir", self.settings.featurized_output_dir]
            if self.settings.keep_success:
                cmd.append("--keep-success")

            logger.info(f"eval_feature: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                cwd=str(self.playground_path),
                capture_output=True,
                text=True,
                timeout=self.settings.subprocess_timeout_seconds + 60,
                env=_minimal_env(),
                check=False,
            )
        finally:
            Path(feature_path).unlink(missing_ok=True)

        if proc.returncode != 0 and not proc.stdout.strip():
            raise CoderError(
                f"eval_feature.py exited {proc.returncode} with no JSON; "
                f"stderr tail: {proc.stderr[-1000:]}"
            )

        try:
            envelope: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise CoderError(
                f"eval_feature.py stdout was not JSON: {exc}; "
                f"stdout tail: {proc.stdout[-1000:]}"
            ) from exc

        exp.result = envelope
        status = envelope.get("status")
        logger.info(f"eval_feature status: {status}")

        if status != "ok":
            # Bubble up the failure as a CoderError so RDLoop records a feedback
            # node with the structured failure info preserved in exp.result.
            reason = envelope.get("stderr_tail") or envelope.get("findings") or "unknown"
            raise CoderError(
                f"eval_feature returned status={status} for {task.factor_name}: {reason}"
            )

        return exp
