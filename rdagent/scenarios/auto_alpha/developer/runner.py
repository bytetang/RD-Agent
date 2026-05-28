"""
AutoAlphaFactorRunner — shells out to the playground's `python -m bench run`.

Pipeline per round:
  1. Stage the coded feature plugin into <playground>/featurizer/features/<name>.py
  2. Invoke `python -m bench run --features <feature_col> --cells <preset>`
     which featurizes (cached) + sweeps all configured cells + writes
     `runs/bench/<tag>/_compare.json` digesting per-cell deltas vs the
     committed `benchmarks/baseline.csv`.
  3. Read the digest, pin it onto `exp.result`.
  4. Unstage the plugin file (unless `keep_success=True` and the decision is
     accept — handled by the feedback step that knows the decision).

The runner does NOT decide accept/reject; it only produces the digest. The
feedback class consumes the digest and applies the threshold rule.

On any infrastructure failure (lint stage already done by coder; here it's
featurize crash / sweep crash / missing digest) the runner raises CoderError
so the outer loop records the failure as feedback.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    """Subprocess env: keep PATH/locale/home and pass through AAP_* so the
    playground can find data dirs. Never forward LLM keys."""
    keep: dict[str, str] = {}
    for k in ("PATH", "PYTHONPATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME", "TMPDIR", "USER"):
        if k in os.environ:
            keep[k] = os.environ[k]
    for k, v in os.environ.items():
        if k.startswith("AAP_"):
            keep[k] = v
    return keep


class AutoAlphaFactorRunner(CachedRunner[AutoAlphaFactorExperiment]):
    """Calls `python -m bench run --features <col> --cells <preset>` and
    parses `runs/bench/<tag>/_compare.json`."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self.settings = AUTO_ALPHA_FACTOR_PROP_SETTING
        self.playground_path = Path(self.settings.playground_path).expanduser().resolve()
        self.features_dir = self.playground_path / "featurizer" / "features"

    def get_cache_key(self, exp: Experiment) -> str:
        base = super().get_cache_key(exp)
        return f"{base}__bench_{self.settings.bench_cells}"

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

        # bench --features takes the *column* name (e.g. `atr_14`), not the
        # registered name. Column = registered_name + "_" + param_values when
        # the plugin has params, else just registered_name. This must match
        # featurizer/features/__init__.py::_default_column.
        params = task.variables or {}
        if params:
            feature_col = "_".join(
                [task.factor_name] + [str(v) for v in params.values()]
            )
        else:
            feature_col = task.factor_name

        staged_path = self.features_dir / filename
        if staged_path.exists():
            raise CoderError(
                f"staging collision: {staged_path} already exists "
                f"(prior round leak or duplicate proposal)"
            )

        staged_path.write_text(code, encoding="utf-8")
        logger.info(f"staged {staged_path}")

        try:
            cmd = [
                sys.executable, "-m", "bench", "run",
                "--features", feature_col,
                "--cells", self.settings.bench_cells,
            ]
            logger.info(f"bench run: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                cwd=str(self.playground_path),
                capture_output=True,
                text=True,
                timeout=self.settings.subprocess_timeout_seconds,
                env=_minimal_env(),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr

            # Locate the digest path. bench prints `[bench.compare] tag=<tag>
            # compare=<csv> digest=<json> (verbose=...)`.
            digest_path: Path | None = None
            for line in stdout.splitlines():
                if "[bench.compare]" in line and "digest=" in line:
                    for tok in line.split():
                        if tok.startswith("digest="):
                            digest_path = Path(tok.split("=", 1)[1])
                            break
            # Fallback: also accept the sweep-only log line.
            if digest_path is None:
                for line in stdout.splitlines():
                    if "[bench.sweep]" in line and "tag=" in line:
                        # tag=<utc>__<slug>
                        for tok in line.split():
                            if tok.startswith("tag="):
                                tag = tok.split("=", 1)[1]
                                digest_path = (
                                    self.playground_path / "runs" / "bench" / tag / "_compare.json"
                                )
                                break

            if not digest_path or not digest_path.exists():
                tail_out = stdout[-1500:]
                tail_err = stderr[-1500:]
                raise CoderError(
                    f"bench run exit={proc.returncode}; digest not found "
                    f"(parsed path={digest_path}). stdout_tail={tail_out!r} "
                    f"stderr_tail={tail_err!r}"
                )

            try:
                digest: dict[str, Any] = json.loads(digest_path.read_text())
            except json.JSONDecodeError as exc:
                raise CoderError(f"could not parse {digest_path}: {exc}") from exc

            # Annotate digest with bench exit code so feedback can see partial failures.
            digest["_bench_returncode"] = proc.returncode
            digest["_bench_stderr_tail"] = stderr[-1000:] if proc.returncode != 0 else ""
            digest["_feature_col"] = feature_col

            exp.result = digest
            logger.info(
                f"bench digest: {len(digest.get('cells', []))} cells, "
                f"compared_on={digest.get('compared_on')}, "
                f"bench_rc={proc.returncode}"
            )

            if proc.returncode != 0:
                # Some cells failed — surface as CoderError; feedback won't run.
                raise CoderError(
                    f"bench run exit={proc.returncode}; "
                    f"stderr_tail={stderr[-800:]!r}"
                )

            return exp
        finally:
            # Always unstage. (Future: if accept + keep_success, leave it; but
            # that's risky because next-round baseline would then include this
            # feature and shift the ground truth. Keep clean by default.)
            if staged_path.exists() and not self.settings.keep_success:
                staged_path.unlink()
