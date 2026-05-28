"""
Proposal pipeline for auto_alpha factor mining.

- AutoAlphaFactorHypothesisGen: LLM proposes the next feature direction.
- AutoAlphaFactorHypothesis2Experiment: LLM turns hypothesis into one
  concrete AutoAlphaFactorTask.
- AutoAlphaFactorExperiment2Feedback: parses eval_feature.py's JSON envelope,
  computes val_logloss delta vs baseline, returns HypothesisFeedback.
"""
from __future__ import annotations

import json
from typing import Any

from rdagent.components.proposal import (
    FactorHypothesis2Experiment,
    FactorHypothesisGen,
)
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis,
    HypothesisFeedback,
    Scenario,
    Trace,
)
from rdagent.scenarios.auto_alpha.experiment.factor_experiment import (
    AutoAlphaFactorExperiment,
    AutoAlphaFactorTask,
)
from rdagent.utils.agent.tpl import T


def _prior_proposal_names(trace: Trace) -> tuple[list[str], list[str]]:
    """Walk trace.hist and split prior proposed factor_names into
    (accepted, rejected). Used to give the LLM an explicit "don't repeat
    these" list — observed in Round-2 that the LLM proposed eom 3× and
    vwap_dev 2× because the prompt only listed accepted features.
    """
    accepted: list[str] = []
    rejected: list[str] = []
    for past_exp, fb in trace.hist:
        if not isinstance(past_exp, AutoAlphaFactorExperiment):
            continue
        for task in past_exp.sub_tasks or []:
            name = getattr(task, "factor_name", None)
            if not name:
                continue
            target = accepted if (fb and getattr(fb, "decision", False)) else rejected
            target.append(name)
    return accepted, rejected


class AutoAlphaFactorHypothesisGen(FactorHypothesisGen):
    """Asks the LLM to propose the next feature direction.

    Inherits the LLM call plumbing from `FactorHypothesisGen ->
    LLMHypothesisGen`. We just need to build the prompt context and parse
    the JSON response into a `Hypothesis`.
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> tuple[dict, bool]:
        if trace.hist:
            hypothesis_and_feedback = T(
                "scenarios.auto_alpha.prompts:hypothesis_and_feedback"
            ).r(trace=trace)
            last = trace.hist[-1]
            last_hypothesis_and_feedback = T(
                "scenarios.auto_alpha.prompts:last_hypothesis_and_feedback"
            ).r(experiment=last[0], feedback=last[1])
        else:
            hypothesis_and_feedback = (
                "No previous hypothesis and feedback — this is the first round."
            )
            last_hypothesis_and_feedback = hypothesis_and_feedback

        accepted_names, rejected_names = _prior_proposal_names(trace)

        context = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": (
                "Try simple, vectorized pandas features first (rolling means, "
                "EWM smoothing, volatility, range bands). Save look-ahead-prone "
                "ideas for the labels module, never put them in features."
                if len(trace.hist) < 10
                else "Now try features that combine multiple existing signals, "
                "or capture microstructure (trade intensity, volume profile)."
            ),
            "hypothesis_output_format": T(
                "scenarios.auto_alpha.prompts:auto_alpha_hypothesis_output_format"
            ).r(),
            "hypothesis_specification": T(
                "scenarios.auto_alpha.prompts:auto_alpha_hypothesis_specification"
            ).r(
                baseline_feature_names=getattr(self.scen, "baseline_feature_names", []),
                baseline_feature_columns=getattr(self.scen, "baseline_feature_columns", []),
                prior_accepted_names=accepted_names,
                prior_rejected_names=rejected_names,
            ),
        }
        return context, True

    def convert_response(self, response: str) -> Hypothesis:
        resp = json.loads(response)
        return Hypothesis(
            hypothesis=resp.get("hypothesis", ""),
            reason=resp.get("reason", ""),
            concise_reason=resp.get("concise_reason", ""),
            concise_observation=resp.get("concise_observation", ""),
            concise_justification=resp.get("concise_justification", ""),
            concise_knowledge=resp.get("concise_knowledge", ""),
        )


class AutoAlphaFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    """Turns a hypothesis into exactly ONE AutoAlphaFactorTask."""

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> tuple[dict, bool]:
        scenario_desc = trace.scen.get_scenario_all_desc()
        if trace.hist:
            hyp_and_fb = T(
                "scenarios.auto_alpha.prompts:hypothesis_and_feedback"
            ).r(trace=trace)
        else:
            hyp_and_fb = "No previous hypothesis and feedback available."

        accepted_names, rejected_names = _prior_proposal_names(trace)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario_desc,
            "hypothesis_and_feedback": hyp_and_fb,
            "experiment_output_format": T(
                "scenarios.auto_alpha.prompts:auto_alpha_experiment_output_format"
            ).r(
                baseline_feature_names=getattr(trace.scen, "baseline_feature_names", []),
                baseline_feature_columns=getattr(trace.scen, "baseline_feature_columns", []),
                prior_accepted_names=accepted_names,
                prior_rejected_names=rejected_names,
            ),
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(
        self,
        response: str,
        hypothesis: Hypothesis,
        trace: Trace,
    ) -> AutoAlphaFactorExperiment:
        resp = json.loads(response)
        # Enforce single-feature contract: take the first entry if the LLM
        # returns multiple.
        if not isinstance(resp, dict) or not resp:
            raise ValueError(f"hypothesis2experiment response is empty: {resp!r}")
        factor_name, spec = next(iter(resp.items()))
        if not isinstance(spec, dict):
            raise ValueError(
                f"expected dict spec for {factor_name!r}, got {type(spec).__name__}"
            )

        task = AutoAlphaFactorTask(
            factor_name=factor_name,
            factor_description=spec.get("description", ""),
            factor_formulation=spec.get("formulation", ""),
            variables=spec.get("variables", {}) or {},
        )
        exp = AutoAlphaFactorExperiment(sub_tasks=[task], hypothesis=hypothesis)

        # Dedup against successful prior experiments by factor_name.
        prior_names: set[str] = set()
        for past_exp, fb in trace.hist:
            if not fb or not getattr(fb, "decision", False):
                continue
            if not isinstance(past_exp, AutoAlphaFactorExperiment):
                continue
            for past_task in past_exp.sub_tasks:
                if isinstance(past_task, AutoAlphaFactorTask):
                    prior_names.add(past_task.factor_name)
        if factor_name in prior_names:
            raise ValueError(
                f"factor name {factor_name!r} already accepted in trace; "
                f"the hypothesis generator should propose something new"
            )

        return exp


class AutoAlphaFactorExperiment2Feedback(Experiment2Feedback):
    """Translates bench's `_compare.json` digest into HypothesisFeedback.

    The digest has the shape:
        {
          "schema_version": 1,
          "tag": "...",
          "compared_on": "val" | "test",
          "features": [...],
          "total_cells": N,
          "headline_columns": {"classification": "val_acc", "regression": "val_ic_pearson"},
          "cells": [
            {"interval": "1m", "label_kind": "bucket", "horizon_s": 900,
             "model": "lgbm", "task": "classification",
             "headline_metric": "val_acc",
             "baseline_headline": 0.731, "bench_headline": 0.733,
             "delta_headline": 0.002, "delta_ic_p": ..., "delta_icir_p": ...,
             "delta_acc": ..., "baseline_hash": "...", "bench_hash": "..."},
            ...
          ]
        }

    Decision: improved_cells >= accept_min_improved_cells AND
              worst delta_headline >= -accept_max_drop.
    Improved = delta_headline > 0 (headline metrics are all "higher is better"
    given the current baseline sign convention).
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        from rdagent.app.auto_alpha_loop.conf import AUTO_ALPHA_FACTOR_PROP_SETTING
        self.settings = AUTO_ALPHA_FACTOR_PROP_SETTING

    def generate_feedback(
        self,
        exp: Experiment,
        trace: Trace,
        exception: Exception | None = None,
    ) -> HypothesisFeedback:
        if exception is not None:
            return HypothesisFeedback(
                reason=f"experiment raised: {exception}",
                decision=False,
                acceptable=False,
                exception=exception,
            )

        digest: dict[str, Any] | None = getattr(exp, "result", None)
        if not digest or not isinstance(digest, dict):
            return HypothesisFeedback(
                reason="no bench digest on experiment",
                decision=False,
                acceptable=False,
            )

        feature_name = digest.get("_feature_col", "?")
        bench_rc = digest.get("_bench_returncode", 0)
        cells = digest.get("cells") or []

        if not cells:
            return HypothesisFeedback(
                reason=(
                    f"feature {feature_name!r}: bench digest had 0 cells "
                    f"(bench_rc={bench_rc}, "
                    f"stderr_tail={digest.get('_bench_stderr_tail','')[-300:]!r})"
                ),
                decision=False,
                acceptable=False,
                observations="empty digest",
            )

        # Aggregate. delta_headline may be None on a cell that failed to run.
        improved = 0
        worsened: list[tuple[str, float]] = []
        missing = 0
        deltas_by_task: dict[str, list[float]] = {"classification": [], "regression": []}

        for c in cells:
            cell_id = (
                f"{c.get('model')}_{c.get('interval')}_"
                f"{c.get('label_kind')}_{c.get('horizon_s')}s"
            )
            delta_h = c.get("delta_headline")
            task = c.get("task", "?")
            if delta_h is None:
                missing += 1
                continue
            if task in deltas_by_task:
                deltas_by_task[task].append(float(delta_h))
            if delta_h > 0:
                improved += 1
            worsened.append((cell_id, float(delta_h)))

        worst_cell, worst_delta = min(worsened, key=lambda kv: kv[1]) if worsened else ("?", 0.0)
        best_cell, best_delta = max(worsened, key=lambda kv: kv[1]) if worsened else ("?", 0.0)

        min_improved = self.settings.accept_min_improved_cells
        max_drop = self.settings.accept_max_drop
        decision = (
            improved >= min_improved
            and worst_delta >= -max_drop
            and missing == 0
        )

        cls_avg = (
            sum(deltas_by_task["classification"]) / len(deltas_by_task["classification"])
            if deltas_by_task["classification"] else None
        )
        reg_avg = (
            sum(deltas_by_task["regression"]) / len(deltas_by_task["regression"])
            if deltas_by_task["regression"] else None
        )

        reason_parts = [
            f"feature={feature_name}",
            f"cells={len(cells)} improved={improved} missing={missing}",
            f"best={best_cell}:{best_delta:+.5f}",
            f"worst={worst_cell}:{worst_delta:+.5f}",
            f"avg_dheadline cls={cls_avg if cls_avg is None else f'{cls_avg:+.5f}'} "
            f"reg={reg_avg if reg_avg is None else f'{reg_avg:+.5f}'}",
            f"rule: improved>={min_improved} AND worst>=-{max_drop}",
        ]
        if not decision:
            if missing:
                reason_parts.append(f"REJECTED: {missing} cells missing delta")
            elif improved < min_improved:
                reason_parts.append(
                    f"REJECTED: only {improved}/{len(cells)} cells improved "
                    f"(need >={min_improved})"
                )
            elif worst_delta < -max_drop:
                reason_parts.append(
                    f"REJECTED: worst cell {worst_cell} dropped {worst_delta:+.5f} "
                    f"(< -{max_drop})"
                )

        # Compact per-cell line list for the next-round prompt to learn from.
        per_cell = []
        for c in cells:
            per_cell.append(
                f"{c.get('model')}/{c.get('interval')}/{c.get('label_kind')}/"
                f"{c.get('horizon_s')}s {c.get('headline_metric')}: "
                f"{c.get('baseline_headline')} -> {c.get('bench_headline')} "
                f"(Δ={c.get('delta_headline')})"
            )

        return HypothesisFeedback(
            reason="; ".join(str(p) for p in reason_parts),
            decision=decision,
            acceptable=decision,
            observations="cells:\n  " + "\n  ".join(per_cell),
        )
