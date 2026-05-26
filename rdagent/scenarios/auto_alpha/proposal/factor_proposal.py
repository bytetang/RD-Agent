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
    """Translates eval_feature.py's JSON envelope into HypothesisFeedback.

    Decision logic:
      - status != ok                                          -> reject
      - status == ok and delta.val_logloss < accept_threshold -> accept (improvement)
      - n_rows_train drops > nan_drop_threshold fraction      -> reject (NaN drift)
      - else                                                   -> reject (no signal)
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        # Accept only when BOTH val and test logloss improve.
        # Round-1 (5-feature baseline) found 70% val accepts with 79% test
        # robustness; Round-2 (10-feature baseline) saw val accepts drop to
        # 40% but test robustness collapsed to 15% — most "val winners"
        # were noise / val-overfit. The dual-improvement rule keeps test
        # honest by treating it as a second hold-out, not just a report.
        # Both deltas must be strictly negative (we'll only call a candidate
        # a winner if it survives BOTH the validation and test set).
        self.accept_logloss_delta_threshold: float = 0.0
        # If n_rows_train drops by more than this fraction, suspect NaN leakage.
        self.nan_drop_fraction_threshold: float = 0.05

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

        envelope: dict[str, Any] | None = getattr(exp, "result", None)
        if not envelope or not isinstance(envelope, dict):
            return HypothesisFeedback(
                reason="no result envelope on experiment",
                decision=False,
                acceptable=False,
            )

        status = envelope.get("status")
        feature_name = envelope.get("feature_name", "?")

        if status != "ok":
            findings = envelope.get("findings") or []
            stderr_tail = envelope.get("stderr_tail") or ""
            return HypothesisFeedback(
                reason=(
                    f"feature {feature_name!r} failed with status={status}. "
                    f"findings={findings} stderr_tail_excerpt={stderr_tail[-400:]}"
                ),
                decision=False,
                acceptable=False,
                observations=f"status={status}",
            )

        metrics = envelope.get("metrics") or {}
        baseline = envelope.get("baseline_metrics") or {}
        delta = envelope.get("delta") or {}

        val_delta = delta.get("val_logloss")
        test_delta = delta.get("test_logloss")
        cand_rows = metrics.get("n_rows_train")
        base_rows = baseline.get("n_rows_train")

        nan_drop_flag = False
        if isinstance(cand_rows, (int, float)) and isinstance(base_rows, (int, float)) and base_rows > 0:
            drop_frac = (base_rows - cand_rows) / base_rows
            if drop_frac > self.nan_drop_fraction_threshold:
                nan_drop_flag = True

        val_passes = (
            isinstance(val_delta, (int, float))
            and val_delta < self.accept_logloss_delta_threshold
        )
        test_passes = (
            isinstance(test_delta, (int, float))
            and test_delta < self.accept_logloss_delta_threshold
        )
        decision = val_passes and test_passes and not nan_drop_flag

        reason_parts = [
            f"feature={feature_name}",
            f"status=ok",
            f"val_logloss={metrics.get('val_logloss')}",
            f"val_logloss_delta={val_delta}",
            f"test_logloss_delta={test_delta}",
            f"val_auc_delta={delta.get('val_auc')}",
            f"n_rows_train_delta={delta.get('n_rows_train')}",
        ]
        # Spell out exactly why the candidate was rejected so the next-round
        # LLM has a precise signal to learn from.
        if not val_passes:
            reason_parts.append("REJECTED: val_logloss did not improve")
        elif not test_passes:
            reason_parts.append(
                "REJECTED: val improved but test did not — likely val-overfit"
            )
        if nan_drop_flag:
            reason_parts.append(
                f"REJECTED: n_rows_train drop fraction exceeds "
                f"{self.nan_drop_fraction_threshold:.2%}"
            )

        return HypothesisFeedback(
            reason="; ".join(str(p) for p in reason_parts),
            decision=decision,
            acceptable=decision,
            observations=(
                f"metrics={metrics}\nbaseline={baseline}\ndelta={delta}"
            ),
        )
