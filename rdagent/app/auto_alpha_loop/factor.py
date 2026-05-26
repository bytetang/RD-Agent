"""
auto_alpha_factor R&D loop driver.

Loop: propose -> code -> run -> evaluate -> feedback.

The loop is just a thin subclass of `RDLoop` with `running()` overridden to
re-raise on FactorEmptyError so failures become structured feedback rather
than silent skips.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from rdagent.app.auto_alpha_loop.conf import AUTO_ALPHA_FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, FactorEmptyError
from rdagent.log import rdagent_logger as logger


class AutoAlphaFactorRDLoop(RDLoop):
    """Same control flow as qlib's FactorRDLoop, but no qlib coupling."""

    skip_loop_error = (FactorEmptyError, CoderError)
    skip_loop_error_stepname = "feedback"

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error("auto_alpha runner returned None")
            raise FactorEmptyError("AutoAlpha factor evaluation returned None.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
    **kwargs: Any,
) -> None:
    """Auto R&D loop for auto-alpha-playground feature mining.

    Resume an interrupted session by passing the snapshot path.
    """
    if path is None:
        loop = AutoAlphaFactorRDLoop(AUTO_ALPHA_FACTOR_PROP_SETTING)
    else:
        loop = AutoAlphaFactorRDLoop.load(path, checkout=checkout)

    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
