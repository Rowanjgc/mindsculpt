"""Verified execution and teacher-forced replay.

Public surface:

* :func:`prepare_plan` turns a trace into an explicit replay plan.
* :func:`execute_replay` runs it and reports measured agreement.
* :class:`HuggingFaceReplayModel` / :class:`TorchReplayModel` are the
  supported adapters.  Optional instrumentation backends plug in at the same
  seam rather than replacing it.
"""

from wsp.replay.backends import (
    HuggingFaceReplayModel,
    ReplayModel,
    TorchReplayModel,
    TokenizationSupport,
    load_local_replay_model,
)
from wsp.replay.engine import ReplayExecution, execute_replay, parameter_fingerprint
from wsp.replay.models import (
    SCHEMA_PLAN,
    SCHEMA_RESULT,
    ReplayAgreement,
    ReplayCheck,
    ReplayExecutionConfig,
    ReplayExpectation,
    ReplayForwardInfo,
    ReplayIdentity,
    ReplayModelInfo,
    ReplayPlan,
    ReplayResult,
    ReplayTokenAgreement,
    ReplayTolerances,
)
from wsp.replay.plan import plan_checksum, prepare_plan, sequence_checksum

__all__ = [
    "SCHEMA_PLAN",
    "SCHEMA_RESULT",
    "HuggingFaceReplayModel",
    "ReplayAgreement",
    "ReplayCheck",
    "ReplayExecution",
    "ReplayExecutionConfig",
    "ReplayExpectation",
    "ReplayForwardInfo",
    "ReplayIdentity",
    "ReplayModel",
    "ReplayModelInfo",
    "ReplayPlan",
    "ReplayResult",
    "ReplayTokenAgreement",
    "ReplayTolerances",
    "TokenizationSupport",
    "TorchReplayModel",
    "execute_replay",
    "load_local_replay_model",
    "parameter_fingerprint",
    "plan_checksum",
    "prepare_plan",
    "sequence_checksum",
]
