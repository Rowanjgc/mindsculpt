"""Token alignment and span resolution.

Public surface:

* :func:`build_alignment` maps recorded trace tokens onto forward positions.
* :func:`resolve_span` maps a behavioral span onto those same positions.
* :class:`TokenAlignment` / :class:`SpanResolution` are the versioned records
  that other artifacts reference.
"""

from wsp.alignment.build import build_alignment
from wsp.alignment.ids import new_alignment_id, new_span_id
from wsp.alignment.models import (
    PREDICTION_NOTE,
    PREDICTION_SEMANTICS,
    SCHEMA_ALIGNMENT,
    SCHEMA_SPAN,
    AlignmentPosition,
    AlignmentSegment,
    SpanResolution,
    TokenAlignment,
)
from wsp.alignment.spans import resolve_span

__all__ = [
    "PREDICTION_NOTE",
    "PREDICTION_SEMANTICS",
    "SCHEMA_ALIGNMENT",
    "SCHEMA_SPAN",
    "AlignmentPosition",
    "AlignmentSegment",
    "SpanResolution",
    "TokenAlignment",
    "build_alignment",
    "new_alignment_id",
    "new_span_id",
    "resolve_span",
]
