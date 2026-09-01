"""Shim: EhrAgent import `tracing`, còn code thật nằm ở `adapt_tracing.py` (repo root)
để ReAct / algo / agentdriver dùng chung một module."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapt_tracing import (  # noqa: F401,E402
    Step,
    flush,
    is_enabled,
    setup_logging,
    step,
    wrap_openai_client,
)
