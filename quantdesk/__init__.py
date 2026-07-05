"""quant-desk — deterministic, auditable quantitative-finance math + a
no-arithmetic-by-LLM dispatch agent."""
from . import quant
from .agent import dispatch, TOOLS

__all__ = ["quant", "dispatch", "TOOLS"]
