"""Heuristic scheduling policies; no model endpoint is required."""
from .min_e import MinEScheduler
from .min_t import MinTScheduler
from .min_ct import MCTScheduler
from .diana import DIANAScheduler
from .least_loaded import LeastLoadedScheduler
from .random_sched import RandomScheduler

__all__ = ["MinEScheduler", "MinTScheduler", "MCTScheduler", "DIANAScheduler",
           "LeastLoadedScheduler", "RandomScheduler"]
