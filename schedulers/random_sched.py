"""
Random — assign each job to a uniformly random site. Ignores locality, load
and everything else; the floor every other baseline should beat.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING, Optional

from env.interfaces import SchedulerInterface

if TYPE_CHECKING:
    from env.simulator import HEPSimulator
    from env.core import Job


class RandomScheduler(SchedulerInterface):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        return self._rng.randrange(len(env.sites))
