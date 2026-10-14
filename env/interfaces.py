"""
Interface for the scheduling decision:
  SchedulerInterface – assigns an incoming job to a site

Implementers provide concrete scheduling policies.  The simulator calls
the hooks at the right moments; the interface returns actions, and the
simulator executes them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .simulator import HEPSimulator
    from .core import Job


# ====================================================================== #
# Job Scheduling Interface
# ====================================================================== #
class SchedulerInterface(ABC):
    """
    Called by the simulator whenever a new job arrives.

    schedule(job, env) → site_index | None
      - Return an integer in [0, len(env.sites)) to assign the job.
      - Return None to defer the decision; the simulator will retry
        after a brief delay (1 µs).  Use sparingly — tight defer loops
        are expensive.
    """

    @abstractmethod
    def schedule(self, job: "Job", env: "HEPSimulator") -> Optional[int]:
        ...

    # Optional hook called after every assignment (for stateful schedulers)
    def on_job_assigned(self, job: "Job", site_idx: int, env: "HEPSimulator") -> None:
        pass

    # Optional hook called after every job completion
    def on_job_complete(self, job: "Job", env: "HEPSimulator") -> None:
        pass
