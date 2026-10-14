"""TransferNetwork: bandwidth matrix governing inter-site and local transfers."""
from __future__ import annotations

from typing import List
import numpy as np


class TransferNetwork:
    """
    Bandwidth matrix  A[src][dst]  in bytes/second.

    Convention (matching the paper):
      - A[n][n]   = local LAN bandwidth at site n  (disk → CPU staging)
      - A[m][n]   = WAN bandwidth from site m to site n  (m ≠ n)

    Typically  A[n][n] = lan_multiplier * A[m][n]  (default multiplier = 10).
    """

    def __init__(self, bandwidth_matrix: np.ndarray):
        assert bandwidth_matrix.ndim == 2, "bandwidth_matrix must be 2-D"
        n, m = bandwidth_matrix.shape
        assert n == m, "bandwidth_matrix must be square"
        self.bandwidth_matrix = bandwidth_matrix.astype(float)
        self.n_sites = n

    # ------------------------------------------------------------------ #
    @classmethod
    def from_wan_matrix(
        cls,
        wan_matrix: np.ndarray,
        lan_bandwidth: np.ndarray | None = None,
        lan_multiplier: float = 10.0,
    ) -> "TransferNetwork":
        """
        Build a full bandwidth matrix from a WAN matrix.

        wan_matrix[i][j]  bandwidth between site i and j  (i ≠ j).
        lan_bandwidth[i]  local LAN bandwidth at site i;
                          if None, computed as lan_multiplier * max(wan_matrix[i]).
        """
        n = wan_matrix.shape[0]
        bw = wan_matrix.astype(float).copy()
        for i in range(n):
            if lan_bandwidth is not None:
                bw[i, i] = float(lan_bandwidth[i])
            else:
                row_max = np.max(np.concatenate([wan_matrix[i, :i], wan_matrix[i, i + 1 :]]))
                bw[i, i] = lan_multiplier * row_max
        return cls(bw)

    # ------------------------------------------------------------------ #
    def bandwidth(self, src: int, dst: int) -> float:
        return self.bandwidth_matrix[src, dst]

    def transfer_time(self, src: int, dst: int, size_bytes: float) -> float:
        bw = self.bandwidth_matrix[src, dst]
        if bw <= 0:
            return float("inf")
        return size_bytes / bw

    def staging_time(self, site: int, size_bytes: float) -> float:
        """Local disk-to-CPU staging time at `site`."""
        return self.transfer_time(site, site, size_bytes)

    def best_source(self, dst: int, candidates: List[int]) -> int:
        """Return the candidate with the highest bandwidth to `dst`."""
        return max(candidates, key=lambda s: self.bandwidth_matrix[s, dst])

    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "n_sites": self.n_sites,
            "bandwidth_matrix_bytes_per_sec": self.bandwidth_matrix.tolist(),
        }
