"""AverageMeter utility class tracking averages of metrics.

Author: Antigravity
Purpose: Track and accumulate values across epochs or batches and expose public averages.
"""


class AverageMeter:
    """Computes and stores the average and current value of a running metric.

    Exposes a frozen public interface consisting of reset(), update(value),
    and the average property.
    """

    def __init__(self) -> None:
        """Initialize AverageMeter."""
        self._val = 0.0
        self._avg = 0.0
        self._sum = 0.0
        self._count = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset all metric tracking state variables."""
        self._val = 0.0
        self._avg = 0.0
        self._sum = 0.0
        self._count = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Update tracking metrics state variables.

        Args:
            value (float): The new metric value.
            n (int): The quantity weight factor (e.g. batch size). Defaults to 1.
        """
        self._val = value
        self._sum += value * n
        self._count += n
        self._avg = self._sum / self._count if self._count > 0 else 0.0

    @property
    def average(self) -> float:
        """Retrieve the computed average score.

        Returns:
            float: Averaged metric value.
        """
        return self._avg
