"""
Path management system for creating and retrieving run-specific workspace subdirectories.
"""

from pathlib import Path


class PathManager:
    """Manages the creation and retrieval of experiment workspace directories.

    Creates a structured output hierarchy for every experiment:
    base_dir/
        runs/
            experiment_name/
                checkpoints/
                logs/
                samples/
                configs/
    """

    def __init__(self, experiment_name: str, base_dir: str = "outputs") -> None:
        """Initialize the PathManager and generate runtime directories.

        Args:
            experiment_name: Unique tag identifier for the run.
            base_dir: Root directory for storing all experiments. Defaults to "outputs".
        """
        self._experiment_name = experiment_name
        self._base_path = Path(base_dir).resolve()
        self._run_path = self._base_path / "runs" / self._experiment_name

        # Define subfolders
        self._checkpoint_path = self._run_path / "checkpoints"
        self._log_path = self._run_path / "logs"
        self._sample_path = self._run_path / "samples"
        self._config_path = self._run_path / "configs"

        # Create directories immediately
        self.create_directories()

    def create_directories(self) -> None:
        """Create all experiment-specific subdirectories if they do not exist."""
        self._run_path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._sample_path.mkdir(parents=True, exist_ok=True)
        self._config_path.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """Return the top-level run directory.

        Returns:
            Path: Absolute run directory.
        """
        return self._run_path

    @property
    def checkpoint_dir(self) -> Path:
        """Return the run's checkpoints directory.

        Returns:
            Path: Absolute checkpoints directory.
        """
        return self._checkpoint_path

    @property
    def log_dir(self) -> Path:
        """Return the run's logging directory.

        Returns:
            Path: Absolute logging directory.
        """
        return self._log_path

    @property
    def sample_dir(self) -> Path:
        """Return the run's validation sample output directory.

        Returns:
            Path: Absolute validation samples directory.
        """
        return self._sample_path

    @property
    def config_dir(self) -> Path:
        """Return the run's configurations backup directory.

        Returns:
            Path: Absolute configs backup directory.
        """
        return self._config_path
