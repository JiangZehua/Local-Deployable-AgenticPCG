"""Termination condition checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Config

if TYPE_CHECKING:
    from ..evaluator import EvalResult


class TerminationChecker:
    """Checks termination conditions for optimization."""

    def __init__(self, config: Config):
        """Initialize termination checker.

        Args:
            config: Configuration with termination settings.
        """
        self.config = config
        self.multiplier = config.termination.change_budget_multiplier
        self.height = config.env.height
        self.width = config.env.width

        # Compute budget
        self.change_budget = int(self.multiplier * self.height * self.width)

        # Track changes
        self._total_changes = 0

    @property
    def total_changes(self) -> int:
        """Total tiles changed so far."""
        return self._total_changes

    @property
    def remaining_budget(self) -> int:
        """Remaining change budget."""
        return max(0, self.change_budget - self._total_changes)

    def add_changes(self, num_changes: int) -> None:
        """Record tile changes.

        Args:
            num_changes: Number of tiles changed.
        """
        self._total_changes += num_changes

    def _exact_target_reached(self, result: "EvalResult | None") -> bool:
        """Return whether a domain's configured exact objective is satisfied."""
        if result is None or not result.valid:
            return False
        metrics_by_domain = {
            "zelda": ("player_key", "key_door", "solution_length"),
            "smb_static": (
                "pipe_count",
                "enemy_tile_count",
                "coin_tile_count",
                "block_tile_count",
            ),
        }
        metric_names = metrics_by_domain.get(result.problem_type)
        if metric_names is None:
            return False
        exact_targets = []
        for name in metric_names:
            target = getattr(self.config.targets, name, None)
            if target is not None and target.mode == "target":
                exact_targets.append((name, target.value))
        return bool(exact_targets) and all(
            result.metrics.get(name) == value for name, value in exact_targets
        )

    def should_terminate(self, result: "EvalResult | None" = None) -> bool:
        """Check if optimization should terminate.

        Returns:
            True if change budget exhausted.
        """
        # The explicit paper switch separates evaluator-driven exact stopping
        # from commit/rollback. ``mode_default`` preserves legacy behavior.
        exact_stop = getattr(
            self.config.optimizer, "evaluator_exact_stop", "mode_default"
        )
        if exact_stop not in {"mode_default", "on", "off"}:
            raise ValueError(
                f"Unknown optimizer.evaluator_exact_stop={exact_stop!r}. "
                "Expected mode_default, on, or off."
            )
        evaluator_controls_stopping = (
            self.config.optimizer.ablation_mode != "no_evaluation_loop"
            if exact_stop == "mode_default"
            else exact_stop == "on"
        )
        if not evaluator_controls_stopping:
            return self._total_changes >= self.change_budget
        return self._exact_target_reached(result) or self._total_changes >= self.change_budget

    def set_total_changes(self, total: int) -> None:
        """Set total changes directly (used for resume).

        Args:
            total: Total tiles changed to restore.
        """
        self._total_changes = total

    def reset(self) -> None:
        """Reset change counter."""
        self._total_changes = 0

    def status(self) -> dict:
        """Get current termination status.

        Returns:
            Dictionary with budget info.
        """
        return {
            "total_changes": self._total_changes,
            "change_budget": self.change_budget,
            "remaining_budget": self.remaining_budget,
            "budget_exhausted": self.should_terminate(),
            "budget_percent_used": (self._total_changes / self.change_budget * 100) if self.change_budget > 0 else 100,
        }

    def format_status(self) -> str:
        """Format status for display.

        Returns:
            Human-readable status string.
        """
        status = self.status()
        return (
            f"Changes: {status['total_changes']}/{status['change_budget']} "
            f"({status['budget_percent_used']:.1f}% used, "
            f"{status['remaining_budget']} remaining)"
        )
