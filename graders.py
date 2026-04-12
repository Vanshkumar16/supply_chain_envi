"""
graders.py — Supply Chain Disruption Management
================================================
Deterministic, reproducible graders for all 3 tasks.
Each grader returns a score in [0.0, 1.0].

These graders are independent of the environment instance so they can be
run post-hoc against a recorded trajectory or inline during evaluation.

Usage (standalone):
    from graders import grade_trajectory, GRADERS
    score = grade_trajectory("task_0", service_levels=[...], costs=[...])
"""

from __future__ import annotations
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task specifications (mirrors TASKS in environment.py)
# ---------------------------------------------------------------------------

TASK_SPECS = {
    "task_0": {
        "difficulty":             "easy",
        "max_steps":              10,
        "target_service_level":   0.80,
        "max_cost":               1.0,
        "cost_constrained":       False,
    },
    "task_1": {
        "difficulty":             "medium",
        "max_steps":              20,
        "target_service_level":   0.75,
        "max_cost":               1.0,
        "cost_constrained":       False,
    },
    "task_2": {
        "difficulty":             "hard",
        "max_steps":              30,
        "target_service_level":   0.70,
        "max_cost":               0.40,
        "cost_constrained":       True,
    },
}


# ---------------------------------------------------------------------------
# Core grading logic
# ---------------------------------------------------------------------------

def grade_trajectory(
    task_id: str,
    service_levels: List[float],
    costs: List[float],
    steps_completed: Optional[int] = None,
) -> float:
    """
    Grade a trajectory deterministically.

    Parameters
    ----------
    task_id         : one of "task_0", "task_1", "task_2"
    service_levels  : per-step service level values [0,1]
    costs           : per-step normalised cost values [0,1]
    steps_completed : if None, inferred from len(service_levels)

    Returns
    -------
    score : float in [0.0, 1.0]

    Rubric
    ------
    service_score  = mean(service_levels) / target_sl  (capped at 1.0)
    cost_score     = 1 - clamp(mean_cost / max_cost, 0, 1)  [only if cost-constrained]
    completion     = steps_completed / max_steps  (partial episode penalty)

    task_0/1 (not cost-constrained):
        raw_score = service_score
    task_2 (cost-constrained):
        raw_score = 0.6 * service_score + 0.4 * cost_score

    final = raw_score * completion
    """
    if task_id not in TASK_SPECS:
        raise ValueError(f"Unknown task_id '{task_id}'")

    spec = TASK_SPECS[task_id]
    n = len(service_levels)
    if n == 0:
        return 0.0

    steps_done = steps_completed if steps_completed is not None else n
    completion = min(steps_done / spec["max_steps"], 1.0)

    mean_sl   = sum(service_levels) / n
    service_score = min(mean_sl / spec["target_service_level"], 1.0)

    if spec["cost_constrained"] and costs:
        mean_cost  = sum(costs) / len(costs)
        cost_score = 1.0 - min(mean_cost / spec["max_cost"], 1.0)
        raw = 0.6 * service_score + 0.4 * cost_score
    else:
        raw = service_score

    return round(raw * completion, 4)


# ---------------------------------------------------------------------------
# Task-specific grader callables
# ---------------------------------------------------------------------------

class TaskGrader:
    """
    Callable grader object for a specific task.
    Accumulates trajectory data and returns score on demand.
    """

    def __init__(self, task_id: str):
        if task_id not in TASK_SPECS:
            raise ValueError(f"Unknown task_id '{task_id}'")
        self.task_id        = task_id
        self.spec           = TASK_SPECS[task_id]
        self._service_levels: List[float] = []
        self._costs:          List[float] = []

    def record_step(self, service_level: float, cost: float):
        """Call after each env step to accumulate trajectory data."""
        self._service_levels.append(service_level)
        self._costs.append(cost)

    def score(self) -> float:
        """Return current grader score [0.0–1.0]."""
        return grade_trajectory(
            task_id=self.task_id,
            service_levels=self._service_levels,
            costs=self._costs,
        )

    def reset(self):
        self._service_levels = []
        self._costs = []

    def __repr__(self):
        return (
            f"TaskGrader(task={self.task_id} "
            f"steps={len(self._service_levels)} "
            f"score={self.score():.4f})"
        )


# ---------------------------------------------------------------------------
# Convenience dict of graders
# ---------------------------------------------------------------------------

GRADERS = {tid: TaskGrader(tid) for tid in TASK_SPECS}


# ---------------------------------------------------------------------------
# Success / failure helpers
# ---------------------------------------------------------------------------

def is_success(task_id: str, score: float) -> bool:
    """
    Deterministic pass/fail criterion per task.
    task_0: score >= 0.70  (easy)
    task_1: score >= 0.55  (medium)
    task_2: score >= 0.40  (hard)
    """
    thresholds = {"task_0": 0.70, "task_1": 0.55, "task_2": 0.40}
    return score >= thresholds.get(task_id, 0.50)


def score_summary(scores: dict) -> str:
    """Return a human-readable summary of scores."""
    lines = ["Grader Score Summary", "-" * 40]
    for tid, score in scores.items():
        spec = TASK_SPECS.get(tid, {})
        diff = spec.get("difficulty", "?")
        status = "✓ PASS" if is_success(tid, score) else "✗ FAIL"
        lines.append(f"  {tid} ({diff:6s})  {score:.4f}  {status}")
    if len(scores) == 3:
        agg = sum(scores.values()) / 3
        lines.append(f"\n  Aggregate : {agg:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running grader self-tests ...\n")

    # Perfect agent
    perfect_sl   = [1.0] * 10
    perfect_cost = [0.1] * 10
    s = grade_trajectory("task_0", perfect_sl, perfect_cost)
    assert s == 1.0, f"Expected 1.0, got {s}"
    print(f"task_0 perfect agent  : {s:.4f}  ✓")

    # Zero service level
    zero_sl = [0.0] * 10
    s2 = grade_trajectory("task_0", zero_sl, perfect_cost)
    assert s2 == 0.0, f"Expected 0.0, got {s2}"
    print(f"task_0 zero agent     : {s2:.4f}  ✓")

    # Partial episode
    s3 = grade_trajectory("task_1", [0.80] * 10, [0.20] * 10, steps_completed=10)
    assert 0.0 < s3 < 1.0, f"Expected partial score, got {s3}"
    print(f"task_1 partial ep     : {s3:.4f}  ✓")

    # Hard task cost constraint
    s4 = grade_trajectory("task_2", [0.80] * 30, [0.60] * 30)
    s5 = grade_trajectory("task_2", [0.80] * 30, [0.20] * 30)
    assert s5 > s4, "Lower cost should yield higher score on task_2"
    print(f"task_2 high cost      : {s4:.4f}")
    print(f"task_2 low cost       : {s5:.4f}  ✓ (lower cost scores higher)")

    # TaskGrader object
    g = TaskGrader("task_0")
    for _ in range(10):
        g.record_step(0.90, 0.15)
    print(f"\nTaskGrader test       : {g}")
    assert g.score() > 0.0

    print("\nAll grader tests passed ✓")
