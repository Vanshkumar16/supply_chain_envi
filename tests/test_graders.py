"""
tests/test_graders.py
=====================
Tests for the standalone grader module.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graders import grade_trajectory, TaskGrader, GRADERS, is_success, score_summary


class TestGradeTrajectory:

    def test_perfect_easy(self):
        s = grade_trajectory("task_0", [1.0]*10, [0.0]*10)
        assert s == 1.0

    def test_zero_service_level(self):
        s = grade_trajectory("task_0", [0.0]*10, [0.0]*10)
        assert s == 0.0

    def test_partial_episode(self):
        s = grade_trajectory("task_0", [0.9]*5, [0.1]*5, steps_completed=5)
        assert 0.0 < s < 1.0

    def test_hard_task_cost_penalty(self):
        s_highcost = grade_trajectory("task_2", [0.9]*30, [0.9]*30)
        s_lowcost  = grade_trajectory("task_2", [0.9]*30, [0.1]*30)
        assert s_lowcost > s_highcost

    def test_score_in_range_all_tasks(self):
        for tid in ["task_0", "task_1", "task_2"]:
            s = grade_trajectory(tid, [0.75]*15, [0.25]*15)
            assert 0.0 <= s <= 1.0

    def test_empty_trajectory(self):
        s = grade_trajectory("task_0", [], [])
        assert s == 0.0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            grade_trajectory("task_99", [0.8]*10, [0.2]*10)


class TestTaskGrader:

    def test_initial_score_is_zero(self):
        g = TaskGrader("task_0")
        assert g.score() == 0.0

    def test_records_steps(self):
        g = TaskGrader("task_1")
        for _ in range(5):
            g.record_step(0.8, 0.2)
        assert g.score() > 0.0

    def test_reset_clears_history(self):
        g = TaskGrader("task_0")
        g.record_step(0.9, 0.1)
        g.reset()
        assert g.score() == 0.0

    def test_graders_dict_has_all_tasks(self):
        assert set(GRADERS.keys()) == {"task_0", "task_1", "task_2"}


class TestIsSuccess:

    def test_easy_pass(self):
        assert is_success("task_0", 0.75)

    def test_easy_fail(self):
        assert not is_success("task_0", 0.60)

    def test_hard_pass(self):
        assert is_success("task_2", 0.45)

    def test_hard_fail(self):
        assert not is_success("task_2", 0.30)


class TestScoreSummary:

    def test_summary_contains_task_ids(self):
        scores = {"task_0": 0.8, "task_1": 0.6, "task_2": 0.4}
        summary = score_summary(scores)
        for tid in scores:
            assert tid in summary

    def test_summary_contains_aggregate(self):
        scores = {"task_0": 0.8, "task_1": 0.6, "task_2": 0.4}
        summary = score_summary(scores)
        assert "Aggregate" in summary
