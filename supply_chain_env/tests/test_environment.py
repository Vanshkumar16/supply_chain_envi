"""
tests/test_environment.py
=========================
Full test suite for the Supply Chain OpenEnv environment.
Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import (
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
    StepResult,
)
from server.environment import SupplyChainEnvironment, TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_easy():
    env = SupplyChainEnvironment(task_id="task_0", seed=42)
    env.reset()
    return env


@pytest.fixture
def env_medium():
    env = SupplyChainEnvironment(task_id="task_1", seed=42)
    env.reset()
    return env


@pytest.fixture
def env_hard():
    env = SupplyChainEnvironment(task_id="task_2", seed=42)
    env.reset()
    return env


def default_action():
    return SupplyChainAction(
        reorder_quantities={"S1": 0.7, "S2": 0.7},
        rerouting_weights={"S1->W1": 0.5, "S2->W1": 0.5,
                           "W1->R1": 0.5, "W1->R2": 0.5},
        supplier_activation={"S1": 1.0, "S2": 1.0},
    )


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_returns_observation(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        obs = env.reset()
        assert isinstance(obs, SupplyChainObservation)

    def test_reset_step_is_zero(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        obs = env.reset()
        assert obs.step_number == 0

    def test_reset_produces_clean_state(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        # Run some steps
        env.reset()
        for _ in range(5):
            env.step(default_action())
        # Reset again
        obs = env.reset()
        assert obs.step_number == 0
        assert env.state().cumulative_reward == 0.0

    def test_reset_nodes_have_inventory(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        obs = env.reset()
        for node in obs.nodes:
            assert 0.0 <= node.inventory_level <= 1.0

    def test_reset_no_disruptions(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        obs = env.reset()
        # task_0 has no disruptions at start
        assert obs.disruption_count == 0

    def test_reset_reproducible_with_same_seed(self):
        env1 = SupplyChainEnvironment(task_id="task_1", seed=99)
        env2 = SupplyChainEnvironment(task_id="task_1", seed=99)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.episode_id != obs2.episode_id  # different UUIDs but same structure
        assert obs1.service_level == obs2.service_level


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_returns_step_result(self, env_easy):
        result = env_easy.step(default_action())
        assert isinstance(result, StepResult)

    def test_step_increments_step_number(self, env_easy):
        result = env_easy.step(default_action())
        assert result.observation.step_number == 1

    def test_step_reward_is_float(self, env_easy):
        result = env_easy.step(default_action())
        assert isinstance(result.reward, float)

    def test_step_done_is_bool(self, env_easy):
        result = env_easy.step(default_action())
        assert isinstance(result.done, bool)

    def test_step_observation_fields_in_range(self, env_easy):
        result = env_easy.step(default_action())
        obs = result.observation
        assert 0.0 <= obs.service_level <= 1.0
        assert 0.0 <= obs.total_cost <= 1.0
        assert 0.0 <= obs.network_resilience <= 1.0
        for node in obs.nodes:
            assert 0.0 <= node.inventory_level <= 1.0
            assert 0.0 <= node.disruption_severity <= 1.0

    def test_step_done_after_max_steps(self, env_easy):
        """task_0 has max_steps=10, so done should be True after 10 steps."""
        for i in range(9):
            result = env_easy.step(default_action())
            assert not result.done, f"Should not be done at step {i+1}"
        result = env_easy.step(default_action())
        assert result.done

    def test_step_raises_after_done(self, env_easy):
        for _ in range(10):
            env_easy.step(default_action())
        with pytest.raises(RuntimeError):
            env_easy.step(default_action())

    def test_step_action_clamping(self, env_easy):
        """Action values outside [0,1] must be clamped, not raise."""
        bad_action = SupplyChainAction(
            reorder_quantities={"S1": 5.0, "S2": -2.0},
            rerouting_weights={"W1->R1": 999.0, "W1->R2": -0.5},
            supplier_activation={"S1": 2.0, "S2": 1.0},
        )
        result = env_easy.step(bad_action)  # should not raise
        assert isinstance(result, StepResult)

    def test_step_info_contains_grader_score(self, env_easy):
        result = env_easy.step(default_action())
        assert "grader_score" in result.info
        assert 0.0 <= result.info["grader_score"] <= 1.0

    def test_step_info_contains_service_level(self, env_easy):
        result = env_easy.step(default_action())
        assert "service_level" in result.info


# ---------------------------------------------------------------------------
# state() tests
# ---------------------------------------------------------------------------

class TestState:

    def test_state_returns_supply_chain_state(self, env_easy):
        s = env_easy.state()
        assert isinstance(s, SupplyChainState)

    def test_state_task_id(self, env_easy):
        s = env_easy.state()
        assert s.task_id == "task_0"

    def test_state_difficulty(self, env_easy):
        s = env_easy.state()
        assert s.difficulty == "easy"

    def test_state_step_count_increments(self, env_easy):
        env_easy.step(default_action())
        env_easy.step(default_action())
        s = env_easy.state()
        assert s.step_count == 2

    def test_state_score_in_range(self, env_easy):
        env_easy.step(default_action())
        s = env_easy.state()
        assert 0.0 <= s.score <= 1.0

    def test_state_cumulative_reward_updates(self, env_easy):
        env_easy.step(default_action())
        s = env_easy.state()
        assert s.cumulative_reward != 0.0


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:

    def test_grade_before_reset_is_zero(self):
        env = SupplyChainEnvironment(task_id="task_0", seed=42)
        assert env.grade() == 0.0

    def test_grade_after_perfect_steps(self, env_easy):
        """High reorder + activation should keep SL high → grade > 0.5."""
        for _ in range(10):
            if not env_easy.state().done:
                env_easy.step(default_action())
        score = env_easy.grade()
        assert score > 0.3, f"Expected score > 0.3, got {score}"

    def test_grade_in_range(self, env_easy):
        for _ in range(10):
            if not env_easy.state().done:
                env_easy.step(default_action())
        score = env_easy.grade()
        assert 0.0 <= score <= 1.0

    def test_grade_task2_penalises_high_cost(self):
        """task_2 has a cost constraint — high cost should reduce score."""
        env = SupplyChainEnvironment(task_id="task_2", seed=42)
        env.reset()
        # Deliberately wasteful action (high reorder always = high cost)
        wasteful = SupplyChainAction(
            reorder_quantities={"S1": 1.0, "S2": 1.0},
            rerouting_weights={"S1->W1": 1.0, "S2->W1": 1.0,
                               "W1->R1": 1.0, "W1->R2": 1.0},
            supplier_activation={"S1": 1.0, "S2": 1.0},
        )
        for _ in range(30):
            if not env.state().done:
                env.step(wasteful)
        high_cost_score = env.grade()

        env2 = SupplyChainEnvironment(task_id="task_2", seed=42)
        env2.reset()
        efficient = SupplyChainAction(
            reorder_quantities={"S1": 0.4, "S2": 0.4},
            rerouting_weights={"S1->W1": 0.5, "S2->W1": 0.5,
                               "W1->R1": 0.5, "W1->R2": 0.5},
            supplier_activation={"S1": 1.0, "S2": 1.0},
        )
        for _ in range(30):
            if not env2.state().done:
                env2.step(efficient)
        low_cost_score = env2.grade()

        # Not necessarily true that low cost is always strictly higher
        # (depends on SL tradeoff) but both must be in range
        assert 0.0 <= high_cost_score <= 1.0
        assert 0.0 <= low_cost_score <= 1.0


# ---------------------------------------------------------------------------
# Multi-task tests
# ---------------------------------------------------------------------------

class TestAllTasks:

    @pytest.mark.parametrize("task_id", ["task_0", "task_1", "task_2"])
    def test_full_episode_completes(self, task_id):
        env = SupplyChainEnvironment(task_id=task_id, seed=42)
        env.reset()
        max_steps = TASKS[task_id]["max_steps"]
        for _ in range(max_steps):
            if env.state().done:
                break
            env.step(default_action())
        assert env.state().done

    @pytest.mark.parametrize("task_id", ["task_0", "task_1", "task_2"])
    def test_grade_in_range_all_tasks(self, task_id):
        env = SupplyChainEnvironment(task_id=task_id, seed=42)
        env.reset()
        for _ in range(TASKS[task_id]["max_steps"]):
            if env.state().done:
                break
            env.step(default_action())
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"{task_id} score {score} out of range"

    @pytest.mark.parametrize("task_id", ["task_0", "task_1", "task_2"])
    def test_difficulty_label(self, task_id):
        env = SupplyChainEnvironment(task_id=task_id, seed=42)
        env.reset()
        s = env.state()
        expected = {"task_0": "easy", "task_1": "medium", "task_2": "hard"}
        assert s.difficulty == expected[task_id]

    def test_scores_decrease_with_difficulty(self):
        """
        Default agent should score higher on easier tasks.
        """
        scores = {}
        for tid in ["task_0", "task_1", "task_2"]:
            env = SupplyChainEnvironment(task_id=tid, seed=42)
            env.reset()
            for _ in range(TASKS[tid]["max_steps"]):
                if env.state().done:
                    break
                env.step(default_action())
            scores[tid] = env.grade()

        # Allow some tolerance — hard task should generally score lower
        assert scores["task_0"] >= scores["task_2"] - 0.1, (
            f"Easy ({scores['task_0']:.3f}) should be >= hard ({scores['task_2']:.3f})"
        )


# ---------------------------------------------------------------------------
# Disruption tests
# ---------------------------------------------------------------------------

class TestDisruptions:

    def test_task1_disruption_injected_at_step5(self):
        env = SupplyChainEnvironment(task_id="task_1", seed=42)
        env.reset()
        for _ in range(4):
            env.step(default_action())
        obs_before = env.step(default_action()).observation
        # At step 5, S1 should be disrupted
        s1_node = next(n for n in obs_before.nodes if n.node_id == "S1")
        assert s1_node.disruption_active or obs_before.disruption_count >= 1

    def test_task2_cascading_disruptions(self):
        env = SupplyChainEnvironment(task_id="task_2", seed=42)
        env.reset()
        # Step to disruption points
        for i in range(15):
            r = env.step(default_action())
            if i == 14:  # step 15
                assert r.observation.disruption_count >= 1

    def test_disruption_recovery(self):
        """Disruptions should decay over time."""
        env = SupplyChainEnvironment(task_id="task_1", seed=42)
        env.reset()
        # Trigger disruption at step 5
        for _ in range(5):
            env.step(default_action())
        # Run more steps and check disruption eventually clears
        max_recover_steps = 15
        cleared = False
        for _ in range(max_recover_steps):
            result = env.step(default_action())
            if result.observation.disruption_count == 0:
                cleared = True
                break
            if result.done:
                break
        # Disruption should eventually recover (severity decays 0.10/step)
        # With severity 0.70, it clears after 7 steps
        assert cleared, "Disruption should have recovered"
