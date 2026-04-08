"""
Supply Chain Disruption Management — Core Environment
======================================================
Implements the OpenEnv Environment interface:
  reset()  -> SupplyChainObservation
  step()   -> StepResult
  state()  -> SupplyChainState

Domain
------
A 5-node distribution network (2 suppliers → 1 warehouse → 2 retail nodes)
with stochastic disruptions. The agent must keep shelves stocked, costs low,
and the network resilient under pressure.

Nodes
-----
  S1, S2  : supplier nodes  (produce inventory)
  W1      : central warehouse
  R1, R2  : retail/demand nodes

Lanes
-----
  S1->W1, S2->W1  : inbound
  W1->R1, W1->R2  : outbound

Tasks
-----
  task_0  (easy)   : Keep service_level >= 0.80 for 10 steps, no disruptions
  task_1  (medium) : Keep service_level >= 0.75 for 20 steps with 1 random
                     supplier disruption injected at step 5
  task_2  (hard)   : Keep service_level >= 0.70 AND total_cost <= 0.40 for
                     30 steps with cascading disruptions (lane + supplier)
                     injected at steps 5 and 15
"""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    LaneStatus,
    NodeStatus,
    StepResult,
    SupplierStatus,
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODES    = ["S1", "S2", "W1", "R1", "R2"]
SUPPLIERS = ["S1", "S2"]
WAREHOUSES = ["W1"]
RETAIL   = ["R1", "R2"]
LANES    = ["S1->W1", "S2->W1", "W1->R1", "W1->R2"]

MAX_CAPACITY = 1.0   # normalised

TASKS = {
    "task_0": {
        "description": (
            "EASY — Keep service level ≥ 0.80 for 10 steps. "
            "No disruptions will occur. Learn basic reordering."
        ),
        "difficulty": "easy",
        "max_steps": 10,
        "target_service_level": 0.80,
        "max_cost": 1.0,          # no cost constraint
        "disruption_schedule": {},  # step -> list of (type, target, severity)
    },
    "task_1": {
        "description": (
            "MEDIUM — Keep service level ≥ 0.75 for 20 steps. "
            "Supplier S1 is disrupted at step 5 (severity 0.7). "
            "Switch to S2 and manage warehouse buffer."
        ),
        "difficulty": "medium",
        "max_steps": 20,
        "target_service_level": 0.75,
        "max_cost": 1.0,
        "disruption_schedule": {
            5: [("supplier", "S1", 0.70)],
        },
    },
    "task_2": {
        "description": (
            "HARD — Keep service level ≥ 0.70 AND cost ≤ 0.40 for 30 steps. "
            "Cascading disruptions: S2 disrupted at step 5, lane W1->R1 "
            "congested at step 15. Requires multi-objective optimisation."
        ),
        "difficulty": "hard",
        "max_steps": 30,
        "target_service_level": 0.70,
        "max_cost": 0.40,
        "disruption_schedule": {
            5:  [("supplier", "S2", 0.80)],
            15: [("lane",     "W1->R1", 0.90)],
        },
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnvironment:
    """
    OpenEnv-compatible supply chain disruption management environment.
    Thread-safe for single-session use (one episode at a time).
    """

    def __init__(self, task_id: str = "task_0", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASKS)}")
        self.task_id   = task_id
        self.task_cfg  = TASKS[task_id]
        self.seed      = seed
        self._rng      = random.Random(seed)

        # Episode state (initialised by reset)
        self._episode_id: str = ""
        self._step:       int = 0
        self._done:       bool = True

        # Network state
        self._inventory:   Dict[str, float] = {}
        self._backlog:     Dict[str, float] = {}
        self._disruptions: Dict[str, float] = {}   # node/lane -> severity
        self._lane_congestion: Dict[str, float] = {}
        self._supplier_active: Dict[str, bool]  = {}
        self._supplier_reliability: Dict[str, float] = {}

        # KPI tracking
        self._cumulative_reward: float = 0.0
        self._service_levels:    List[float] = []
        self._costs:             List[float] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> SupplyChainObservation:
        """Initialise a fresh episode and return the initial observation."""
        self._rng      = random.Random(self.seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step     = 0
        self._done     = False
        self._cumulative_reward = 0.0
        self._service_levels    = []
        self._costs             = []

        # Inventory: suppliers full, warehouse half, retail quarter
        self._inventory = {
            "S1": 0.90, "S2": 0.90,
            "W1": 0.50,
            "R1": 0.30, "R2": 0.30,
        }
        self._backlog = {n: 0.0 for n in NODES}
        self._disruptions = {}
        self._lane_congestion = {l: 0.0 for l in LANES}
        self._supplier_active = {"S1": True, "S2": True}
        self._supplier_reliability = {"S1": 0.95, "S2": 0.92}

        return self._build_observation("Episode started. Manage the supply chain.")

    def step(self, action: SupplyChainAction) -> StepResult:
        """Execute one action and advance the simulation by one step."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1

        # 1. Apply scheduled disruptions
        self._apply_scheduled_disruptions()

        # 2. Clamp & normalise action values
        action = self._sanitise_action(action)

        # 3. Supplier production → inbound shipments
        inbound = self._compute_inbound(action)

        # 4. Update warehouse inventory
        self._update_warehouse(inbound)

        # 5. Outbound distribution to retail nodes
        service_level = self._distribute_to_retail(action)

        # 6. Demand arrives at retail (stochastic)
        self._apply_retail_demand()

        # 7. Compute step cost
        step_cost = self._compute_cost(action)

        # 8. Reward shaping (partial progress)
        reward = self._compute_reward(service_level, step_cost)
        self._cumulative_reward += reward
        self._service_levels.append(service_level)
        self._costs.append(step_cost)

        # 9. Natural recovery of disruptions
        self._recover_disruptions()

        # 10. Done?
        max_steps = self.task_cfg["max_steps"]
        self._done = (self._step >= max_steps)

        msg = (
            f"Step {self._step}/{max_steps} | "
            f"SL={service_level:.2f} cost={step_cost:.2f} reward={reward:+.3f}"
        )

        obs = self._build_observation(msg)
        info = {
            "step_cost":     step_cost,
            "service_level": service_level,
            "grader_score":  self._grade(),
        }
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> SupplyChainState:
        """Return current episode metadata."""
        return SupplyChainState(
            episode_id=self._episode_id,
            step_count=self._step,
            max_steps=self.task_cfg["max_steps"],
            cumulative_reward=round(self._cumulative_reward, 4),
            task_id=self.task_id,
            task_description=self.task_cfg["description"],
            difficulty=self.task_cfg["difficulty"],
            done=self._done,
            score=self._grade(),
        )

    # ------------------------------------------------------------------
    # Grader  (deterministic, reproducible)
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """
        Score the completed (or in-progress) episode [0.0 – 1.0].

        Scoring rubric
        --------------
        service_score  = mean(service_levels) / target_service_level, capped at 1.0
        cost_score     = 1.0 - clamp(mean_cost / max_cost, 0, 1)   [if max_cost < 1]
        completion     = 1.0 if episode ended naturally, else partial

        final = 0.6 * service_score + 0.4 * cost_score   (hard task)
              = service_score                              (easy/medium)
        """
        return self._grade()

    def _grade(self) -> float:
        if not self._service_levels:
            return 0.0

        target_sl  = self.task_cfg["target_service_level"]
        max_cost   = self.task_cfg["max_cost"]
        max_steps  = self.task_cfg["max_steps"]

        mean_sl   = sum(self._service_levels) / len(self._service_levels)
        mean_cost = sum(self._costs) / len(self._costs) if self._costs else 0.0

        service_score = min(mean_sl / target_sl, 1.0)

        if max_cost < 1.0:
            cost_score = 1.0 - min(mean_cost / max_cost, 1.0)
            raw = 0.6 * service_score + 0.4 * cost_score
        else:
            raw = service_score

        # Partial episode penalty
        completion = min(self._step / max_steps, 1.0)
        return round(raw * completion, 4)

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _apply_scheduled_disruptions(self):
        schedule = self.task_cfg["disruption_schedule"]
        if self._step in schedule:
            for (dtype, target, severity) in schedule[self._step]:
                if dtype == "supplier":
                    self._disruptions[target] = severity
                    self._supplier_reliability[target] = max(
                        0.05, self._supplier_reliability.get(target, 1.0) - severity * 0.8
                    )
                elif dtype == "lane":
                    self._disruptions[target] = severity
                    self._lane_congestion[target] = min(1.0, severity)

    def _sanitise_action(self, action: SupplyChainAction) -> SupplyChainAction:
        """Clamp all action values to [0, 1] to prevent exploits."""
        rq = {k: max(0.0, min(1.0, v)) for k, v in action.reorder_quantities.items()}
        rw = {k: max(0.0, min(1.0, v)) for k, v in action.rerouting_weights.items()}
        sa = {k: max(0.0, min(1.0, v)) for k, v in action.supplier_activation.items()}
        return SupplyChainAction(
            reorder_quantities=rq,
            rerouting_weights=rw,
            supplier_activation=sa,
        )

    def _compute_inbound(self, action: SupplyChainAction) -> Dict[str, float]:
        """Suppliers ship to warehouse based on action + reliability + disruption."""
        inbound: Dict[str, float] = {"W1": 0.0}
        for sid in SUPPLIERS:
            if not self._supplier_active.get(sid, True):
                continue
            activation = action.supplier_activation.get(sid, 1.0)
            reliability = self._supplier_reliability.get(sid, 1.0)
            disruption_penalty = self._disruptions.get(sid, 0.0)
            reorder_frac = action.reorder_quantities.get(sid, 0.5)

            # How much the supplier can actually ship
            available = self._inventory.get(sid, 0.0)
            effective_supply = (
                available
                * reorder_frac
                * reliability
                * activation
                * (1.0 - disruption_penalty * 0.8)
            )
            effective_supply = max(0.0, min(available, effective_supply))

            # Lane congestion reduces throughput
            lane = f"{sid}->W1"
            congestion = self._lane_congestion.get(lane, 0.0)
            lane_disrupted = self._disruptions.get(lane, 0.0)
            throughput = effective_supply * (1.0 - congestion * 0.5) * (1.0 - lane_disrupted * 0.7)

            inbound["W1"] = inbound.get("W1", 0.0) + throughput
            # Replenish supplier (they produce each step)
            self._inventory[sid] = min(1.0, self._inventory[sid] - effective_supply + 0.3)

        return inbound

    def _update_warehouse(self, inbound: Dict[str, float]):
        """Update warehouse inventory with inbound shipments."""
        for node, qty in inbound.items():
            self._inventory[node] = min(1.0, self._inventory.get(node, 0.0) + qty)

    def _distribute_to_retail(self, action: SupplyChainAction) -> float:
        """
        Push inventory from warehouse to retail nodes.
        Returns fraction of demand fulfilled (service level).
        """
        total_demand    = 0.0
        total_fulfilled = 0.0

        # Normalise rerouting weights
        outbound_lanes = ["W1->R1", "W1->R2"]
        weights = {}
        raw_total = sum(
            action.rerouting_weights.get(l, 0.5) for l in outbound_lanes
        )
        for l in outbound_lanes:
            weights[l] = action.rerouting_weights.get(l, 0.5) / max(raw_total, 1e-6)

        w1_inventory = self._inventory.get("W1", 0.0)

        for lane in outbound_lanes:
            dst = lane.split("->")[1]
            congestion    = self._lane_congestion.get(lane, 0.0)
            lane_disrupted = self._disruptions.get(lane, 0.0)

            demand = self._backlog.get(dst, 0.0) + self._rng.uniform(0.05, 0.20)
            total_demand += demand

            allocation = w1_inventory * weights[lane]
            effective   = allocation * (1.0 - congestion * 0.4) * (1.0 - lane_disrupted * 0.6)
            fulfilled   = min(demand, effective)
            total_fulfilled += fulfilled

            self._inventory[dst]  = min(1.0, self._inventory.get(dst, 0.0) + fulfilled)
            self._inventory["W1"] = max(0.0, w1_inventory - allocation)
            w1_inventory          = self._inventory["W1"]

            unfulfilled = max(0.0, demand - fulfilled)
            self._backlog[dst] = min(0.5, unfulfilled)

        if total_demand < 1e-9:
            return 1.0
        return round(min(1.0, total_fulfilled / total_demand), 4)

    def _apply_retail_demand(self):
        """Retail nodes consume from their own inventory (end-consumer demand)."""
        for r in RETAIL:
            consumption = self._rng.uniform(0.05, 0.15)
            self._inventory[r] = max(0.0, self._inventory.get(r, 0.0) - consumption)

    def _compute_cost(self, action: SupplyChainAction) -> float:
        """
        Normalised cost [0,1].
        Components: holding cost + reorder cost + penalty cost for backlogs.
        """
        holding = sum(self._inventory.values()) / len(NODES) * 0.15
        reorder = sum(action.reorder_quantities.values()) / max(len(SUPPLIERS), 1) * 0.30
        backlog_penalty = sum(self._backlog.values()) / len(NODES) * 0.55
        return round(min(1.0, holding + reorder + backlog_penalty), 4)

    def _compute_reward(self, service_level: float, cost: float) -> float:
        """
        Shaped reward providing partial-progress signal every step.

        Components (all normalised [-1, +1] range):
          +service_level * 2.0         (main signal)
          -cost * 0.5                  (cost efficiency)
          -disruption_load * 0.3       (penalise unmanaged disruptions)
          -backlog_load * 0.5          (penalise stock-outs)
          +resilience_bonus * 0.3      (reward keeping redundant suppliers active)
        """
        disruption_load = sum(self._disruptions.values()) / max(len(self._disruptions), 1) \
            if self._disruptions else 0.0
        backlog_load = sum(self._backlog.values()) / len(NODES)
        active_suppliers = sum(1 for s in SUPPLIERS if self._supplier_active.get(s, True))
        resilience_bonus = active_suppliers / len(SUPPLIERS)

        reward = (
            service_level * 2.0
            - cost * 0.5
            - disruption_load * 0.3
            - backlog_load * 0.5
            + resilience_bonus * 0.3
        )
        return round(reward, 4)

    def _recover_disruptions(self):
        """Disruptions naturally decay each step (partial recovery)."""
        recovered = []
        for key, severity in self._disruptions.items():
            new_severity = severity - 0.10
            if new_severity <= 0.0:
                recovered.append(key)
                # Restore reliability if it was a supplier
                if key in SUPPLIERS:
                    self._supplier_reliability[key] = min(
                        1.0, self._supplier_reliability.get(key, 0.5) + 0.15
                    )
                if key in LANES:
                    self._lane_congestion[key] = max(0.0, self._lane_congestion.get(key, 0.0) - 0.15)
            else:
                self._disruptions[key] = new_severity
        for key in recovered:
            del self._disruptions[key]

    def _build_observation(self, message: str = "") -> SupplyChainObservation:
        nodes = [
            NodeStatus(
                node_id=n,
                inventory_level=round(self._inventory.get(n, 0.0), 4),
                demand_forecast=round(self._rng.uniform(0.05, 0.20), 4),
                disruption_active=(n in self._disruptions),
                disruption_severity=round(self._disruptions.get(n, 0.0), 4),
                backlog=round(self._backlog.get(n, 0.0), 4),
            )
            for n in NODES
        ]
        suppliers = [
            SupplierStatus(
                supplier_id=s,
                reliability=round(self._supplier_reliability.get(s, 1.0), 4),
                lead_time=round(self._rng.uniform(0.1, 0.4), 4),
                active=self._supplier_active.get(s, True),
            )
            for s in SUPPLIERS
        ]
        lanes = [
            LaneStatus(
                lane_id=l,
                capacity_used=round(self._rng.uniform(0.2, 0.8), 4),
                congestion=round(self._lane_congestion.get(l, 0.0), 4),
                disrupted=(l in self._disruptions),
            )
            for l in LANES
        ]
        resilience = (
            sum(self._supplier_reliability.values()) / len(SUPPLIERS)
            * (1.0 - sum(self._disruptions.values()) / max(len(self._disruptions), 1)
               if self._disruptions else 1.0)
        )
        sl = self._service_levels[-1] if self._service_levels else 1.0
        cost = self._costs[-1] if self._costs else 0.0

        return SupplyChainObservation(
            nodes=nodes,
            suppliers=suppliers,
            lanes=lanes,
            service_level=round(sl, 4),
            total_cost=round(cost, 4),
            disruption_count=len(self._disruptions),
            network_resilience=round(min(1.0, max(0.0, resilience)), 4),
            step_number=self._step,
            episode_id=self._episode_id,
            message=message,
        )
