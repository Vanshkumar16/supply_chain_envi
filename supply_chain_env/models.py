"""
Supply Chain Disruption Management — OpenEnv Models
====================================================
Typed Pydantic models: Action, Observation, State, StepResult.
Compatible with openenv-core and openenv validate.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action  (continuous, all floats in [0.0, 1.0])
# ---------------------------------------------------------------------------

class SupplyChainAction(BaseModel):
    """
    Continuous action space controlling three resource-allocation vectors.

    reorder_quantities   : node_id  -> fraction of max reorder capacity [0,1]
    rerouting_weights    : "A->B"   -> relative lane weight [0,1]
                           (env normalises per-source so they sum to 1)
    supplier_activation  : supplier_id -> activation level [0,1]
                           (0=suspend, 1=fully activate)
    """
    reorder_quantities:  Dict[str, float] = Field(default_factory=dict)
    rerouting_weights:   Dict[str, float] = Field(default_factory=dict)
    supplier_activation: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation sub-models
# ---------------------------------------------------------------------------

class NodeStatus(BaseModel):
    node_id:              str
    inventory_level:      float  # [0,1] fraction of capacity
    demand_forecast:      float  # [0,1] expected demand next step / capacity
    disruption_active:    bool
    disruption_severity:  float  # [0,1]
    backlog:              float  # [0,1] unfulfilled orders / capacity


class SupplierStatus(BaseModel):
    supplier_id:  str
    reliability:  float  # [0,1]
    lead_time:    float  # [0,1] normalised (lower = better)
    active:       bool


class LaneStatus(BaseModel):
    lane_id:       str    # "src->dst"
    capacity_used: float  # [0,1]
    congestion:    float  # [0,1]
    disrupted:     bool


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SupplyChainObservation(BaseModel):
    nodes:     List[NodeStatus]
    suppliers: List[SupplierStatus]
    lanes:     List[LaneStatus]

    # Global KPIs — the partial-progress reward signals
    service_level:      float  # fraction of demand fulfilled this step [0,1]
    total_cost:         float  # normalised cost this step [0,1]
    disruption_count:   int    # active disruptions across the network
    network_resilience: float  # composite resilience score [0,1]

    step_number: int
    episode_id:  str
    message:     str = ""


# ---------------------------------------------------------------------------
# State  (episode metadata)
# ---------------------------------------------------------------------------

class SupplyChainState(BaseModel):
    episode_id:       str
    step_count:       int
    max_steps:        int
    cumulative_reward: float
    task_id:          str
    task_description: str
    difficulty:       str   # "easy" | "medium" | "hard"
    done:             bool
    score:            float  # grader score so far [0,1]


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: SupplyChainObservation
    reward:      float
    done:        bool
    info:        Dict = Field(default_factory=dict)
