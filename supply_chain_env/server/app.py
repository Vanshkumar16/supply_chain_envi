"""
Supply Chain Disruption Management — FastAPI Server
====================================================
Exposes the OpenEnv interface over HTTP + WebSocket.

Endpoints
---------
GET  /                  health check → {"status": "ok", ...}
POST /reset             start new episode → SupplyChainObservation
POST /step              advance one step  → StepResult
GET  /state             episode metadata  → SupplyChainState
GET  /tasks             list available tasks
GET  /openenv.yaml      serve the manifest

WebSocket
---------
WS   /ws                streaming interface (reset / step / state messages)
"""

from __future__ import annotations

import json
import os
import sys

# Make models importable from parent dir when run inside server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from models import (
    StepResult,
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
)
from server.environment import SupplyChainEnvironment, TASKS

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Supply Chain Disruption Management — OpenEnv",
    description=(
        "An OpenEnv-compliant RL environment where an AI agent manages a "
        "multi-node supply chain under stochastic disruptions. "
        "Implements step() / reset() / state() with typed Pydantic models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# One environment instance per process (suitable for single-session HF Space)
_env: SupplyChainEnvironment = SupplyChainEnvironment(task_id="task_0")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_0"
    seed:    int = 42


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health():
    state = _env.state()
    return {
        "status":       "ok",
        "environment":  "supply_chain_env",
        "version":      "1.0.0",
        "task_id":      state.task_id,
        "step":         state.step_count,
        "done":         state.done,
        "openenv_spec": "step/reset/state",
    }


@app.post("/reset", response_model=SupplyChainObservation, tags=["openenv"])
def reset(req: ResetRequest = None):
    """
    Reset the environment to a fresh episode.
    Optionally specify task_id (task_0/task_1/task_2) and seed.
    """
    global _env
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Choose from {list(TASKS)}")
    _env = SupplyChainEnvironment(task_id=req.task_id, seed=req.seed)
    return _env.reset()


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(action: SupplyChainAction):
    """Execute one action and return (observation, reward, done, info)."""
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/state", response_model=SupplyChainState, tags=["openenv"])
def state():
    """Return current episode metadata."""
    return _env.state()


@app.get("/tasks", tags=["info"])
def list_tasks():
    """List all available tasks with their descriptions."""
    return {
        tid: {
            "description": cfg["description"],
            "difficulty":  cfg["difficulty"],
            "max_steps":   cfg["max_steps"],
            "target_service_level": cfg["target_service_level"],
            "max_cost":    cfg["max_cost"],
        }
        for tid, cfg in TASKS.items()
    }


@app.post("/grade", tags=["openenv"])
def grade():
    """Return the grader score for the current episode [0.0 – 1.0]."""
    return {"score": _env.grade(), "task_id": _env.task_id}


@app.get("/openenv.yaml", response_class=PlainTextResponse, tags=["info"])
def serve_openenv_yaml():
    """Serve the openenv.yaml manifest."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    try:
        with open(yaml_path) as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(404, "openenv.yaml not found")


# ---------------------------------------------------------------------------
# WebSocket  (streaming interface expected by openenv-core EnvClient)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface for the OpenEnv EnvClient.

    Message format (JSON):
      {"type": "reset",  "task_id": "task_0", "seed": 42}
      {"type": "step",   "action": { ... SupplyChainAction fields ... }}
      {"type": "state"}

    Response format:
      {"type": "reset_result",  "observation": {...}}
      {"type": "step_result",   "observation": {...}, "reward": ..., "done": ..., "info": {...}}
      {"type": "state_result",  "state": {...}}
      {"type": "error",         "message": "..."}
    """
    global _env
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id", "task_0")
                seed    = msg.get("seed", 42)
                if task_id not in TASKS:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown task_id '{task_id}'"}
                    )
                    continue
                _env = SupplyChainEnvironment(task_id=task_id, seed=seed)
                obs  = _env.reset()
                await websocket.send_json(
                    {"type": "reset_result", "observation": obs.model_dump()}
                )

            elif msg_type == "step":
                try:
                    action = SupplyChainAction(**msg.get("action", {}))
                    result = _env.step(action)
                    await websocket.send_json(
                        {
                            "type":        "step_result",
                            "observation": result.observation.model_dump(),
                            "reward":      result.reward,
                            "done":        result.done,
                            "info":        result.info,
                        }
                    )
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "state":
                s = _env.state()
                await websocket.send_json({"type": "state_result", "state": s.model_dump()})

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type '{msg_type}'"}
                )

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
