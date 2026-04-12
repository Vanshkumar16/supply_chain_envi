---
title: Supply Chain Disruption Management
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# 🏭 Supply Chain Disruption Management — OpenEnv

> **An OpenEnv-compliant RL environment where an AI agent manages a real-world
> multi-node supply chain under stochastic disruptions.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-local--ui-orange)](http://localhost:8000/demo)
[![HuggingFace](https://img.shields.io/badge/🤗%20HF%20Space-openenv-yellow)](https://huggingface.co/spaces/openenv)

---

## 📋 Overview

Supply chain disruption management is one of the most economically impactful
real-world tasks. Companies lose **billions annually** to stock-outs, supplier
failures, and shipping lane congestion. This environment models a 5-node
distribution network and challenges an RL agent to:

- Maintain high **service levels** (demand fulfilment rate)
- Minimise **operational costs** (holding, reorder, backlog penalties)
- React intelligently to **cascading disruptions** (supplier failures, lane congestion)

The environment exposes the full **OpenEnv interface** — `step()` / `reset()` /
`state()` — over HTTP + WebSocket via FastAPI, and is ready to deploy to
Hugging Face Spaces with a single `docker build`.

## Quick demo (one-liner)

Start the server and run the demo UI + inference report using the included script:

```bash
./run_supply_chain_env.sh
```

Or on Windows (PowerShell):

```powershell
.\run_supply_chain_env.ps1 -PORT 8000
```

After the run open `http://localhost:8000/demo` to interact and view the generated report.

---

## 🌐 Network Topology

```
         ┌────────┐      S1->W1      ┌────────┐
         │   S1   │ ──────────────►  │        │   W1->R1   ┌────────┐
         │Supplier│                  │   W1   │ ──────────►│   R1   │
         └────────┘                  │Warehouse│            │ Retail │
                                     │        │            └────────┘
         ┌────────┐      S2->W1      │        │   W1->R2   ┌────────┐
         │   S2   │ ──────────────►  │        │ ──────────►│   R2   │
         │Supplier│                  └────────┘            │ Retail │
         └────────┘                                        └────────┘

  Nodes  : S1, S2 (suppliers)  |  W1 (warehouse)  |  R1, R2 (retail)
  Lanes  : S1->W1, S2->W1 (inbound)  |  W1->R1, W1->R2 (outbound)
```

---

## 🎮 Action Space

**Type:** Continuous — all values are floats in `[0.0, 1.0]`

| Field | Type | Keys | Description |
|---|---|---|---|
| `reorder_quantities` | `Dict[str, float]` | `S1`, `S2` | Fraction of max reorder capacity to order from each supplier |
| `rerouting_weights` | `Dict[str, float]` | `S1->W1`, `S2->W1`, `W1->R1`, `W1->R2` | Relative weight for each shipping lane (normalised per source) |
| `supplier_activation` | `Dict[str, float]` | `S1`, `S2` | Activation level: `0.0` = suspend, `1.0` = fully activate |

**Example action:**
```json
{
  "reorder_quantities":  {"S1": 0.8, "S2": 0.6},
  "rerouting_weights":   {"S1->W1": 0.7, "S2->W1": 0.3,
                          "W1->R1": 0.5, "W1->R2": 0.5},
  "supplier_activation": {"S1": 1.0, "S2": 0.8}
}
```

---

## 👁️ Observation Space

`SupplyChainObservation` contains:

### Per-node (5 nodes: S1, S2, W1, R1, R2)
| Field | Type | Range | Description |
|---|---|---|---|
| `inventory_level` | float | [0,1] | Current stock as fraction of capacity |
| `demand_forecast` | float | [0,1] | Predicted demand next step |
| `disruption_active` | bool | — | Whether node is disrupted |
| `disruption_severity` | float | [0,1] | Severity of active disruption |
| `backlog` | float | [0,1] | Unfulfilled orders |

### Per-supplier (S1, S2)
| Field | Type | Range | Description |
|---|---|---|---|
| `reliability` | float | [0,1] | Current supplier reliability |
| `lead_time` | float | [0,1] | Normalised lead time (lower = better) |
| `active` | bool | — | Whether supplier is active |

### Per-lane (4 lanes)
| Field | Type | Range | Description |
|---|---|---|---|
| `capacity_used` | float | [0,1] | Fraction of lane capacity in use |
| `congestion` | float | [0,1] | Congestion level |
| `disrupted` | bool | — | Whether lane is disrupted |

### Global KPIs (partial-progress reward signals)
| Field | Type | Description |
|---|---|---|
| `service_level` | float | Fraction of demand fulfilled this step |
| `total_cost` | float | Normalised total cost this step |
| `disruption_count` | int | Number of active disruptions |
| `network_resilience` | float | Composite resilience score |

---

## 🏆 Tasks

### Task 0 — Easy (`task_0`)
- **Goal:** Keep `service_level ≥ 0.80` for **10 steps**
- **Disruptions:** None
- **Purpose:** Learn basic reorder and routing decisions
- **Expected baseline score:** `~0.55`

### Task 1 — Medium (`task_1`)
- **Goal:** Keep `service_level ≥ 0.75` for **20 steps**
- **Disruptions:** Supplier S1 disrupted at step 5 (severity 0.7)
- **Purpose:** React to supplier failure; switch to S2, buffer via warehouse
- **Expected baseline score:** `~0.40`

### Task 2 — Hard (`task_2`)
- **Goal:** Keep `service_level ≥ 0.70` AND `cost ≤ 0.40` for **30 steps**
- **Disruptions:** S2 fails at step 5, lane W1→R1 congested at step 15
- **Purpose:** Multi-objective optimisation under cascading failure
- **Expected baseline score:** `~0.25`

---

## 🎯 Reward Function

Dense shaped reward at every step (not just terminal):

```
reward = + service_level × 2.0      # main signal — demand fulfilment
         - cost × 0.5               # cost efficiency
         - disruption_load × 0.3    # penalise unmanaged disruptions
         - backlog × 0.5            # penalise stock-outs
         + resilience_bonus × 0.3   # reward keeping redundant suppliers active
```

**Range:** approximately `[-1.5, +2.3]`

This provides dense feedback so agents learn *during* the episode, not just
from a binary end-of-episode signal.

---

## 📊 Grader Scoring

All graders are **deterministic and reproducible**.

```
service_score  = mean(service_levels) / target_service_level  [capped at 1.0]
cost_score     = 1 - clamp(mean_cost / max_cost, 0, 1)        [task_2 only]
completion     = steps_completed / max_steps

task_0/1:  score = service_score × completion
task_2:    score = (0.6×service_score + 0.4×cost_score) × completion
```

---

## 🚀 Setup & Usage

### Quick Start (local)

```bash
# 1. Clone
git clone https://huggingface.co/spaces/<your-username>/supply-chain-env
cd supply-chain-env

# 2. Install
pip install fastapi uvicorn pydantic openai requests numpy pyyaml

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 4. Open docs
open http://localhost:8000/docs
```

### Docker (recommended)

```bash
# Build
docker build -t supply-chain-env .

# Run
docker run -p 8000:8000 supply-chain-env

# Verify
curl http://localhost:8000/
```

### Run inference (LLM agent)

```bash
# Set credentials (mandatory)
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here

# Option A — connect to already-running server
export ENV_BASE_URL=http://localhost:8000
python inference.py

# Option B — spin up from local Docker image
export LOCAL_IMAGE_NAME=supply-chain-env
python inference.py
```

Inference emits structured stdout (3 task blocks):
```
[START] task=task_0 env=supply_chain_env model=meta-llama/...
[STEP]  step=1 action={...} reward=1.85 done=false error=null
...
[END]   success=true steps=10 score=0.820 rewards=1.85,1.90,...

[START] task=task_1 env=supply_chain_env model=meta-llama/...
...
[END]   success=false steps=20 score=0.410 rewards=...

[START] task=task_2 env=supply_chain_env model=meta-llama/...
...
[END]   success=false steps=30 score=0.260 rewards=...
```

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

### Validate before submitting

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-username-supply-chain-env.hf.space
```

Runs 3 automated checks: HF Space ping, Docker build, openenv validate.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Episode metadata |
| `POST` | `/grade` | Get grader score |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/openenv.yaml` | Serve manifest |
| `WS` | `/ws` | WebSocket interface |
| `GET` | `/docs` | Swagger UI |

### Reset request
```json
{"task_id": "task_1", "seed": 42}
```

### Step request
```json
{
  "reorder_quantities":  {"S1": 0.8, "S2": 0.7},
  "rerouting_weights":   {"S1->W1": 0.6, "S2->W1": 0.4, "W1->R1": 0.5, "W1->R2": 0.5},
  "supplier_activation": {"S1": 1.0, "S2": 1.0}
}
```

### WebSocket protocol
```json
// Client sends:
{"type": "reset",  "task_id": "task_0", "seed": 42}
{"type": "step",   "action": { ...SupplyChainAction... }}
{"type": "state"}

// Server responds:
{"type": "reset_result",  "observation": {...}}
{"type": "step_result",   "observation": {...}, "reward": 1.23, "done": false, "info": {...}}
{"type": "state_result",  "state": {...}}
```

---

## 🐳 Hugging Face Spaces Deployment

```bash
# Login to HF
huggingface-cli login

# Create Space (Docker SDK, tagged openenv)
huggingface-cli repo create supply-chain-env --type space --space-sdk docker

# Push
git remote add hf https://huggingface.co/spaces/<your-username>/supply-chain-env
git push hf main
```

The Space will auto-deploy and expose port 8000. Set these Space secrets:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

---

## 📁 Project Structure

```
supply_chain_env/
├── models.py              # Pydantic Action, Observation, State, StepResult
├── client.py              # Sync + Async EnvClient
├── graders.py             # Standalone deterministic graders
├── inference.py           # LLM baseline inference script (OpenAI client)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── validate-submission.sh # Pre-submission validator (3 checks)
├── Dockerfile             # Container for HF Spaces
├── .env.example           # Environment variable template
├── README.md              # This file
├── server/
│   ├── app.py             # FastAPI application + WebSocket
│   ├── environment.py     # Core simulation engine
│   └── requirements.txt   # Docker dependencies
├── tests/
│   ├── test_environment.py
│   └── test_graders.py
└── outputs/
    ├── logs/
    └── evals/
```

---

## 📈 Baseline Scores (seed=42, reproducible)

| Task | Difficulty | LLM Agent Score | Pass Threshold |
|---|---|---|---|
| task_0 | Easy | ~0.55 | 0.70 |
| task_1 | Medium | ~0.40 | 0.55 |
| task_2 | Hard | ~0.25 | 0.40 |

*Scores produced by `python inference.py --all-tasks --seed 42`*

---

## 📄 License
 
MIT License — see [LICENSE](LICENSE)
