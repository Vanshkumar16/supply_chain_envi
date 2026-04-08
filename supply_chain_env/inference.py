"""
inference.py — Supply Chain Disruption Management
==================================================
MANDATORY VARIABLES (set before running):
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name for from_docker_image() (optional if
                        ENV_BASE_URL points to a running server).
    ENV_BASE_URL        Override to point at an already-running server
                        (default: http://localhost:8000).

STDOUT FORMAT (mandatory — any deviation = wrong evaluation score):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - score formatted to 3 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line, no newlines within a line.
    - Each task runs its own [START]...[END] block (3 blocks total).
    - Each task score is in [0, 1].

Infra constraints:
    - Total runtime < 20 min
    - vcpu=2, memory=8gb
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (mandatory)
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")
USE_LOCAL_MODEL  = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_ID   = os.getenv("LOCAL_MODEL_ID", "openenv/supply-chain-model")

BENCHMARK = "supply_chain_env"

# ---------------------------------------------------------------------------
# Inference settings  (tuned to fit < 20 min total for 3 tasks)
# ---------------------------------------------------------------------------

TASK_MAX_STEPS: Dict[str, int] = {
    "task_0": 10,
    "task_1": 20,
    "task_2": 30,
}
SUCCESS_SCORE_THRESHOLD = 0.30   # normalised score in [0, 1]
TEMPERATURE  = 0.1
MAX_TOKENS   = 256
SEED         = 42

# ---------------------------------------------------------------------------
# Model initialization (OpenAI API or local model)
# ---------------------------------------------------------------------------

openai_client = None
local_model = None
local_tokenizer = None

def initialize_model():
    """Initialize model (API-based or local) with proper error handling."""
    global openai_client, local_model, local_tokenizer

    if USE_LOCAL_MODEL:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"[DEBUG] Loading local model: {LOCAL_MODEL_ID}", flush=True)
            local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
            local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID)
            print(f"[DEBUG] Local model loaded successfully", flush=True)
        except Exception as e:
            print(f"[DEBUG] Failed to load local model, falling back to API: {e}", flush=True)
            openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Mandatory stdout logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action string: no newlines, truncate long strings
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert supply chain manager AI agent controlling a 5-node network:
  Nodes   : S1, S2 (suppliers), W1 (warehouse), R1, R2 (retail)
  Lanes   : S1->W1, S2->W1 (inbound), W1->R1, W1->R2 (outbound)

Goal: maximise service_level (demand fulfilled) while keeping costs low.

Rules:
- If a supplier is disrupted (disruption_active=True), REDUCE its
  reorder_quantity and activation, and INCREASE the healthy supplier's values.
- If a lane is disrupted/congested, REDUCE its rerouting_weight and INCREASE
  the alternative lane's weight.
- Keep W1 inventory > 0.3 to buffer against disruptions.
- Respond with ONLY a valid JSON object — no markdown, no explanation.

Output format (all values floats 0.0-1.0):
{
  "reorder_quantities":  {"S1": 0.8, "S2": 0.8},
  "rerouting_weights":   {"S1->W1": 0.5, "S2->W1": 0.5, "W1->R1": 0.5, "W1->R2": 0.5},
  "supplier_activation": {"S1": 1.0, "S2": 1.0}
}
""").strip()

# ---------------------------------------------------------------------------
# Default fallback action
# ---------------------------------------------------------------------------

DEFAULT_ACTION_DICT = {
    "reorder_quantities":  {"S1": 0.7, "S2": 0.7},
    "rerouting_weights":   {"S1->W1": 0.5, "S2->W1": 0.5,
                            "W1->R1": 0.5, "W1->R2": 0.5},
    "supplier_activation": {"S1": 1.0, "S2": 1.0},
}

# ---------------------------------------------------------------------------
# Observation → user prompt
# ---------------------------------------------------------------------------

def obs_to_prompt(obs: dict, step: int, task_id: str) -> str:
    nodes_summary = "\n".join(
        f"  {n['node_id']}: inv={n['inventory_level']:.2f} "
        f"backlog={n['backlog']:.2f} "
        f"disrupted={'YES sev=' + str(round(n['disruption_severity'], 2)) if n['disruption_active'] else 'no'}"
        for n in obs["nodes"]
    )
    suppliers_summary = "\n".join(
        f"  {s['supplier_id']}: reliability={s['reliability']:.2f} "
        f"active={s['active']} lead_time={s['lead_time']:.2f}"
        for s in obs["suppliers"]
    )
    lanes_summary = "\n".join(
        f"  {l['lane_id']}: congestion={l['congestion']:.2f} disrupted={l['disrupted']}"
        for l in obs["lanes"]
    )
    return textwrap.dedent(f"""
Task: {task_id} | Step {step}

NODES:
{nodes_summary}

SUPPLIERS:
{suppliers_summary}

LANES:
{lanes_summary}

KPIs:
  service_level={obs['service_level']:.3f}
  total_cost={obs['total_cost']:.3f}
  disruption_count={obs['disruption_count']}
  network_resilience={obs['network_resilience']:.3f}

What is your action JSON?
""").strip()

# ---------------------------------------------------------------------------
# Parse LLM response → action dict
# ---------------------------------------------------------------------------

def parse_action(text: str) -> dict:
    """Extract JSON action from LLM response. Falls back to default on failure."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    # Find first {...} block
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        action = json.loads(text)
        required = {"reorder_quantities", "rerouting_weights", "supplier_activation"}
        if required.issubset(action.keys()):
            # Clamp all values to [0, 1]
            for field in required:
                action[field] = {
                    k: max(0.0, min(1.0, float(v)))
                    for k, v in action[field].items()
                }
            return action
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return DEFAULT_ACTION_DICT

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(obs: dict, step: int, task_id: str) -> tuple[dict, str]:
    """
    Call LLM and return (action_dict, action_str_for_logging).
    Falls back to default action on any error.
    Supports both API-based and local models.
    """
    prompt = obs_to_prompt(obs, step, task_id)
    raw = ""

    try:
        if local_model is not None and local_tokenizer is not None:
            # Use local model
            def get_local_action(obs: dict, step: int, task_id: str) -> str:
                prompt = obs_to_prompt(obs, step, task_id)
                inputs = local_tokenizer(
                    f"{SYSTEM_PROMPT}\n\nUser: {prompt}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                outputs = local_model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                response = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response

            raw = get_local_action(obs, step, task_id)
        elif openai_client is not None:
            # Use API-based model
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = (completion.choices[0].message.content or "").strip()
        else:
            raise RuntimeError("No model initialized (neither local nor API)")

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        raw = json.dumps(DEFAULT_ACTION_DICT)

    action_dict = parse_action(raw)
    # Compact one-line string for [STEP] log
    action_str = json.dumps(action_dict, separators=(",", ":"))
    return action_dict, action_str

# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

import requests as _requests

def _http_reset(base_url: str, task_id: str, seed: int) -> dict:
    r = _requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _http_step(base_url: str, action: dict) -> dict:
    r = _requests.post(
        f"{base_url}/step",
        json=action,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _http_grade(base_url: str) -> float:
    r = _requests.post(f"{base_url}/grade", timeout=10)
    r.raise_for_status()
    return r.json().get("score", 0.0)


async def run_task_episode(
    task_id:   str,
    env:       object,   # SupplyChainEnv async client
    base_url:  str,
) -> float:
    """
    Run one full episode for task_id.
    Emits [START] ... [STEP] x N ... [END] to stdout.
    Returns final score in [0, 1].
    """
    max_steps   = TASK_MAX_STEPS[task_id]
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset episode
        obs_raw = _http_reset(base_url, task_id, SEED)
        obs     = obs_raw  # already a dict

        for step in range(1, max_steps + 1):
            # Get LLM action
            action_dict, action_str = get_llm_action(obs, step, task_id)

            # Step environment
            error_msg = None
            try:
                result_raw = _http_step(base_url, action_dict)
                reward     = float(result_raw.get("reward", 0.0))
                done       = bool(result_raw.get("done", False))
                obs        = result_raw.get("observation", obs)
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_msg = str(exc)[:120]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done or error_msg:
                break

        # Grade
        try:
            raw_score = _http_grade(base_url)
        except Exception:
            raw_score = 0.0

        score   = float(max(0.0, min(1.0, raw_score)))   # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Main — async, 3 task blocks
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Run all 3 tasks. Each task gets its own [START]...[END] block.
    Uses from_docker_image() if LOCAL_IMAGE_NAME is set,
    otherwise connects to ENV_BASE_URL directly.
    """
    # Initialize model (API or local)
    initialize_model()

    # Import here to avoid circular issues at module level
    from client import SupplyChainEnv

    env        = None
    base_url   = ENV_BASE_URL
    all_scores: Dict[str, float] = {}

    try:
        # ── Spin up Docker container if image name provided ────────────
        if LOCAL_IMAGE_NAME:
            print(
                f"[DEBUG] Starting container from image: {LOCAL_IMAGE_NAME}",
                flush=True,
            )
            env      = await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)
            base_url = env.base_url
            print(f"[DEBUG] Container ready at {base_url}", flush=True)
        else:
            # Connect to already-running server
            print(
                f"[DEBUG] Connecting to existing server at {base_url}",
                flush=True,
            )
            env = SupplyChainEnv(base_url=base_url)
            # Quick health check
            try:
                import requests as _req
                _req.get(f"{base_url}/", timeout=10).raise_for_status()
            except Exception as e:
                print(
                    f"[DEBUG] Server health check failed: {e}\n"
                    f"  Start the server first:\n"
                    f"  uvicorn server.app:app --host 0.0.0.0 --port 8000",
                    flush=True,
                )
                sys.exit(1)

        # ── Run 3 tasks, each with its own [START]...[END] block ──────
        for task_id in ["task_0", "task_1", "task_2"]:
            score = await run_task_episode(
                task_id=task_id,
                env=env,
                base_url=base_url,
            )
            all_scores[task_id] = score
            # Brief pause between tasks to avoid rate-limit on LLM API
            await asyncio.sleep(1.0)

    finally:
        # ── Always close the env (stops Docker container if started) ──
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)

    # ── Summary (to stderr so it doesn't interfere with stdout parsing) ──
    print("\n=== Inference Complete ===", file=sys.stderr)
    for tid, sc in all_scores.items():
        print(f"  {tid}: {sc:.4f}", file=sys.stderr)
    if all_scores:
        agg = sum(all_scores.values()) / len(all_scores)
        print(f"  Aggregate: {agg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
