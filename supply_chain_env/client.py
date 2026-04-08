"""
client.py — Supply Chain Disruption Management EnvClient
=========================================================
Full OpenEnv-compatible async client with:
  - from_docker_image()  classmethod to spin up env from local Docker image
  - close()              to stop the container cleanly
  - async reset() / step() / state() / grade()
  - .sync()              wrapper for synchronous usage

Usage (async, from docker image):
    env = await SupplyChainEnv.from_docker_image(image_name="supply-chain-env")
    try:
        result = await env.reset(task_id="task_1")
        result = await env.step(SupplyChainAction(...))
        state  = await env.state()
    finally:
        await env.close()

Usage (async, existing server):
    async with SupplyChainEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(SupplyChainAction(...))

Usage (sync):
    with SupplyChainEnv(base_url="http://localhost:8000").sync() as env:
        obs    = env.reset()
        result = env.step(SupplyChainAction(...))
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import os
from typing import Optional

import requests

from models import (
    StepResult,
    SupplyChainAction,
    SupplyChainObservation,
    SupplyChainState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT      = 8000
CONTAINER_STARTUP_TIMEOUT = 60   # seconds to wait for container to be healthy
CONTAINER_POLL_INTERVAL   = 2    # seconds between health-check polls


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class SupplyChainEnv:
    """
    Async OpenEnv client for the Supply Chain Disruption Management environment.

    Can connect to an already-running server OR spin up a local Docker container.
    """

    def __init__(self, base_url: str = f"http://localhost:{DEFAULT_PORT}"):
        self.base_url        = base_url.rstrip("/")
        self._container_id:  Optional[str] = None   # set by from_docker_image()
        self._container_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SupplyChainEnv":
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ------------------------------------------------------------------
    # Factory — spin up from local Docker image
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: int = DEFAULT_PORT,
        extra_env: Optional[dict] = None,
    ) -> "SupplyChainEnv":
        """
        Start a local Docker container from `image_name` and return a connected client.

        Parameters
        ----------
        image_name  : Docker image name (e.g. "supply-chain-env" or from LOCAL_IMAGE_NAME env var)
        port        : Host port to bind (default 8000)
        extra_env   : Additional environment variables to pass to the container

        Raises
        ------
        RuntimeError  if the container fails to start or become healthy within timeout
        """
        if not image_name:
            raise ValueError(
                "image_name must be provided (or set LOCAL_IMAGE_NAME env var)"
            )

        container_name = f"supply-chain-env-{int(time.time())}"

        # Build docker run command
        cmd = [
            "docker", "run",
            "--detach",
            "--name", container_name,
            "-p", f"{port}:{DEFAULT_PORT}",
            "--memory", "8g",
            "--cpus", "2",
        ]

        # Pass through env vars
        env_vars = {
            "API_BASE_URL": os.getenv("API_BASE_URL", ""),
            "MODEL_NAME":   os.getenv("MODEL_NAME", ""),
            "HF_TOKEN":     os.getenv("HF_TOKEN", ""),
        }
        if extra_env:
            env_vars.update(extra_env)
        for k, v in env_vars.items():
            if v:
                cmd += ["-e", f"{k}={v}"]

        cmd.append(image_name)

        # Start container
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is not installed or not on PATH. "
                "Install Docker: https://docs.docker.com/get-docker/"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("docker run timed out after 30s")

        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed:\n{result.stderr.strip()}"
            )

        container_id = result.stdout.strip()
        base_url     = f"http://localhost:{port}"

        instance                  = cls(base_url=base_url)
        instance._container_id   = container_id
        instance._container_name = container_name

        # Wait for the server to become healthy
        await instance._wait_until_healthy()
        return instance

    async def _wait_until_healthy(self):
        """Poll / until the server responds with HTTP 200."""
        deadline = time.time() + CONTAINER_STARTUP_TIMEOUT
        last_error = None
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.base_url}/", timeout=5)
                if resp.status_code == 200:
                    return   # healthy!
            except Exception as e:
                last_error = e
            await asyncio.sleep(CONTAINER_POLL_INTERVAL)

        # Timed out — collect container logs for debugging
        logs = ""
        if self._container_name:
            try:
                r = subprocess.run(
                    ["docker", "logs", "--tail", "30", self._container_name],
                    capture_output=True, text=True, timeout=10,
                )
                logs = r.stdout + r.stderr
            except Exception:
                pass

        raise RuntimeError(
            f"Container did not become healthy within {CONTAINER_STARTUP_TIMEOUT}s. "
            f"Last error: {last_error}\nContainer logs:\n{logs}"
        )

    # ------------------------------------------------------------------
    # close() — stop and remove the Docker container
    # ------------------------------------------------------------------

    async def close(self):
        """
        Stop and remove the Docker container (if started via from_docker_image).
        Safe to call even if no container was started.
        """
        if not self._container_name:
            return
        try:
            subprocess.run(
                ["docker", "stop", self._container_name],
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["docker", "rm", self._container_name],
                capture_output=True, timeout=10,
            )
        except Exception as e:
            # Non-fatal — container may have already exited
            print(f"[DEBUG] Container cleanup warning: {e}", flush=True)
        finally:
            self._container_id   = None
            self._container_name = None

    # ------------------------------------------------------------------
    # sync() wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "SupplyChainEnvSync":
        """Return a synchronous wrapper around this client."""
        return SupplyChainEnvSync(base_url=self.base_url)

    # ------------------------------------------------------------------
    # OpenEnv interface — async
    # ------------------------------------------------------------------

    async def reset(
        self,
        task_id: str = "task_0",
        seed:    int = 42,
    ) -> StepResult:
        """
        Reset the environment. Returns a StepResult-like object.
        Observation is in result.observation.
        """
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "seed": seed},
                timeout=30,
            ),
        )
        resp.raise_for_status()
        obs = SupplyChainObservation(**resp.json())
        # Wrap in StepResult so callers can use result.observation consistently
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    async def step(self, action: SupplyChainAction) -> StepResult:
        """Execute one action. Returns StepResult(observation, reward, done, info)."""
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{self.base_url}/step",
                json=action.model_dump(),
                timeout=30,
            ),
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    async def state(self) -> SupplyChainState:
        """Return current episode metadata."""
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{self.base_url}/state", timeout=10),
        )
        resp.raise_for_status()
        return SupplyChainState(**resp.json())

    async def grade(self) -> float:
        """Return grader score for current episode [0.0–1.0]."""
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(f"{self.base_url}/grade", timeout=10),
        )
        resp.raise_for_status()
        return resp.json().get("score", 0.0)


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------

class SupplyChainEnvSync:
    """
    Synchronous HTTP client — thin wrapper for non-async usage.
    Use via SupplyChainEnv(...).sync() or directly.
    """

    def __init__(self, base_url: str = f"http://localhost:{DEFAULT_PORT}"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def close(self):
        self._session.close()

    def reset(self, task_id: str = "task_0", seed: int = 42) -> SupplyChainObservation:
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        return SupplyChainObservation(**resp.json())

    def step(self, action: SupplyChainAction) -> StepResult:
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> SupplyChainState:
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return SupplyChainState(**resp.json())

    def grade(self) -> float:
        resp = self._session.post(f"{self.base_url}/grade", timeout=10)
        resp.raise_for_status()
        return resp.json().get("score", 0.0)

    def health(self) -> dict:
        resp = self._session.get(f"{self.base_url}/", timeout=10)
        resp.raise_for_status()
        return resp.json()
