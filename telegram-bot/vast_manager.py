"""
Vast.ai instance lifecycle.

Endpoints follow the official CLI (github.com/vast-ai/vast-cli):
    search   PUT    /api/v0/search/asks/      {"select_cols": ["*"], "q": {...}}
    create   PUT    /api/v0/asks/{offer_id}/  {"client_id": "me", "image": ..., ...}
    list     GET    /api/v0/instances/
    destroy  DELETE /api/v0/instances/{id}/

The single most important property of this module is that a rented instance is
always destroyed. Every path out of `rent()` — success, exception, cancellation,
timeout — runs the teardown, and `reap_orphans()` catches anything a hard crash
left behind.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
    "-o", "ConnectTimeout=15",
]


class VastError(RuntimeError):
    """Any failure talking to Vast.ai."""


@dataclass
class Offer:
    offer_id: int
    gpu_name: str
    gpu_ram_gb: float
    dph: float
    cuda_version: str | None = None

    def __str__(self) -> str:
        return f"{self.gpu_name} ({self.gpu_ram_gb:.0f} GB) @ ${self.dph:.3f}/hr"


@dataclass
class Instance:
    instance_id: int
    ssh_host: str | None = None
    ssh_port: int | None = None
    status: str = "unknown"

    @property
    def ready(self) -> bool:
        return bool(self.ssh_host and self.ssh_port and self.status == "running")


class VastManager:
    def __init__(self, api_key: str, api_base: str, ssh_key: Path, label: str = "i2v-bot"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.ssh_key = Path(ssh_key)
        self.label = label

    # -- HTTP ---------------------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _request(self, method: str, path: str, **kw) -> dict:
        url = f"{self.api_base}{path}"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.request(method, url, headers=self._headers(), **kw)
        if r.status_code >= 400:
            raise VastError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            raise VastError(f"{method} {path} returned non-JSON: {r.text[:200]}")

    # -- Account ------------------------------------------------------------

    async def ping(self) -> dict:
        """Verify the API key works. Used by verify.py."""
        return await self._request("GET", "/instances/")

    # -- Offers -------------------------------------------------------------

    async def find_cheapest(self, query: dict, gpu_names: tuple[str, ...] = ()) -> Offer:
        """
        Return the cheapest rentable offer satisfying `query`.

        gpu_names is applied client-side: Vast's `gpu_name` filter takes a
        single value, so querying several and picking the minimum is simpler
        and cheaper than N round trips.
        """
        payload = {"select_cols": ["*"], "q": dict(query, type="on-demand", order=[["dph_total", "asc"]])}
        data = await self._request("PUT", "/search/asks/", json=payload)
        offers = data.get("offers") or []
        if not offers:
            raise VastError(
                "No Vast.ai offers matched. Try raising VAST_MAX_DPH or lowering VAST_MIN_VRAM_GB."
            )

        wanted = {n.upper() for n in gpu_names}
        candidates = []
        for o in offers:
            name = str(o.get("gpu_name", "")).replace(" ", "_").upper()
            if wanted and not any(w in name or name in w for w in wanted):
                continue
            candidates.append(
                Offer(
                    offer_id=int(o["id"]),
                    gpu_name=str(o.get("gpu_name", "?")),
                    gpu_ram_gb=float(o.get("gpu_ram", 0)) / 1024,
                    dph=float(o.get("dph_total", 0)),
                    cuda_version=str(o.get("cuda_max_good", "")) or None,
                )
            )

        if not candidates:
            raise VastError(
                f"Offers exist but none matched VAST_GPU_NAMES={','.join(gpu_names)}. "
                "Widen that list or clear it to accept any GPU."
            )
        return min(candidates, key=lambda c: c.dph)

    # -- Instances ----------------------------------------------------------

    async def create(self, offer: Offer, image: str, disk_gb: int, public_key: str) -> int:
        onstart = (
            "mkdir -p ~/.ssh && "
            'echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && '
            "chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
        )
        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": self.label,
            "runtype": "ssh",
            "onstart": onstart,
            "env": {"PUBLIC_KEY": public_key},
        }
        data = await self._request("PUT", f"/asks/{offer.offer_id}/", json=payload)
        instance_id = data.get("new_contract")
        if not instance_id:
            raise VastError(f"Create returned no instance id: {data}")
        log.info("Created Vast.ai instance %s on %s", instance_id, offer)
        return int(instance_id)

    async def get(self, instance_id: int) -> Instance:
        data = await self._request("GET", "/instances/")
        for row in data.get("instances", []):
            if int(row.get("id", -1)) == instance_id:
                port = row.get("ssh_port")
                return Instance(
                    instance_id=instance_id,
                    ssh_host=row.get("ssh_host"),
                    ssh_port=int(port) if port else None,
                    status=str(row.get("actual_status") or row.get("cur_state") or "unknown"),
                )
        return Instance(instance_id=instance_id, status="missing")

    async def destroy(self, instance_id: int) -> bool:
        """Destroy an instance. Never raises — teardown must not mask a job error."""
        try:
            await self._request("DELETE", f"/instances/{instance_id}/", json={})
            log.info("Destroyed Vast.ai instance %s", instance_id)
            return True
        except Exception as exc:
            log.error("FAILED to destroy instance %s (%s) — destroy it manually at "
                      "https://console.vast.ai/instances/ or you will keep paying", instance_id, exc)
            return False

    async def reap_orphans(self) -> list[int]:
        """
        Destroy every instance carrying our label.

        Called at startup: if the bot was killed mid-job, an instance may still
        be running and billing. This is the backstop for that.
        """
        try:
            data = await self._request("GET", "/instances/")
        except Exception as exc:
            log.warning("Could not list instances to reap orphans: %s", exc)
            return []

        reaped = []
        for row in data.get("instances", []):
            if str(row.get("label") or "") == self.label:
                iid = int(row["id"])
                if await self.destroy(iid):
                    reaped.append(iid)
        if reaped:
            log.warning("Reaped %d orphaned instance(s) from a previous run: %s", len(reaped), reaped)
        return reaped

    async def wait_until_ready(self, instance_id: int, timeout: int, on_status=None) -> Instance:
        deadline = asyncio.get_event_loop().time() + timeout
        last = None
        while asyncio.get_event_loop().time() < deadline:
            inst = await self.get(instance_id)
            if inst.status != last:
                last = inst.status
                if on_status:
                    await on_status(f"Instance {inst.status}...")
            if inst.ready:
                return inst
            if inst.status in ("missing", "exited"):
                raise VastError(f"Instance {instance_id} died before becoming ready ({inst.status}).")
            await asyncio.sleep(10)
        raise VastError(f"Instance {instance_id} not ready within {timeout}s.")

    # -- SSH ----------------------------------------------------------------

    def _ssh_base(self, inst: Instance) -> list[str]:
        return ["ssh", *SSH_OPTS, "-i", str(self.ssh_key), "-p", str(inst.ssh_port),
                f"root@{inst.ssh_host}"]

    async def _run(self, argv: list[str], timeout: int) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise VastError(f"Command timed out after {timeout}s: {argv[0]}")
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

    async def wait_for_ssh(self, inst: Instance, timeout: int = 420) -> None:
        """Poll until sshd accepts our key. Fresh containers take a while."""
        deadline = asyncio.get_event_loop().time() + timeout
        last_err = ""
        while asyncio.get_event_loop().time() < deadline:
            code, _, err = await self._run([*self._ssh_base(inst), "true"], timeout=30)
            if code == 0:
                return
            last_err = err.strip()
            await asyncio.sleep(10)
        raise VastError(f"SSH never came up on {inst.ssh_host}:{inst.ssh_port}. Last error: {last_err[:200]}")

    async def exec(self, inst: Instance, command: str, timeout: int) -> str:
        code, out, err = await self._run([*self._ssh_base(inst), command], timeout=timeout)
        if code != 0:
            raise VastError(f"Remote command failed ({code}): {err.strip()[:500] or out.strip()[:500]}")
        return out

    async def upload(self, inst: Instance, local: Path, remote: str, timeout: int = 900) -> None:
        argv = ["scp", *SSH_OPTS, "-i", str(self.ssh_key), "-P", str(inst.ssh_port),
                str(local), f"root@{inst.ssh_host}:{remote}"]
        code, _, err = await self._run(argv, timeout=timeout)
        if code != 0:
            raise VastError(f"Upload of {local.name} failed: {err.strip()[:300]}")

    async def download(self, inst: Instance, remote: str, local: Path, timeout: int = 1800) -> None:
        argv = ["scp", *SSH_OPTS, "-i", str(self.ssh_key), "-P", str(inst.ssh_port),
                f"root@{inst.ssh_host}:{remote}", str(local)]
        code, _, err = await self._run(argv, timeout=timeout)
        if code != 0:
            raise VastError(f"Download of {remote} failed: {err.strip()[:300]}")

    # -- Guaranteed-teardown rental ----------------------------------------

    @contextlib.asynccontextmanager
    async def rent(self, query: dict, gpu_names, image: str, disk_gb: int,
                   public_key: str, boot_timeout: int, on_status=None):
        """
        Rent the cheapest matching GPU, yield (instance, offer), destroy on exit.

        The destroy runs in a finally block, so it happens on success, on
        exception, and on task cancellation alike.
        """
        async def status(msg: str) -> None:
            log.info(msg)
            if on_status:
                await on_status(msg)

        await status("Searching Vast.ai for the cheapest suitable GPU...")
        offer = await self.find_cheapest(query, gpu_names)
        await status(f"Renting {offer}")

        instance_id = await self.create(offer, image, disk_gb, public_key)
        try:
            inst = await self.wait_until_ready(instance_id, boot_timeout, on_status=status)
            await status("Waiting for SSH...")
            await self.wait_for_ssh(inst)
            yield inst, offer
        finally:
            await status("Destroying GPU instance...")
            await self.destroy(instance_id)
