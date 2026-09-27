"""Admission-only bounds for GLM's two native cache owners.

Physical headroom is a conditional non-file-backed budget ceiling, not a
promise that every file-backed page is immediately free. No decode polling.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import cast


@dataclass(frozen=True)
class MemoryLimits:
    cache_bytes: int
    physical_margin_bytes: int

    @classmethod
    def configured(cls) -> "MemoryLimits | None":
        capacity_raw = os.getenv("EXO_COOPERATIVE_CACHE_BUDGET_BYTES")
        margin_raw = os.getenv("EXO_COOPERATIVE_PHYSICAL_MARGIN_BYTES")
        if capacity_raw is None and margin_raw is None:
            return None
        try:
            capacity = int(capacity_raw or "")
            margin = int(margin_raw or "")
            if capacity <= 0 or margin <= 0:
                raise ValueError("nonpositive memory bound")
            return cls(capacity, margin)
        except ValueError as error:
            raise ValueError("invalid_cooperative_memory_configuration") from error


def reservation_bytes(prompt_tokens: int, output_tokens: int) -> int:
    if prompt_tokens < 0 or output_tokens <= 0:
        raise ValueError("unbounded_native_output")
    # 256-row allocation floors, INT8 MLA/indexer/MTP and scalar metadata.
    # MTP draft/transient allocations belong to the explicit physical margin.
    return ((prompt_tokens + output_tokens + 255) // 256) * 256 * 65536


@dataclass(frozen=True)
class PhysicalSnapshot:
    captured: float = 0
    total: int = 0
    anonymous: int = 0
    wired: int = 0
    compressor: int = 0
    free: int = 0
    speculative: int = 0
    file_backed: int = 0
    pressure: int = 0
    swap_bytes: int = 0
    swapouts: int = 0
    valid: bool = False

    @property
    def headroom(self) -> int:
        return max(
            0,
            min(
                self.total - self.anonymous - self.wired - self.compressor,
                self.free + self.speculative + self.file_backed,
            ),
        )

    def healthy(self, baseline: "PhysicalSnapshot", now: float) -> bool:
        return (
            self.valid
            and baseline.valid
            and baseline.pressure == 1
            and 0 <= now - self.captured <= 5
            and self.total == baseline.total
            and self.pressure == 1
            and self.swap_bytes <= baseline.swap_bytes
            and self.swapouts <= baseline.swapouts
        )


def parse_physical(
    vm_stat: str, total: str, pressure: str, swap: str, captured: float
) -> PhysicalSnapshot:
    page = re.search(r"page size of (\d+) bytes", vm_stat)
    if page is None:
        raise ValueError("unknown_vm_stat_page_size")
    page_size = int(cast(str, page.group(1)))
    counters = {
        name: int(value)
        for name, value in cast(
            list[tuple[str, str]],
            re.findall(r"^([^:\n]+):\s*(\d+)\.", vm_stat, re.MULTILINE),
        )
    }
    used = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
    if used is None:
        raise ValueError("unknown_swap_usage")
    used_bytes = int(
        cast(
            Decimal,
            Decimal(cast(str, used.group(1)))
            * 1024 ** ("KMG".index(cast(str, used.group(2))) + 1),
        )
    )
    return PhysicalSnapshot(
        captured=captured,
        total=int(total),
        anonymous=counters["Anonymous pages"] * page_size,
        wired=counters["Pages wired down"] * page_size,
        compressor=counters["Pages occupied by compressor"] * page_size,
        free=counters["Pages free"] * page_size,
        speculative=counters["Pages speculative"] * page_size,
        file_backed=counters["File-backed pages"] * page_size,
        pressure=int(pressure),
        swap_bytes=used_bytes,
        swapouts=counters["Swapouts"],
        valid=True,
    )


def capture_physical() -> PhysicalSnapshot:
    started = time.monotonic()

    def command(arguments: list[str]) -> str:
        remaining = 2 - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(arguments, 2)
        return subprocess.check_output(arguments, text=True, timeout=remaining)

    try:
        vm_stat = command(["/usr/bin/vm_stat"])
        total = command(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        pressure = command(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
        )
        swap = command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
        return parse_physical(vm_stat, total, pressure, swap, started)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, ArithmeticError):
        # All ranks still enter the same collective with an unsafe row.
        return PhysicalSnapshot()


def projected_cache(
    plans: list[int], allocated: list[int], chosen: int, requested: int
) -> tuple[int, int]:
    """Return full resident projection and incremental growth.

    Existing allocations are already in the host physical ledger. Credit only
    allocator-authoritative, evaluated, nonoverlapping owner buffer sets.
    """
    future = [
        max(plan - actual, 0) for plan, actual in zip(plans, allocated, strict=True)
    ]
    future[chosen] = max(requested - allocated[chosen], 0)
    growth = sum(future)
    return sum(allocated) + growth, growth
