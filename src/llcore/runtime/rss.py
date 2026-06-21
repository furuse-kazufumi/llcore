# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for **process RSS / working-set** measurement.

Several memory harnesses (``scripts/recurrent_runtime_rss.py`` = peak working set vs
context length, ``scripts/runtime_floor_rss.py`` = static runtime RSS floor) need the
same fiddly, error-prone WinAPI call. Duplicating the ``ctypes`` struct + ``GetProcessMemoryInfo``
plumbing across scripts risks the copies drifting; this module centralizes it.

Two readings are exposed:

- :func:`peak_working_set_bytes` — ``PeakWorkingSetSize``: the *maximum* working set over the
  process lifetime. Use when isolating each measurement in its own subprocess (process-lifetime peak).
- :func:`working_set_bytes` — ``WorkingSetSize``: the working set *right now*. Use for staged
  point-in-time floor measurements.

Both prefer the Windows WinAPI; :func:`working_set_bytes` falls back to ``/proc/self/statm`` on
Linux. Any failure returns ``0`` (RSS is an auxiliary signal — never crash a benchmark over it).
"""
from __future__ import annotations

import ctypes


class _PMC(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS — layout per the Win32 psapi header."""

    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _process_memory_counters() -> _PMC | None:
    """Return a populated ``PROCESS_MEMORY_COUNTERS`` via WinAPI, or ``None`` off-Windows/on failure."""
    try:
        import ctypes.wintypes as wt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters if ok else None
    except Exception:  # noqa: BLE001 - non-Windows / lookup failure: caller handles None
        return None


def peak_working_set_bytes() -> int:
    """Peak working set (``PeakWorkingSetSize``) in bytes; ``0`` if unavailable."""
    counters = _process_memory_counters()
    return int(counters.PeakWorkingSetSize) if counters is not None else 0


def working_set_bytes() -> int:
    """Current working set (``WorkingSetSize``) in bytes; ``0`` if unavailable.

    Falls back to ``/proc/self/statm`` (RSS pages x 4096) on Linux.
    """
    counters = _process_memory_counters()
    if counters is not None:
        return int(counters.WorkingSetSize)
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            rss_pages = int(fh.read().split()[1])
        return rss_pages * 4096
    except Exception:  # noqa: BLE001 - RSS is auxiliary; 0 keeps benchmarks running
        return 0
