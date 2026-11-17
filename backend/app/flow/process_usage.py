"""Sample the main process's resident memory during a flow evaluation."""
import ctypes
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter, process_time


class WorkingSet:
    INTERVAL_SECONDS = .1

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def __init__(self):
        self.value = None
        self.stop = Event()
        self.thread = Thread(target=self._sample, daemon=True)

    @classmethod
    def current(cls):
        if sys.platform != "win32":
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            raise OSError("No se pudo medir RSS en este sistema.")
        counters = cls._Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(cls._Counters), ctypes.c_ulong]
        get_info.restype = ctypes.c_int
        if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)

    def _record(self):
        try:
            value = self.current()
        except (OSError, ValueError):
            return
        self.value = max(self.value or 0, value)

    def _sample(self):
        while not self.stop.wait(self.INTERVAL_SECONDS):
            self._record()

    def __enter__(self):
        self._record()
        if self.value is not None:
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self._record()


class ProcessUsage:
    def __enter__(self):
        self.memory = WorkingSet()
        self.wall_start = perf_counter()
        self.cpu_start = process_time()
        self.memory.__enter__()
        return self

    def __exit__(self, *exc):
        cpu = process_time() - self.cpu_start
        wall = perf_counter() - self.wall_start
        self.memory.__exit__(*exc)
        self.measurements = {
            "cpu_seconds": round(cpu, 6),
            "cpu_average_percent": round(100 * cpu / wall, 2) if wall > 0 else None,
            "peak_rss_bytes": self.memory.value,
            "rss_interval_ms": round(self.memory.INTERVAL_SECONDS * 1000),
        }
