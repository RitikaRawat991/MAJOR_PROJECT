"""injectors.py -- Step 9. Fault hooks for EdgeService (simulated edge node).

make_fault(name) returns (on, off): two functions that each take the EdgeService.
Resource faults use separate OS processes, so the load is real (like stress-ng).
"""
import multiprocessing as mp
import time
import psutil


# ---------------- resource stress (runs in separate processes) ----------------
def _cpu_worker(load, stop):
    period = 0.1
    while not stop.is_set():
        busy_until = time.perf_counter() + period * load
        while time.perf_counter() < busy_until:
            pass                                    # busy loop = CPU used
        time.sleep(period * (1 - load))


def _mem_worker(target_mb, stop):
    blocks = []
    for _ in range(target_mb // 10):
        blocks.append(bytearray(b"\x01") * (10 * 1024 * 1024))   # really touches the memory
        time.sleep(0.005)
    while not stop.is_set():
        time.sleep(0.1)


class ResourceStress:
    def __init__(self, kind, load=0.95, n_workers=None, mem_fraction=0.7):
        self.kind, self.load = kind, load
        self.n_workers = n_workers or psutil.cpu_count(logical=True)
        self.mem_fraction = mem_fraction            # of memory that is FREE right now
        self._stop = mp.Event()
        self._procs = []

    def start(self):
        self._stop.clear()
        if self.kind == "cpu":
            for _ in range(self.n_workers):
                p = mp.Process(target=_cpu_worker, args=(self.load, self._stop), daemon=True)
                p.start()
                self._procs.append(p)
        else:
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            p = mp.Process(target=_mem_worker,
                           args=(int(avail_mb * self.mem_fraction), self._stop), daemon=True)
            p.start()
            self._procs.append(p)

    def stop(self):
        self._stop.set()
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        self._procs.clear()


# ---------------- fault factory ----------------
def make_fault(name):
    if name == "kill":                              # permanent crash; the system must restart it
        return (lambda s: s.P.proc.terminate()), (lambda s: None)
    if name == "hang":                              # process alive but frozen; fault ends by itself
        return (lambda s: setattr(s.P.hang, "value", 1)), (lambda s: setattr(s.P.hang, "value", 0))
    if name == "latency":                           # serving-wrapper delay (F2)
        return (lambda s: setattr(s.P.delay_ms, "value", 25.0)), (lambda s: setattr(s.P.delay_ms, "value", 0.0))
    if name == "noise":                             # sensor noise on the load channel (F3)
        return (lambda s: setattr(s, "input_noise_kw", 1.0)), (lambda s: setattr(s, "input_noise_kw", 0.0))
    if name in ("cpu", "memory"):
        st = ResourceStress(name)
        return (lambda s: st.start()), (lambda s: st.stop())
    raise ValueError(f"unknown scenario: {name}")