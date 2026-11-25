"""monitor.py -- utilisation sensor (U_t in the paper)."""
import psutil


class UtilisationMonitor:
    """U_t = EMA-smoothed max(CPU utilisation, memory pressure), in [0, 1].

    Memory pressure is measured against the memory that was FREE when the
    service started (a simulated edge-node memory budget):
        U_mem = 1 - available_now / available_at_baseline
    A laptop usually runs at 70-90% RAM already, so raw "RAM percent" would
    look saturated all the time and could not show a fault.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.u = 0.0
        self._first = True
        self.reset_baseline()

    def reset_baseline(self):
        psutil.cpu_percent(interval=None)          # warm-up call
        self.budget = psutil.virtual_memory().available

    def sample(self) -> float:
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = 1.0 - psutil.virtual_memory().available / self.budget
        raw = min(1.0, max(0.0, max(cpu, mem)))
        self.u = raw if self._first else self.alpha * raw + (1 - self.alpha) * self.u
        self._first = False
        return self.u
