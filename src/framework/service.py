"""service.py -- Step 8. Edge service: gateway + health monitoring + controller loop.

Layers of the paper's architecture, in one place:
  DAL  : the replayed request stream (windows) enters _serve()
  HML  : rolling latency / reliability windows + utilisation monitor
  FDL  : _hard() flags and health_score() -> Z
  DSC  : SwitchController (state_machine.py), _maintain() restarts workers
  IEL  : the two worker processes (workers.py)

Closed-loop client: one request in flight at a time (documented limitation).
"""
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from src.framework.health_score import HealthConfig, health_score
from src.framework.monitor import UtilisationMonitor
from src.framework.state_machine import Obs, State, SwitchConfig, SwitchController
from src.framework.workers import WorkerHandle


@dataclass(frozen=True)
class RuntimeConfig:
    rate_hz: float
    dt: float
    tau_w: float
    T_hb: float
    T_req: float
    restart_cooldown: float
    wd_consecutive: int

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(open(path))["runtime"])


def _stats(win):
    """(p95 latency, mean reliability) over a window of (t, lat_ms, c); None if empty."""
    if not win:
        return None, None
    lat = [w[1] for w in win]
    cs = [w[2] for w in win if w[2] is not None]
    return float(np.percentile(lat, 95)), (float(np.mean(cs)) if cs else None)


class EdgeService:
    def __init__(self, health: HealthConfig, sw: SwitchConfig, rt: RuntimeConfig, scaler_std: float):
        self.h, self.sw, self.rt, self.std = health, sw, rt, scaler_std
        self.P = WorkerHandle("primary")
        self.F = WorkerHandle("fallback")
        self.ctl = SwitchController(health, sw)
        self.mon = UtilisationMonitor()
        self.win_live = deque()          # live Primary requests: (t, latency_ms, c)
        self.win_probe = deque()         # canary probes of Primary
        self.tP = 0                      # consecutive live-request timeouts (watchdog)
        self.tF = 0
        self.last_restart = {"primary": -1e9, "fallback": -1e9}
        self.input_noise_kw = 0.0        # F3 fault hook: Gaussian noise on the load channel
        self._rng = np.random.default_rng(0)
        self._last_window = None
        self.requests, self.ticks, self.events_log = [], [], []
        self.t0 = None

    # ------------------------------------------------------------ startup / shutdown
    def wait_ready(self, timeout=90.0):
        t = time.time()
        while not (self.P.ready.value and self.F.ready.value):
            if time.time() - t > timeout:
                raise RuntimeError("workers did not become ready")
            time.sleep(0.1)
        time.sleep(0.5)
        self.mon.reset_baseline()

    def shutdown(self):
        for w in (self.P, self.F):
            w.kill()

    # ------------------------------------------------------------ health / faults
    def _hard(self, w, timeouts):
        """Hard-fault flag: process dead, heartbeat too old, or watchdog (repeated timeouts)."""
        if not w.alive():
            return True
        if w.ready.value and time.time() - w.hb.value > self.rt.T_hb:
            return True
        return timeouts >= self.rt.wd_consecutive

    def _live(self, w):
        return bool(w.alive() and w.ready.value and time.time() - w.hb.value <= self.rt.T_hb)

    def _needs_restart(self, w):
        return (not w.alive()) or bool(w.ready.value and time.time() - w.hb.value > self.rt.T_hb)

    def _restart(self, name, now):
        if now - self.last_restart[name] < self.rt.restart_cooldown:
            return
        self.last_restart[name] = now
        if name == "primary":
            self.P.restart()
            self.tP = 0
            self.win_probe.clear()
        else:
            self.F.restart()
            self.tF = 0
        self.events_log.append((now, f"restart_{name}"))

    def _maintain(self, now):
        if self.ctl.state in (State.FALLBACK, State.FAILSAFE) and self._needs_restart(self.P):
            self._restart("primary", now)
        if self._needs_restart(self.F):
            self._restart("fallback", now)

    # ------------------------------------------------------------ control tick
    def _probe(self, now):
        r = self.P.request(self._last_window, True, self.rt.T_req)
        if r is None:                                   # failed probe counts as very bad
            self.win_probe.append((now, self.rt.T_req * 1000.0, 0.0))
        else:
            self.win_probe.append((now, r[2], r[1]))

    def _trim(self, win, now):
        while win and win[0][0] < now - self.rt.tau_w:
            win.popleft()

    def _on_transition(self, tr):
        if tr.dst is State.PRIMARY:
            self.win_live.clear()
            self.tP = 0
        if tr.dst is State.FALLBACK:
            self.win_probe.clear()
        if tr.dst is State.FALLBACK and tr.src is State.FAILSAFE:
            self.tF = 0

    def _tick(self, now):
        U = self.mon.sample()
        self._trim(self.win_live, now)
        self._trim(self.win_probe, now)
        hP, hF = self._hard(self.P, self.tP), self._hard(self.F, self.tF)
        P_live, F_live = self._live(self.P), self._live(self.F)

        Z = Zhat = L = C = None
        if self.ctl.state is State.PRIMARY:
            L, C = _stats(self.win_live)
            Z = health_score(self.h, L, U, C)[0]
        if self._last_window is not None and self.ctl.probe_allowed(Obs(t=now, P_live=P_live, U=U)):
            self._probe(now)
            L, C = _stats(self.win_probe)
            Zhat = health_score(self.h, L, U, C)[0]

        obs = Obs(t=now, h_P=hP, h_F=hF, Z=Z, Z_hat=Zhat,
                  P_live=P_live, F_live=F_live, U=U)
        tr = self.ctl.step(obs)
        if tr:
            self._on_transition(tr)
        self._maintain(now)
        self.ticks.append(dict(t=now, state=self.ctl.state.value, Z=Z, Z_hat=Zhat,
                               L_p95=L, C=C, U=U, h_P=int(hP), h_F=int(hF)))

    # ------------------------------------------------------------ serving
    def _serve(self, i, window, y, now):
        if self.input_noise_kw > 0:                     # F3: noisy sensor input
            window = window.copy()
            window[:, 0] += self._rng.normal(0, self.input_noise_kw / self.std, len(window)).astype("float32")
        self._last_window = window
        who = self.ctl.serving
        rec = dict(t=now, i=i, state=self.ctl.state.value, served=who,
                   lat_ms=None, pred=None, y=y, c=None)
        if who is not None:
            w = self.P if who == "primary" else self.F
            r = w.request(window, False, self.rt.T_req)
            if r is None:
                if who == "primary":
                    self.tP += 1
                else:
                    self.tF += 1
            else:
                pred, c, lat = r
                rec.update(lat_ms=lat, pred=pred, c=c)
                if who == "primary":
                    self.tP = 0
                    self.win_live.append((now, lat, c))
                else:
                    self.tF = 0
        self.requests.append(rec)

    # ------------------------------------------------------------ main loop
    def run(self, X, y, duration, events=()):
        """events: list of (t_seconds, fn(service)) executed once when t is reached."""
        self.wait_ready()
        events = sorted(events, key=lambda e: e[0])
        period = 1.0 / self.rt.rate_hz
        self.t0 = time.perf_counter()
        next_tick, i = 0.0, 0
        while True:
            now = time.perf_counter() - self.t0
            if now >= duration:
                break
            while events and events[0][0] <= now:
                events.pop(0)[1](self)
                self.events_log.append((now, "fault_event"))
            if now >= next_tick:
                self._tick(now)
                next_tick += self.rt.dt
            k = i % len(X)
            self._serve(i, X[k], float(y[k]), now)
            i += 1
            wait = self.t0 + i * period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

    # ------------------------------------------------------------ output
    def save(self, name, out_dir="results"):
        Path(out_dir).mkdir(exist_ok=True)

        def dump(rows, suffix):
            if not rows:
                return
            with open(Path(out_dir) / f"{name}_{suffix}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)

        dump(self.requests, "requests")
        dump(self.ticks, "ticks")
        dump([dict(t=t.t, src=t.src.value, dst=t.dst.value, reason=t.reason, score=t.score)
              for t in self.ctl.log], "transitions")
        dump([dict(t=t, event=e) for t, e in self.events_log], "events")