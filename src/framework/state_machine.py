"""state_machine.py -- Step 7. Four-state switching controller (paper Section VII).

PRIMARY -> FALLBACK -> RECOVERY -> PRIMARY, plus FAILSAFE.
Pure logic: no threads, no real clock. Time `t` is passed in with every
observation, so the controller is deterministic and easy to unit-test.

Phase 1 scope: hysteresis, debounce, dwell time, recovery probing.
Backoff (D doubling) is left for Phase 2, so D is constant here (D == D_min).
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import yaml
from src.framework.health_score import HealthConfig


class State(Enum):
    PRIMARY = "PRIMARY"
    FALLBACK = "FALLBACK"
    RECOVERY = "RECOVERY"
    FAILSAFE = "FAILSAFE"


@dataclass(frozen=True)
class SwitchConfig:
    N_w: int          # debounce: consecutive samples with Z >= Z_switch
    N_rec: int        # consecutive good probes needed to return to PRIMARY
    D: float          # minimum dwell time after any transition (s)
    T_rec: float      # recovery attempt timeout (s)

    def __post_init__(self):
        assert self.N_w >= 1 and self.N_rec >= 1 and self.D >= 0 and self.T_rec > 0

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(open(path))["switching"])


@dataclass
class Obs:
    """Everything the controller sees at one sampling instant."""
    t: float
    h_P: bool = False                 # Primary hard-fault flag (heartbeat/watchdog)
    h_F: bool = False                 # Fallback hard-fault flag
    Z: Optional[float] = None         # health score from live Primary traffic
    Z_hat: Optional[float] = None     # health score from canary probes of Primary
    P_live: bool = False              # Primary process is running (can be probed)
    F_live: bool = True               # Fallback process is running
    U: float = 0.0                    # smoothed utilisation (gates probing)


@dataclass(frozen=True)
class Transition:
    t: float
    src: State
    dst: State
    reason: str
    score: Optional[float]


class SwitchController:
    def __init__(self, health: HealthConfig, sw: SwitchConfig, t0: float = 0.0):
        self.h, self.sw = health, sw
        self.state = State.PRIMARY
        self.c_v = 0                  # consecutive bad samples (debounce counter)
        self.c_r = 0                  # consecutive good probes
        self.t_last = t0              # time of the most recent state change
        self.t_R = None               # time the current recovery attempt started
        self.log: List[Transition] = []

    # ---- what the rest of the system should do ----
    @property
    def serving(self) -> Optional[str]:
        return {State.PRIMARY: "primary", State.FALLBACK: "fallback",
                State.RECOVERY: "fallback", State.FAILSAFE: None}[self.state]

    def probe_allowed(self, o: Obs) -> bool:
        """Canary probes only while Primary is inactive, alive, and the system is not saturated."""
        return (self.state in (State.FALLBACK, State.RECOVERY)
                and o.P_live and o.U < self.h.U_max)

    # ---- one control step (call every sampling period) ----
    def step(self, o: Obs) -> Optional[Transition]:
        dwell_ok = (o.t - self.t_last) >= self.sw.D

        if self.state is State.PRIMARY:
            if o.h_P:                                        # hard fault: immediate, ignores dwell
                return self._go(o, State.FAILSAFE if o.h_F else State.FALLBACK, "hard", o.Z)
            z = 1.0 if o.Z is None else o.Z                  # missing score -> fail-safe
            self.c_v = self.c_v + 1 if z >= self.h.Z_switch else 0
            if self.c_v >= self.sw.N_w and dwell_ok:         # debounce + dwell
                return self._go(o, State.FALLBACK, "soft", z)

        elif self.state is State.FALLBACK:
            if o.h_F:
                return self._go(o, State.FAILSAFE, "hard", None)
            if (o.P_live and dwell_ok and o.U < self.h.U_max
                    and o.Z_hat is not None and o.Z_hat < self.h.Z_recover):
                return self._go(o, State.RECOVERY, "probe", o.Z_hat)

        elif self.state is State.RECOVERY:
            if o.h_F:
                return self._go(o, State.FAILSAFE, "hard", None)
            if o.h_P or (o.Z_hat is not None and o.Z_hat >= self.h.Z_recover) \
                    or (o.t - self.t_R) > self.sw.T_rec:
                return self._go(o, State.FALLBACK, "failed", o.Z_hat)   # one bad probe aborts
            if o.Z_hat is not None:                          # a good probe
                self.c_r += 1
                if self.c_r >= self.sw.N_rec:
                    return self._go(o, State.PRIMARY, "recovered", o.Z_hat)

        else:                                                # FAILSAFE
            if o.F_live and not o.h_F:
                return self._go(o, State.FALLBACK, "restored", None)
        return None

    def _go(self, o: Obs, dst: State, reason: str, score) -> Transition:
        tr = Transition(o.t, self.state, dst, reason, score)
        self.log.append(tr)
        self.state, self.t_last = dst, o.t
        self.c_v, self.c_r = 0, 0
        if dst is State.RECOVERY:
            self.t_R, self.c_r = o.t, 1                      # entry probe counts as the first
        return tr
