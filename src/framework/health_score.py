"""health_score.py -- Step 6. Health score Z_t (paper Eq. 1-4)."""
from dataclasses import dataclass
import yaml


@dataclass(frozen=True)
class HealthConfig:
    L_slo: float
    U_max: float
    C_target: float
    w_L: float
    w_U: float
    w_C: float
    rho_max: float
    U_crit: float
    C_min: float
    Z_switch: float
    Z_recover: float

    def __post_init__(self):                      # enforce the paper's constraints
        assert abs(self.w_L + self.w_U + self.w_C - 1.0) < 1e-9, "weights must sum to 1"
        assert self.rho_max >= 1 and self.U_crit >= self.U_max
        assert 0 <= self.C_min < self.C_target
        assert 0 < self.Z_recover < self.Z_switch <= 1

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(open(path))["health"])


def phi(x):                                       # clamp to [0, 1]
    return min(1.0, max(0.0, x))


def indicators(cfg, L, U, C):
    """Eq. (1). None = stale/missing metric -> indicator = 1 (fail-safe)."""
    sL = 1.0 if L is None else phi(L / cfg.L_slo)
    sU = 1.0 if U is None else phi(U / cfg.U_max)
    sC = 1.0 if C is None else phi(1.0 - C / cfg.C_target)
    return sL, sU, sC


def health_score(cfg, L, U, C):
    """Returns (Z, Z_lin, severe).  L: windowed p95 latency (ms), U in [0,1], C in [0,1]."""
    sL, sU, sC = indicators(cfg, L, U, C)
    z_lin = cfg.w_L * sL + cfg.w_U * sU + cfg.w_C * sC              # Eq. (2)
    severe = ((L is not None and L / cfg.L_slo >= cfg.rho_max) or   # Eq. (3)
              (U is not None and U >= cfg.U_crit) or
              (C is not None and C <= cfg.C_min))
    return max(z_lin, 1.0 if severe else 0.0), z_lin, severe