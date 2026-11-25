import numpy as np
from src.framework.health_score import HealthConfig, health_score

cfg = HealthConfig.from_yaml("configs/default.yaml")


def test_bounded():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        z, _, _ = health_score(cfg, rng.uniform(0, 50), rng.uniform(0, 1), rng.uniform(0, 1))
        assert 0.0 <= z <= 1.0


def test_nondecreasing_in_latency():
    zs = [health_score(cfg, L, 0.3, 0.5)[0] for L in np.linspace(0, 30, 200)]
    assert all(b >= a - 1e-12 for a, b in zip(zs, zs[1:]))


def test_nondecreasing_in_utilisation():
    zs = [health_score(cfg, 1.0, U, 0.5)[0] for U in np.linspace(0, 1, 200)]
    assert all(b >= a - 1e-12 for a, b in zip(zs, zs[1:]))


def test_nonincreasing_in_confidence():
    zs = [health_score(cfg, 3.0, 0.5, C)[0] for C in np.linspace(0, 1, 200)]
    assert all(b <= a + 1e-12 for a, b in zip(zs, zs[1:]))


def test_severe_override_is_noncompensatory():
    # perfect confidence cannot hide a severe latency violation
    z, z_lin, severe = health_score(cfg, 3 * cfg.L_slo, 0.1, 1.0)
    assert severe and z == 1.0 and z_lin < 1.0


def test_latency_alone_triggers_switch_but_cpu_alone_does_not():
    assert health_score(cfg, 2 * cfg.L_slo, 0.1, 0.9)[0] >= cfg.Z_switch
    assert health_score(cfg, 0.0, 0.90, 0.9)[0] < cfg.Z_switch


def test_stale_metrics_fail_safe():
    assert health_score(cfg, None, None, None)[0] == 1.0
