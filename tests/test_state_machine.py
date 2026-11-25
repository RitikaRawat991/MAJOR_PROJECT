import numpy as np
from src.framework.health_score import HealthConfig
from src.framework.state_machine import (Obs, State, SwitchConfig,
                                         SwitchController)

H = HealthConfig.from_yaml("configs/default.yaml")
S = SwitchConfig.from_yaml("configs/default.yaml")
BAD = 1.0          # Z >= Z_switch
GOOD = 0.0         # Z <  Z_recover
DEAD = 0.30        # Z_recover <= DEAD < Z_switch  (dead band)


def new():
    return SwitchController(H, S, t0=0.0)


def run(c, t0, n, **kw):
    """Feed n one-second observations starting at t0; return the last time used."""
    t = t0
    for _ in range(n):
        c.step(Obs(t=t, **kw))
        t += 1.0
    return t


# ---------------- basic behaviour ----------------
def test_debounce_short_spike_does_not_switch():
    c = new()
    t = 20.0                                           # dwell already satisfied
    for z in [BAD, BAD, GOOD, BAD, BAD, GOOD]:         # never N_w=3 in a row
        c.step(Obs(t=t, Z=z)); t += 1
    assert c.state is State.PRIMARY


def test_sustained_bad_score_switches_after_Nw_samples():
    c = new()
    run(c, 20.0, S.N_w, Z=BAD)
    assert c.state is State.FALLBACK and c.log[-1].reason == "soft"


def test_dwell_blocks_soft_switch_but_not_hard_fault():
    c = new()
    run(c, 0.0, 5, Z=BAD)                              # t < D=10 -> must stay
    assert c.state is State.PRIMARY
    c.step(Obs(t=5.0, h_P=True, Z=BAD))                # hard fault preempts dwell
    assert c.state is State.FALLBACK and c.log[-1].reason == "hard"


def test_both_hard_faults_go_to_failsafe_and_restore():
    c = new()
    c.step(Obs(t=1.0, h_P=True, h_F=True))
    assert c.state is State.FAILSAFE and c.serving is None
    c.step(Obs(t=2.0, h_F=False, F_live=True))
    assert c.state is State.FALLBACK


def test_full_recovery_path():
    c = new()
    c.step(Obs(t=1.0, h_P=True))                       # -> FALLBACK
    t = 12.0                                           # dwell satisfied
    c.step(Obs(t=t, P_live=True, U=0.3, Z_hat=GOOD))   # -> RECOVERY (counts as probe 1)
    assert c.state is State.RECOVERY
    for k in range(1, S.N_rec):
        c.step(Obs(t=t + k, P_live=True, U=0.3, Z_hat=GOOD))
    assert c.state is State.PRIMARY and c.log[-1].reason == "recovered"


def test_one_bad_probe_aborts_recovery():
    c = new()
    c.step(Obs(t=1.0, h_P=True))
    c.step(Obs(t=12.0, P_live=True, U=0.3, Z_hat=GOOD))
    c.step(Obs(t=13.0, P_live=True, U=0.3, Z_hat=0.9))
    assert c.state is State.FALLBACK and c.log[-1].reason == "failed"


def test_recovery_times_out():
    c = new()
    c.step(Obs(t=1.0, h_P=True))
    c.step(Obs(t=12.0, P_live=True, U=0.3, Z_hat=GOOD))
    c.step(Obs(t=12.0 + S.T_rec + 1, P_live=True, U=0.99, Z_hat=None))   # probes suspended
    assert c.state is State.FALLBACK


def test_no_recovery_attempt_when_saturated():
    c = new()
    c.step(Obs(t=1.0, h_P=True))
    run(c, 12.0, 20, P_live=True, U=0.95, Z_hat=GOOD)  # U >= U_max -> probes suspended
    assert c.state is State.FALLBACK


# ---------------- paper propositions ----------------
def test_prop2a_dead_band_never_leaves_primary():
    c = new()
    run(c, 20.0, 500, Z=DEAD)
    assert c.state is State.PRIMARY and not c.log


def test_prop2b_dead_band_never_starts_recovery():
    c = new()
    c.step(Obs(t=1.0, h_P=True))
    run(c, 12.0, 500, P_live=True, U=0.3, Z_hat=DEAD)
    assert c.state is State.FALLBACK


def test_prop2_recovery_aborted_in_dead_band():
    c = new()
    c.step(Obs(t=1.0, h_P=True))
    c.step(Obs(t=12.0, P_live=True, U=0.3, Z_hat=GOOD))
    c.step(Obs(t=13.0, P_live=True, U=0.3, Z_hat=DEAD))
    assert c.state is State.FALLBACK


ALLOWED = {(State.PRIMARY, State.FALLBACK), (State.PRIMARY, State.FAILSAFE),
           (State.FALLBACK, State.RECOVERY), (State.FALLBACK, State.FAILSAFE),
           (State.RECOVERY, State.PRIMARY), (State.RECOVERY, State.FALLBACK),
           (State.RECOVERY, State.FAILSAFE), (State.FAILSAFE, State.FALLBACK)}


def _random_run(seed, T=3000):
    rng = np.random.default_rng(seed)
    c = new()
    for k in range(T):
        c.step(Obs(t=float(k),
                   h_P=rng.random() < 0.03, h_F=rng.random() < 0.01,
                   Z=rng.random(), Z_hat=rng.random() * 0.5,
                   P_live=rng.random() < 0.8, F_live=rng.random() < 0.7,
                   U=rng.random()))
    return c, T


def test_prop1a_soft_and_probe_transitions_are_dwell_gated():
    for seed in range(20):
        c, _ = _random_run(seed)
        prev_t = 0.0
        for tr in c.log:
            gated = (tr.src, tr.dst, tr.reason) in {
                (State.PRIMARY, State.FALLBACK, "soft"),
                (State.FALLBACK, State.RECOVERY, "probe")}
            if gated:
                assert tr.t - prev_t >= S.D            # >= D since the previous transition
            prev_t = tr.t


def test_prop1b_primary_exits_are_bounded():
    for seed in range(20):
        c, T = _random_run(seed)
        exits = sum(1 for tr in c.log if tr.src is State.PRIMARY)
        assert exits <= T // S.D + 3


def test_only_allowed_transitions_ever_happen():
    for seed in range(20):
        c, _ = _random_run(seed)
        assert all((tr.src, tr.dst) in ALLOWED for tr in c.log)
