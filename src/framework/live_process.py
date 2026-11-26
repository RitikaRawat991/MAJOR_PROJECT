"""live_process.py -- runs the live demo system in its OWN process.

Why: when the system ran as a thread inside the dashboard, every chart refresh (and Python's
global interpreter lock) delayed its requests, so the live demo showed late answers and switches
that the experiments (separate process) did not show. Here the dashboard only talks to the system
through a pipe:

  dashboard -> ("snap", n_req, n_ticks)      reply: new requests/ticks since then + current status
  dashboard -> ("inject", label, key, value) reply: time of the fault
  dashboard -> ("stop", run_name)            reply: "saved"; the process then exits

If the dashboard disappears (pipe closed), the system stops and shuts its workers down.
"""
import threading
import time

from experiments.run_scenario import load_configs, load_stream
from src.faults.injectors import active_worker
from src.framework.service import EdgeService


def _worker_status(svc):
    w = active_worker(svc)
    if not w.alive():
        return "DEAD"
    if w.hang.value:
        return "HUNG"
    if not w.ready.value:
        return "restarting..."
    return "alive"


def _inject(svc, key, value):
    if key == "kill":
        active_worker(svc).proc.terminate()
    elif key == "hang":
        active_worker(svc).hang.value = int(value)
    elif key == "lat":
        active_worker(svc).delay_ms.value = float(value)
    elif key == "noise":
        svc.input_noise_kw = float(value)
    else:
        raise ValueError(key)


def _snapshot(svc, n_req, n_ticks):
    if svc.t0 is None:
        return {"started": False}
    ctl = svc.ctl
    return {
        "started": True,
        "now": time.perf_counter() - svc.t0,
        "requests": svc.requests[n_req:],
        "ticks": svc.ticks[n_ticks:],
        "log": [(t.t, t.src.value, t.dst.value, t.reason) for t in list(ctl.log)],
        "state": ctl.state.value,
        "c_v": ctl.c_v,
        "c_r": ctl.c_r,
        "serving": svc._serving(),
        "status": _worker_status(svc),
    }


def live_main(policy, cfg, conn):
    X, y, scaler = load_stream("val")
    hc, sw, rt = load_configs(policy, cfg)
    svc = EdgeService(hc, sw, rt, scaler["std"], policy=policy)
    th = threading.Thread(target=svc.run, args=(X, y, 10 ** 9), daemon=True)
    th.start()
    try:
        while True:
            try:
                if not conn.poll(0.5):
                    continue
                msg = conn.recv()
            except (EOFError, OSError):          # dashboard gone
                break
            if msg[0] == "snap":
                conn.send(_snapshot(svc, msg[1], msg[2]))
            elif msg[0] == "inject":
                _, label, key, value = msg
                if svc.t0 is None:
                    conn.send(None)
                    continue
                _inject(svc, key, value)
                t = time.perf_counter() - svc.t0
                svc.events_log.append((t, f"ui_{label}"))
                conn.send(t)
            elif msg[0] == "stop":
                svc.stop_requested = True
                th.join(timeout=10)
                if svc.t0 is not None:
                    svc.save(msg[1])
                conn.send("saved")
                break
    finally:
        svc.stop_requested = True
        th.join(timeout=10)
        svc.shutdown()
