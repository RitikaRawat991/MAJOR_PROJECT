"""workers.py -- Primary and Fallback models as SEPARATE OS processes (simulated edge node).

A separate process makes a crash real (terminate()), and lets one model be slowed
or hung without freezing the other. Fault hooks live in shared memory values:
    delay_ms : extra delay added by the serving wrapper (latency-spike fault)
    hang     : while 1, the worker stops answering AND stops its heartbeat
"""
import multiprocessing as mp
import time
import numpy as np


def worker_main(kind, conn, hb, ready, delay_ms, hang):
    import joblib
    import torch
    torch.set_num_threads(1)
    from src.data.windows import window_to_features

    if kind == "primary":
        from src.models.primary_lstm import BiLSTM
        from src.framework.reliability import Reliability
        net = BiLSTM()
        net.load_state_dict(torch.load("models/primary_bilstm.pt"))
        net.eval()
        rel = Reliability.load("models/reliability.joblib")

        def infer(w):
            with torch.no_grad():
                p = net(torch.from_numpy(w[None])).item()
            c = float(rel.score(w[None], np.array([p], dtype=np.float32))[0])
            return p, c
    else:
        model = joblib.load("models/fallback_ridge.joblib")

        def infer(w):
            return float(model.predict(window_to_features(w[None]))[0]), None

    infer(np.zeros((60, 3), dtype=np.float32))      # warm-up
    ready.value = 1
    while True:
        if hang.value:                               # hung: no heartbeat, no replies
            time.sleep(0.05)
            continue
        hb.value = time.time()                       # heartbeat
        try:
            if not conn.poll(0.02):
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        rid, window, probe = msg
        pred, c = infer(window)
        if delay_ms.value > 0:
            time.sleep(delay_ms.value / 1000.0)
        try:
            conn.send((rid, pred, c, probe))
        except (BrokenPipeError, OSError):
            break


class WorkerHandle:
    def __init__(self, kind):
        self.kind = kind
        self.proc = None
        self.start()

    def start(self):
        self.parent, child = mp.Pipe()
        self.hb = mp.Value("d", 0.0)
        self.ready = mp.Value("i", 0)
        self.delay_ms = mp.Value("d", 0.0)
        self.hang = mp.Value("i", 0)
        self.rid = 0
        self.proc = mp.Process(
            target=worker_main,
            args=(self.kind, child, self.hb, self.ready, self.delay_ms, self.hang),
            daemon=True)
        self.proc.start()
        child.close()

    def alive(self):
        return self.proc.is_alive()

    def kill(self):
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(2)
        try:
            self.parent.close()
        except OSError:
            pass

    def restart(self):
        self.kill()
        self.start()

    def request(self, window, probe, timeout_s):
        """Returns (pred, c, latency_ms) or None on timeout/error."""
        try:
            while self.parent.poll(0):               # drop stale replies
                self.parent.recv()
            self.rid += 1
            t0 = time.perf_counter()
            self.parent.send((self.rid, window, probe))
            deadline = t0 + timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                if self.parent.poll(remaining):
                    rid, pred, c, _ = self.parent.recv()
                    if rid == self.rid:
                        return pred, c, (time.perf_counter() - t0) * 1000.0
        except (EOFError, OSError, BrokenPipeError):
            return None