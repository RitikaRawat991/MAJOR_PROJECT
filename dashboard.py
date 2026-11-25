"""dashboard.py -- one place to run and show everything in Phase 1.
Run (on Windows 'python -m' is needed for the worker processes):
    python -m streamlit run dashboard.py

Pages
  1. Live demo            start the system, inject faults by clicking, watch it react
  2. Run experiments      fault scenarios x policies, run in the background
  3. Results              metrics of all runs, comparison charts, hypothesis checks
  4. Run viewer           timeline of one saved run
  5. Setup & calibration  thresholds, calibration run, nominal check, clear results

Every button here calls the real backend (src/framework, experiments/).
NOTE: the paper's "AI interface" is the orchestration layer in src/framework, not this GUI.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.eval.metrics import T_OFF, T_ON, TOL_KW, run_metrics
from src.faults.injectors import active_worker

ROOT = Path(__file__).resolve().parent
RESULTS = Path("results")
CFG = "configs/calibrated.yaml"
SCENARIOS = ["kill", "hang", "latency", "noise"]
POLICIES = ["P", "A", "B", "C", "E"]
POLICY_NAMES = {"P": "P - proposed framework", "E": "E - Primary + restart", "A": "A - Primary only",
                "C": "C - simple threshold", "B": "B - Fallback only"}
SCEN_NAMES = {"kill": "Process kill (F1)", "hang": "Process hang (F1)",
              "latency": "Latency spike +25 ms (F2)", "noise": "Input noise 1 kW (F3)"}
MODEL_NAMES = {"primary": "Primary (Bi-LSTM)", "fallback": "Fallback (Ridge)"}
LIVE_WINDOW = 60.0
RUN_SECONDS = 140                       # approx. wall time of one scenario run incl. startup
SPLITS = {"val": "Validation data (development runs)", "test": "Test data (final evaluation, run once)"}
PREFIX = {"val": "scenario", "test": "testset"}        # same as experiments/run_scenario.py
FAULT_PREFIXES = ("scenario_", "testset_")

# Fixed list of project commands that page 2 can run (no free-text commands).
# key: (label, python module, arguments, warning shown before running or None)
COMMANDS = {
    "tests": ("Unit tests: health score and state machine (~5 s)",
              "pytest", ["tests", "-v", "-p", "no:cacheprovider"], None),
    "scenario": ("One fault scenario, printed like in the terminal (~2.5 min)",
                 "experiments.run_scenario", None, None),
    "nominal": ("Fault-free check: should give 0 switches (~1.5 min)",
                "experiments.run_nominal", [CFG], None),
    "sweep": ("Fallback selection: compare 5 candidate models (~3-5 min)",
              "src.eval.fallback_sweep", [], None),
    "calibrate": ("Re-calibrate thresholds from a fault-free run (~2 min)",
                  "experiments.calibrate", [],
                  "Overwrites configs/calibrated.yaml. Results saved earlier will no longer match the thresholds."),
    "data": ("Rebuild data: clean + chronological split (~1 min)",
             "src.data.load_data", [], "Overwrites data/processed/*.csv."),
    "primary": ("Re-train the Primary Bi-LSTM (~5-8 min)",
                "src.models.primary_lstm", [],
                "Overwrites models/primary_bilstm.pt and models/scaler.json (same seed, so it should give the same model)."),
    "ridge": ("Re-train the Fallback Ridge (~1 min)",
              "src.models.fallback_ridge", [], "Overwrites models/fallback_ridge.joblib."),
    "reliability": ("Re-fit the reliability score c (~1-2 min)",
                    "src.framework.reliability", [], "Overwrites models/reliability.joblib."),
}

st.set_page_config(page_title="Fault-Tolerant Edge AI", layout="wide")
CAL_OK = Path(CFG).exists()
CFG_ALL = yaml.safe_load(open(CFG if CAL_OK else "configs/default.yaml"))
H_CFG = CFG_ALL["health"]
L_SLO = float(H_CFG["L_slo"])
STD = json.load(open("models/scaler.json"))["std"]


# ======================================================================= shared state
@st.cache_resource
def live_holder():
    return {"svc": None, "thread": None, "events": [], "policy": None, "faults": set()}


@st.cache_resource
def jobs():
    return {"thread": None, "tasks": [], "done": [], "current": None, "t_cur": None,
            "error": None, "cancel": False, "log": []}


def live_running():
    return live_holder()["svc"] is not None


def jobs_running():
    th = jobs()["thread"]
    return th is not None and th.is_alive()


def start_jobs(tasks, kind="batch"):
    J = jobs()
    J.update(tasks=list(tasks), done=[], current=None, error=None, cancel=False, log=[],
             kind=kind, proc=None)

    def worker():
        for label, fn in J["tasks"]:
            if J["cancel"]:
                break
            J["current"], J["t_cur"] = label, time.time()
            try:
                info = fn()
                J["done"].append((label, info, time.time() - J["t_cur"]))
            except Exception as e:                       # show the error in the UI
                J["error"] = f"{label}: {e!r}"
                break
        J["current"] = None

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    J["thread"] = th


def start_command(label, module, args, run=None):
    """Run one project command as a child process and stream its output into jobs()['log'].
    run: name of the results/ run this command will save (its result is shown when it finishes)."""
    J = jobs()

    def fn():
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        p = subprocess.Popen([sys.executable, "-u", "-m", module, *args], cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env)
        J["proc"] = p
        for line in p.stdout:
            J["log"].append(line.rstrip())
        code = p.wait()
        J["proc"] = None
        if J["cancel"]:
            return "stopped by user"
        if code != 0:
            raise RuntimeError(f"exit code {code}")
        return "finished (exit code 0)"

    start_jobs([(label, fn)], kind="command")
    J["run"] = run


def stop_command():
    J = jobs()
    J["cancel"] = True
    p = J.get("proc")
    if p is not None and p.poll() is None:          # kill the whole tree (incl. model workers)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


# ======================================================================= charts
def decorate(chart, bands=(), marks=()):
    layers = []
    if bands:
        layers.append(alt.Chart(pd.DataFrame(bands, columns=["x", "x2"])).mark_rect(
            opacity=0.12, color="#d62728").encode(x="x:Q", x2="x2:Q"))
    layers.append(chart)
    if marks:
        m = pd.DataFrame(marks, columns=["t", "label"])
        layers.append(alt.Chart(m).mark_rule(color="#d62728", strokeDash=[3, 3]).encode(x="t:Q"))
        layers.append(alt.Chart(m).mark_text(align="left", dx=3, dy=-6, color="#d62728").encode(
            x="t:Q", y=alt.value(8), text="label:N"))
    return alt.layer(*layers)


def x_axis(domain):
    return alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=domain) if domain else alt.Undefined)


def served_chart(req, domain=None, bands=(), marks=()):
    lvl = req["served"].map(MODEL_NAMES).fillna("No answer")
    lat = pd.to_numeric(req["lat_ms"], errors="coerce")
    ok = lat.notna() & (lat <= L_SLO)
    d = pd.DataFrame({"t": req["t"], "served": lvl,
                      "answered": ok.map({True: "within SLO", False: "missed SLO"})})
    c = alt.Chart(d).mark_tick(thickness=2).encode(
        x=x_axis(domain),
        y=alt.Y("served:N", title=None, sort=list(MODEL_NAMES.values()) + ["No answer"]),
        color=alt.Color("answered:N", scale=alt.Scale(domain=["within SLO", "missed SLO"],
                                                      range=["#2ca02c", "#d62728"])))
    return decorate(c, bands, marks).properties(height=170)


def latency_chart(req, domain=None, bands=(), marks=()):
    d = pd.DataFrame({"t": req["t"], "latency": pd.to_numeric(req["lat_ms"], errors="coerce"),
                      "model": req["served"].map(MODEL_NAMES)}).dropna()
    c = alt.Chart(d).mark_circle(size=10, opacity=0.6).encode(
        x=x_axis(domain), y=alt.Y("latency:Q", title="ms"), color=alt.Color("model:N", title=None))
    slo = alt.Chart(pd.DataFrame({"y": [L_SLO]})).mark_rule(strokeDash=[4, 4], color="black").encode(y="y:Q")
    return decorate(c + slo, bands, marks).properties(height=220)


def score_chart(ticks, domain=None, bands=(), marks=()):
    cols = [c for c in ("Z", "Z_hat") if c in ticks]
    if not cols:
        return None
    long = ticks.melt(id_vars="t", value_vars=cols, var_name="score", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna()
    if long.empty:
        return None
    c = alt.Chart(long).mark_line(point=True).encode(
        x=x_axis(domain), y=alt.Y("value:Q", title="health score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("score:N", title=None))
    thr = pd.DataFrame({"y": [H_CFG["Z_switch"], H_CFG["Z_recover"]]})
    rules = alt.Chart(thr).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    return decorate(c + rules, bands, marks).properties(height=200)


def model_latency_table(req):
    rows = []
    lat_all = pd.to_numeric(req["lat_ms"], errors="coerce") if not req.empty else pd.Series(dtype=float)
    for key, name in MODEL_NAMES.items():
        lat = lat_all[req["served"] == key].dropna() if not req.empty else lat_all
        if len(lat):
            rows.append({"model": name, "answers": len(lat), "mean ms": round(lat.mean(), 2),
                         "p50 ms": round(lat.quantile(0.50), 2), "p95 ms": round(lat.quantile(0.95), 2),
                         "max ms": round(lat.max(), 2), "within SLO %": round((lat <= L_SLO).mean() * 100, 1)})
    return pd.DataFrame(rows)


# ======================================================================= saved results
def load(run, part):
    p = RESULTS / f"{run}_{part}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def list_runs():
    return sorted(p.name[: -len("_requests.csv")] for p in RESULTS.glob("*_requests.csv"))


@st.cache_data
def metrics_cached(run, mtime, l_slo):
    return run_metrics(load(run, "requests"), load(run, "transitions"), load(run, "events"),
                       l_slo, STD, fault=run.startswith(FAULT_PREFIXES))


def metrics_for(run):
    return metrics_cached(run, (RESULTS / f"{run}_requests.csv").stat().st_mtime, L_SLO)


def scenario_table(split="val"):
    rows = []
    for run in list_runs():
        parts = run.split("_")
        if len(parts) == 3 and parts[0] == PREFIX[split] and parts[1] in SCENARIOS and parts[2] in POLICIES:
            rows.append({"scenario": parts[1], "policy": parts[2], **metrics_for(run)})
    return pd.DataFrame(rows)


# ======================================================================= page 1: live demo
@st.cache_data
def stream():
    from src.data.windows import load_split, make_windows
    scaler = json.load(open("models/scaler.json"))
    Xva, yva, _ = make_windows(load_split("val"), scaler, stride=5)
    h = len(Xva) // 2
    return Xva[h:], yva[h:], scaler["std"]


def start_live(policy):
    from experiments.run_scenario import load_configs
    from src.framework.service import EdgeService
    X, y, std = stream()
    hc, sw, rt = load_configs(policy, CFG)
    svc = EdgeService(hc, sw, rt, std, policy=policy)
    H = live_holder()
    H.update(svc=svc, events=[], policy=policy, faults=set())

    def target():
        try:
            svc.run(X, y, 10 ** 9)
        finally:
            svc.shutdown()

    th = threading.Thread(target=target, daemon=True)
    th.start()
    H["thread"] = th


def stop_live():
    H = live_holder()
    svc = H["svc"]
    svc.stop_requested = True
    if H["thread"] is not None:
        H["thread"].join(timeout=10)
    if svc.t0 is not None:
        svc.save(f"live_{H['policy']}_{time.strftime('%H%M%S')}")
    H.update(svc=None, thread=None, faults=set())


def inject(label, fn, fault_key=None, on=True):
    H = live_holder()
    svc = H["svc"]
    if svc is None or svc.t0 is None:
        st.toast("System is still starting. Try again in a few seconds.")
        return
    fn(svc)
    t = time.perf_counter() - svc.t0
    H["events"].append((t, label))
    svc.events_log.append((t, f"ui_{label}"))
    if fault_key:
        (H["faults"].add if on else H["faults"].discard)(fault_key)


def page_live():
    st.header("1. Live demo")
    H = live_holder()
    svc = H["svc"]
    if svc is None:
        if jobs_running():
            st.warning("Experiments are running in the background. Wait for them to finish "
                       "(page 2), otherwise both would disturb each other's timing.")
            return
        policy = st.selectbox("Policy", POLICIES, format_func=POLICY_NAMES.get)
        st.write("Start the system, wait ~10 s, then inject faults with the buttons and watch it react.")
        if st.button("Start system", type="primary"):
            start_live(policy)
            st.rerun()
        return

    top = st.columns([3, 1])
    top[0].write(f"Running policy **{POLICY_NAMES[H['policy']]}**")
    if top[1].button("Stop system (saves the run)"):
        stop_live()
        st.rerun()

    st.subheader("Inject a fault")
    b = st.columns(5)
    on = H["faults"]
    tgt = "Primary" if svc.P is not None else "Fallback"
    if tgt == "Fallback":
        st.info("Policy B has only the Fallback model, so model faults hit the Fallback. "
                "Nothing restarts it: if it dies, the whole service stops.")
    if b[0].button(f"Kill {tgt}"):
        inject("kill", lambda s: active_worker(s).proc.terminate())
    if b[1].button(f"Unhang {tgt}" if "hang" in on else f"Hang {tgt}"):
        v = 0 if "hang" in on else 1
        inject("unhang" if v == 0 else "hang",
               lambda s: setattr(active_worker(s).hang, "value", v), "hang", v == 1)
    if b[2].button("Latency spike OFF" if "lat" in on else f"Latency spike ON ({tgt} +25 ms)"):
        v = 0.0 if "lat" in on else 25.0
        inject("latency off" if v == 0 else "latency +25ms",
               lambda s: setattr(active_worker(s).delay_ms, "value", v), "lat", v > 0)
    if b[3].button("Noise OFF" if "noise" in on else "Input noise ON"):
        v = 0.0 if "noise" in on else 1.0
        inject("noise off" if v == 0 else "noise on",
               lambda s: setattr(s, "input_noise_kw", v), "noise", v > 0)
    b[4].caption("Kill/hang are repaired by the system itself (restart) in policies P, C and E.")
    active = [name for key, name in (("lat", "latency spike (+25 ms)"), ("hang", f"{tgt} hang"),
                                     ("noise", "input noise")) if key in H["faults"]]
    if active:
        st.warning(f"Fault still ON: {', '.join(active)}. These buttons are switches: press the same "
                   "button again (now labelled OFF / Unhang) to end the fault. The system will not return "
                   "to the Primary while the Primary is still slow.")
    live_view()


@st.fragment(run_every=1.0)
def live_view():
    H = live_holder()
    svc = H["svc"]
    if svc is None:
        return
    if svc.t0 is None:
        st.info("Starting worker processes and loading models...")
        return
    now = time.perf_counter() - svc.t0
    req_all = pd.DataFrame(list(svc.requests))
    if req_all.empty:
        return
    req = req_all[req_all["t"] >= now - LIVE_WINDOW]
    ticks = pd.DataFrame(list(svc.ticks))
    if not ticks.empty:
        ticks = ticks[ticks["t"] >= now - LIVE_WINDOW]
    domain = [max(0.0, now - LIVE_WINDOW), max(LIVE_WINDOW, now)]
    marks = [(t, lab) for t, lab in H["events"] if t >= domain[0]]

    w = active_worker(svc)
    tgt = "Primary" if svc.P is not None else "Fallback"
    if not w.alive():
        status = "DEAD"
    elif w.hang.value:
        status = "HUNG"
    elif not w.ready.value:
        status = "restarting..."
    else:
        status = "alive"
    serving = svc._serving()
    state = svc.ctl.state.value if H["policy"] in ("P", "C") else "-"
    last10 = req_all[req_all["t"] >= now - 10]
    lat10 = pd.to_numeric(last10["lat_ms"], errors="coerce")
    avail10 = ((lat10.notna()) & (lat10 <= L_SLO)).mean() * 100 if len(last10) else 0.0
    missed10 = int(lat10.isna().sum())

    c = st.columns(6)
    c[0].metric("Running for", f"{now:.0f} s")
    c[1].metric(f"{tgt} process", status)
    c[2].metric("Controller state", state)
    c[3].metric("Serving now", MODEL_NAMES.get(serving, "nobody") if serving else "nobody")
    c[4].metric("Availability (last 10 s)", f"{avail10:.0f}%", delta=f"{missed10} unanswered",
                delta_color="inverse" if missed10 else "off")
    c[5].metric("Transitions", len(svc.ctl.log))

    st.markdown("**Who served each request** (green = within SLO, red = missed)")
    st.altair_chart(served_chart(req, domain, marks=marks), use_container_width=True)
    st.markdown(f"**Response time per request** (dotted line = SLO {L_SLO:.0f} ms)")
    st.altair_chart(latency_chart(req, domain, marks=marks), use_container_width=True)
    if not ticks.empty:
        sc = score_chart(ticks, domain, marks=marks)
        if sc is not None:
            st.markdown("**Health score** (dotted: Z_switch above, Z_recover below)")
            st.altair_chart(sc, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Response time by model** (whole run)")
        st.dataframe(model_latency_table(req_all), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**State transitions**")
        log = [dict(t=round(t.t, 1), change=f"{t.src.value} -> {t.dst.value}", reason=t.reason)
               for t in svc.ctl.log]
        st.dataframe(pd.DataFrame(log) if log else pd.DataFrame({"info": ["none yet"]}),
                     use_container_width=True, hide_index=True)


# ======================================================================= page 2: run experiments
def scenario_task(sc, pol, split):
    def fn():
        from experiments.run_scenario import run_one
        svc, _, l_slo = run_one(sc, pol, CFG, split)
        fault = [r for r in svc.requests if T_ON <= r["t"] < T_OFF]
        a = np.mean([r["lat_ms"] is not None and r["lat_ms"] <= l_slo for r in fault]) * 100
        return f"availability in fault window {a:.1f}%, transitions {len(svc.ctl.log)}"
    return (f"{sc} / {pol} ({split})", fn)


def page_experiments():
    st.header("2. Run experiments")
    # Always rendered (disabled while busy), so Streamlit keeps its value across reruns.
    mode = st.radio("What to run", ["One scenario", "Experiment batch", "Project command"], horizontal=True,
                    key="exp_mode", disabled=jobs_running() or live_running())
    if jobs_running():
        progress_view()
        return
    if live_running():
        st.warning("The live demo is running. Stop it on page 1 first (both would disturb each other's timing).")
        return
    if mode == "One scenario":
        scenario_panel()
        return
    if mode == "Project command":
        command_panel()
        return

    st.write(f"Each run lasts 130 s: normal until {T_ON:.0f} s, fault from {T_ON:.0f} s to {T_OFF:.0f} s, "
             "then 50 s to recover. The same data and the same fault times are used for every policy, "
             "so the runs can be compared request by request.")
    if not CAL_OK:
        st.error("No calibrated thresholds yet. Run the calibration on page 5 first.")
        return
    split = st.radio("Data", list(SPLITS), format_func=SPLITS.get, horizontal=True)
    if split == "test":
        st.info("The thresholds were calibrated on validation data. Use the test data only for the "
                "final numbers you report, after the code and thresholds are frozen.")
    c = st.columns(2)
    scen = c[0].multiselect("Fault scenarios", SCENARIOS, default=SCENARIOS, format_func=SCEN_NAMES.get)
    pols = c[1].multiselect("Policies", POLICIES, default=POLICIES, format_func=POLICY_NAMES.get)
    rerun = st.checkbox("Run again even if a result is already saved", value=False)
    todo = [(s, p) for s in scen for p in pols
            if rerun or not (RESULTS / f"{PREFIX[split]}_{s}_{p}_requests.csv").exists()]
    skipped = len(scen) * len(pols) - len(todo)
    st.write(f"**{len(todo)} runs** to do (~{len(todo) * RUN_SECONDS / 60:.0f} min)"
             + (f", {skipped} already saved and skipped." if skipped else "."))
    st.caption("Keep the laptop plugged in and do not use heavy apps while it runs: latency is measured live.")
    if st.button("Run", type="primary", disabled=not todo):
        start_jobs([scenario_task(s, p, split) for s, p in todo])
        st.rerun()
    last = jobs()
    if last["done"] or last["error"]:
        st.subheader("Last batch")
        progress_table(last)


def scenario_panel():
    st.write(f"Pick a fault and a policy and press **Run**. The run takes about {RUN_SECONDS // 60} min "
             f"({T_ON:.0f} s normal, fault from {T_ON:.0f} to {T_OFF:.0f} s, then recovery). "
             "The result with charts appears here when it is done.")
    if not CAL_OK:
        st.error("No calibrated thresholds yet. Run the calibration on page 5 first.")
        return
    c = st.columns(3)
    sc = c[0].selectbox("Fault", SCENARIOS, format_func=SCEN_NAMES.get, key="one_sc")
    pol = c[1].selectbox("Policy", POLICIES, format_func=POLICY_NAMES.get, key="one_pol")
    split = c[2].selectbox("Data", list(SPLITS), format_func=SPLITS.get, key="one_split")
    args = [sc, pol, CFG, split]
    run = f"{PREFIX[split]}_{sc}_{pol}"
    saved = (RESULTS / f"{run}_requests.csv").exists()
    st.caption("Same as typing in the terminal:")
    st.code("python -m experiments.run_scenario " + " ".join(args), language="powershell")
    if st.button("Run again (replaces the saved result)" if saved else "Run", type="primary", key="one_run"):
        start_command(f"{sc} / {pol} ({split})", "experiments.run_scenario", args, run)
        st.rerun()
    st.divider()
    if saved:
        st.subheader(f"Saved result: {SCEN_NAMES[sc]}, {POLICY_NAMES[pol]}")
        show_run(run)
    else:
        st.info("No saved result for this choice yet. Press Run.")


def command_panel():
    st.write("Runs one project command and shows its output here, exactly as the terminal would.")
    key = st.selectbox("Command", list(COMMANDS), format_func=lambda k: COMMANDS[k][0])
    label, module, args, warn = COMMANDS[key]
    if key == "scenario":
        c = st.columns(3)
        sc = c[0].selectbox("Scenario", SCENARIOS, format_func=SCEN_NAMES.get)
        pol = c[1].selectbox("Policy", POLICIES, format_func=POLICY_NAMES.get)
        split = c[2].selectbox("Data", list(SPLITS), format_func=SPLITS.get)
        args = [sc, pol, CFG, split]
    st.caption("Same as typing in the terminal:")
    st.code("python -m " + " ".join([module, *args]), language="powershell")
    if key in ("scenario", "nominal") and not CAL_OK:
        st.error("No calibrated thresholds yet. Run the calibration first.")
        return
    ok = True
    if warn:
        st.warning(warn)
        ok = st.checkbox("I understand, run it")
    if st.button("Run command", type="primary", disabled=not ok):
        run = f"{PREFIX[args[3]]}_{args[0]}_{args[1]}" if key == "scenario" else None
        start_command(label, module, args, run)
        st.rerun()
    J = jobs()
    if J.get("kind") == "command" and J["log"]:
        with st.expander(f"Output of the last command: {J['done'][0][0] if J['done'] else ''}"):
            st.code("\n".join(J["log"][-300:]), language="text")


def command_progress(J):
    running = jobs_running()
    run = J.get("run")
    if running:
        elapsed = time.time() - J["t_cur"]
        if run:                                     # a scenario run: about RUN_SECONDS long
            st.progress(min(elapsed / RUN_SECONDS, 0.99),
                        text=f"Running {J['current']}: {elapsed:.0f} s of about {RUN_SECONDS} s")
        else:
            st.info(f"Running: {J['current']}  ({elapsed:.0f} s)")
        if J["cancel"]:
            st.caption("Stopping...")
        elif st.button("Stop now"):
            stop_command()
        st.code("\n".join(J["log"][-40:]) or "(waiting for output...)", language="text")
        return
    if J["error"]:
        st.error(f"Stopped with an error: {J['error']}")
    else:
        st.success(f"{J['done'][0][0] if J['done'] else 'Command'}: "
                   f"{J['done'][0][1] if J['done'] else 'finished'}")
    if st.button("OK"):
        st.rerun()
    if run and (RESULTS / f"{run}_requests.csv").exists() and not J["error"]:
        show_run(run)
    with st.expander("Terminal output", expanded=not run):
        st.code("\n".join(J["log"][-300:]) or "(no output)", language="text")


def progress_table(J):
    rows = [{"run": lab, "result": info, "took": f"{dt:.0f} s"} for lab, info, dt in J["done"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if J["error"]:
        st.error(f"Stopped with an error: {J['error']}")


@st.fragment(run_every=2.0)
def progress_view():
    J = jobs()
    if J.get("kind") == "command":
        command_progress(J)
        return
    total, done = len(J["tasks"]), len(J["done"])
    if not jobs_running():
        st.success(f"Finished: {done} of {total} runs. Open page 3 (Results).")
        progress_table(J)
        if st.button("OK"):
            st.rerun()
        return
    cur_t = time.time() - J["t_cur"] if J["t_cur"] else 0
    frac = min(1.0, (done + min(cur_t / RUN_SECONDS, 0.99)) / max(total, 1))
    st.progress(frac, text=f"Running {J['current']}  ({done} of {total} done, "
                           f"~{max(0, (total - done) * RUN_SECONDS - cur_t) / 60:.0f} min left)")
    if J["cancel"]:
        st.info("Will stop after the current run.")
    elif st.button("Stop after the current run"):
        J["cancel"] = True
    progress_table(J)


# ======================================================================= page 3: results
def page_results():
    st.header("3. Results")
    split = st.radio("Data", list(SPLITS), format_func=SPLITS.get, horizontal=True)
    df = scenario_table(split)
    if df.empty:
        st.info("No experiment results for this data yet. Run them on page 2.")
        return
    st.caption(f"Pilot results: 1 run per cell. Availability A = answered within {L_SLO:.0f} ms. "
               f"Q = answered within the SLO AND error <= {TOL_KW} kW (paper Eq. 6). "
               "'fault' columns use only the fault window (30-80 s).")

    show = ["scenario", "policy", "A fault %", "Q fault %", "fallback share fault %", "detection s",
            "service recovery s", "back to Primary s", "transitions", "false switches", "MAE kW",
            "p95 primary ms", "p95 fallback ms"]
    view = df[show].sort_values(["scenario", "policy"]).round(2)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download table (CSV)", view.to_csv(index=False), f"phase1_results_{split}.csv")

    st.subheader("Availability and quality during the fault")
    long = df.melt(id_vars=["scenario", "policy"], value_vars=["A fault %", "Q fault %"],
                   var_name="metric", value_name="value")
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("policy:N", title=None), y=alt.Y("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("policy:N", legend=None), tooltip=["scenario", "policy", "metric", "value"]
    ).properties(width=150, height=180).facet(row=alt.Row("metric:N", title=None),
                                              column=alt.Column("scenario:N", title=None))
    st.altair_chart(chart)

    st.subheader("Hypothesis checks (pilot, 1 run each)")
    rows = []
    for sc in ("kill", "hang", "latency"):
        a = df[(df.scenario == sc) & (df.policy == "A")]
        p = df[(df.scenario == sc) & (df.policy == "P")]
        if len(a) and len(p):
            qa, qp = a["Q fault %"].iat[0], p["Q fault %"].iat[0]
            rows.append({"hypothesis": "H1: A has lower Q than P during F1/F2 faults", "scenario": sc,
                         "values": f"A {qa:.1f}% vs P {qp:.1f}%",
                         "result": "supported" if qa < qp else "NOT supported"})
    for sc in SCENARIOS:
        cc = df[(df.scenario == sc) & (df.policy == "C")]
        p = df[(df.scenario == sc) & (df.policy == "P")]
        if len(cc) and len(p):
            tc, tp = cc["transitions"].iat[0], p["transitions"].iat[0]
            rows.append({"hypothesis": "H2: C switches more often than P", "scenario": sc,
                         "values": f"C {tc} vs P {tp}",
                         "result": "supported" if tc > tp else ("tie" if tc == tp else "NOT supported")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("One run per cell is not a statistical test. Phase 2: 30 repetitions, "
                   "bootstrap confidence intervals and Wilcoxon tests.")
    else:
        st.info("Run policies A, C and P to check H1 and H2.")


# ======================================================================= page 4: run viewer
def phase_table(req):
    """Same table the terminal prints: before / during / after the fault."""
    lat = pd.to_numeric(req["lat_ms"], errors="coerce")
    ok = lat.notna() & (lat <= L_SLO)
    err = (pd.to_numeric(req["pred"], errors="coerce") - req["y"]).abs() * STD
    rows = []
    for name, m in [(f"before (0-{T_ON:.0f} s)", req["t"] < T_ON),
                    (f"fault ({T_ON:.0f}-{T_OFF:.0f} s)", (req["t"] >= T_ON) & (req["t"] < T_OFF)),
                    (f"after ({T_OFF:.0f} s-end)", req["t"] >= T_OFF)]:
        rows.append({"phase": name, "requests": int(m.sum()),
                     "availability %": round(ok[m].mean() * 100, 1) if m.any() else np.nan,
                     "MAE kW": round(err[m].mean(), 3) if m.any() else np.nan,
                     "served by Fallback %": round((req["served"][m] == "fallback").mean() * 100, 1)
                     if m.any() else np.nan})
    return pd.DataFrame(rows)


def run_summary(m, trans):
    """One plain sentence describing what the system did."""
    if m["transitions"] == 0:
        return "No switch happened: the Primary kept serving for the whole run."
    parts = []
    if not np.isnan(m.get("detection s", np.nan)):
        parts.append(f"first reaction {m['detection s']:.1f} s after the fault started")
    back = m.get("back to Primary s", np.nan)
    if not np.isnan(back):
        after_off = back - (T_OFF - T_ON)
        parts.append(f"back to the Primary {back:.1f} s after the fault started"
                     + (f" ({after_off:.1f} s after it ended)" if after_off > 0 else ""))
    if m.get("false switches", 0):
        parts.append(f"{m['false switches']} switch(es) before the fault (false alarms)")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."


def show_run(run):
    """Full result of one saved run: phase table, summary, transitions and charts."""
    is_fault = run.startswith(FAULT_PREFIXES)
    req, ticks, trans = load(run, "requests"), load(run, "ticks"), load(run, "transitions")
    events = load(run, "events")
    bands = [(T_ON, T_OFF)] if is_fault else []
    m = metrics_for(run)

    c = st.columns(5)
    c[0].metric("Requests", m["requests"])
    c[1].metric("Availability" + (" (fault)" if is_fault else ""),
                f"{m['A fault %' if is_fault else 'A %']:.1f}%")
    c[2].metric("Quality-cond. availability" + (" (fault)" if is_fault else ""),
                f"{m['Q fault %' if is_fault else 'Q %']:.1f}%")
    c[3].metric("Transitions", m["transitions"])
    if is_fault:
        d = m["detection s"]
        c[4].metric("Detection", "none" if np.isnan(d) else f"{d:.1f} s")
    if is_fault:
        st.info(run_summary(m, trans))
        st.markdown(f"**Before / during / after the fault** (availability = answered within {L_SLO:.0f} ms)")
        st.dataframe(phase_table(req), use_container_width=True, hide_index=True)
        st.caption("Do not compare MAE across phases: each phase uses different data. "
                   "Compare policies on the same phase instead (page 3).")
        restarts = events[events["event"].astype(str).str.startswith("restart")] if not events.empty else events
        if not restarts.empty:
            st.caption("Restarts: " + ", ".join(f"{t:.1f} s" for t in restarts["t"]))

    st.markdown("**Who served each request**" + (" (shaded = fault active)" if is_fault else ""))
    st.altair_chart(served_chart(req, bands=bands), use_container_width=True)
    st.markdown("**Response time per request**")
    st.altair_chart(latency_chart(req, bands=bands), use_container_width=True)
    if not ticks.empty:
        sc = score_chart(ticks, bands=bands)
        if sc is not None:
            st.markdown("**Health score**")
            st.altair_chart(sc, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Response time by model**")
        st.dataframe(model_latency_table(req), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**State transitions**")
        st.dataframe(trans if not trans.empty else pd.DataFrame({"info": ["no transitions"]}),
                     use_container_width=True, hide_index=True)


def page_viewer():
    st.header("4. Run viewer")
    runs = list_runs()
    if not runs:
        st.info("No saved runs yet.")
        return
    run = st.selectbox("Run", runs, index=len(runs) - 1)
    show_run(run)


# ======================================================================= page 5: setup & calibration
def page_setup():
    st.header("5. Setup & calibration")
    st.markdown("**Models in use**: Primary = Bi-LSTM (val MAE 0.378 kW), Fallback = Ridge "
                "(val MAE 0.404 kW, about 4x faster). The Fallback was chosen by measurement; "
                "every Random Forest was slower than the Primary (see DECISIONS.md, section 7).")
    missing = [p for p in ("models/primary_bilstm.pt", "models/fallback_ridge.joblib",
                           "models/reliability.joblib", "models/scaler.json") if not Path(p).exists()]
    if missing:
        st.error(f"Missing model files: {missing}. Train them first (see README).")

    st.subheader("Current thresholds")
    if CAL_OK:
        st.dataframe(pd.DataFrame([{"parameter": k, "value": v} for k, v in H_CFG.items()]),
                     hide_index=True)
        st.caption("Derived from a fault-free run by fixed rules (experiments/calibrate.py).")
    else:
        st.warning("Not calibrated yet: run the calibration below.")

    if jobs_running():
        progress_view()
        return
    busy = live_running()
    if busy:
        st.warning("Stop the live demo on page 1 before running these.")
    c = st.columns(2)
    if c[0].button("Run calibration (~2 min)", disabled=busy):
        def cal():
            from experiments.calibrate import calibrate
            return ", ".join(f"{k}={v}" for k, v in calibrate(log=lambda *_: None).items())
        start_jobs([("calibration", cal)])
        st.rerun()
    if c[1].button("Nominal check: no faults, should give 0 switches (~1.5 min)", disabled=busy or not CAL_OK):
        def nom():
            from experiments.run_nominal import nominal_check
            return ", ".join(f"{k}={v}" for k, v in nominal_check(log=lambda *_: None).items())
        start_jobs([("nominal check", nom)])
        st.rerun()
    if jobs()["done"] or jobs()["error"]:
        progress_table(jobs())

    st.subheader("Clear saved results")
    st.caption("Use this after changing code or thresholds, so old runs are not mixed with new ones.")
    sure = st.checkbox("Yes, delete all saved runs in results/")
    if st.button("Delete all saved runs", disabled=not sure or busy):
        for p in RESULTS.glob("*.csv"):
            p.unlink()
        st.cache_data.clear()
        st.success("Deleted.")


# ======================================================================= page router
st.title("Fault-Tolerant Adaptive AI Interface")
st.caption("Simulated edge node. Every button calls the real backend (src/framework). "
           "The framework itself is the orchestration layer; this page only drives and shows it.")
PAGES = {"1. Live demo": page_live, "2. Run experiments": page_experiments, "3. Results": page_results,
         "4. Run viewer": page_viewer, "5. Setup & calibration": page_setup}
if jobs_running():
    st.sidebar.info(f"Background: running {jobs()['current'] or '...'}")
if live_running():
    st.sidebar.info(f"Live system running (policy {live_holder()['policy']})")
PAGES[st.sidebar.radio("Page", list(PAGES))]()
