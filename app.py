"""
Streamlit dashboard for the 3D-print defect monitor.

Real-time view of the camera feed with the YOLO detector overlay, per-class
detection status, and manual printer controls (pause / resume / stop). The same
src/detector.py (per-class thresholds + temporal smoother) used by main.py and
test_model.py drives the detections here, so the dashboard, the live loop, and the
offline tool all agree.

Run:
    streamlit run app.py

Live printer control is opt-in via the same env vars main.py uses:
    PRINTER_MODE=live PRINTER_URL=http://<giga-ip>:7125 streamlit run app.py
(Defaults to mock mode, safe to run with no printer attached.)

Design note — frame loop:
    Streamlit reruns the script top-to-bottom. We process ONE frame per rerun and
    schedule the next with st.rerun(). At the system's ~1-2 FPS analysis rate this is
    plenty, and unlike a blocking `while` loop it keeps the control buttons clickable.
    The Detector is cached (st.cache_resource) so its rolling temporal window persists
    across reruns.
"""

import time

import cv2
import streamlit as st

from src.detector import Detector, CLASS_THRESHOLDS, TEMPORAL_MIN_HITS, TEMPORAL_WINDOW
from src.printer_interface import PrinterInterface

# Per-class overlay colours (RGB, for st.image which expects RGB).
COLORS = {
    "Spaghetti":      (255, 0,   0  ),
    "Warping":        (255, 165, 0  ),
    "Layer_shifting": (255, 255, 0  ),
    "Stringing":      (0,   120, 255),
    "Cracking":       (0,   220, 0  ),
    "Blob_of_death":  (255, 0,   255),
}

ANALYSE_INTERVAL = 0.7  # seconds between analysed frames (~1.4 FPS)

st.set_page_config(page_title="3D Print Defect Monitor", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(model_path: str, min_hits: int) -> Detector:
    return Detector(model_path=model_path, min_hits=min_hits)


@st.cache_resource(show_spinner=False)
def get_printer() -> PrinterInterface:
    return PrinterInterface()


def draw(frame_bgr, detections):
    """Draw detection boxes on a BGR frame, return an RGB frame for st.image."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        rgb = COLORS.get(det["class_name"], (200, 200, 200))
        bgr = (rgb[2], rgb[1], rgb[0])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), bgr, 3)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(y1, th + 12)
        cv2.rectangle(frame_bgr, (x1, ly - th - 8), (x1 + tw + 6, ly + base - 4), bgr, -1)
        cv2.putText(frame_bgr, label, (x1 + 3, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def open_capture(source: str):
    """Open a cv2 capture from a webcam index (e.g. '0') or a file path."""
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


# --- session state -------------------------------------------------------------
ss = st.session_state
ss.setdefault("monitoring", False)
ss.setdefault("cap", None)
ss.setdefault("system_paused", False)   # True after an auto-pause, until acknowledged
ss.setdefault("last_event", "")

# --- sidebar: configuration + monitoring controls ------------------------------
st.sidebar.title("⚙️ Configuration")
source = st.sidebar.text_input("Camera source", value="0",
                               help="Webcam index (0, 1, …) or path to a video/image file.")
model_path = st.sidebar.text_input("Model weights", value=Detector.DEFAULT_MODEL)
min_hits = st.sidebar.slider("Trigger hits (of last %d frames)" % TEMPORAL_WINDOW,
                             1, TEMPORAL_WINDOW, TEMPORAL_MIN_HITS,
                             help="A defect class must appear in this many of the last "
                                  f"{TEMPORAL_WINDOW} analysed frames to trigger a pause.")

printer = get_printer()
st.sidebar.caption(f"Printer mode: **{printer._mode.upper()}**"
                   + (f" · {printer.firmware.upper()}" if printer._mode == "live" else ""))

c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", use_container_width=True, disabled=ss.monitoring):
    cap = open_capture(source)
    if cap is None:
        st.sidebar.error(f"Could not open source: {source}")
    else:
        ss.cap = cap
        ss.monitoring = True
        ss.system_paused = False
        ss.last_event = "Monitoring started."
        st.rerun()
if c2.button("⏹ Stop", use_container_width=True, disabled=not ss.monitoring):
    ss.monitoring = False
    if ss.cap is not None:
        ss.cap.release()
        ss.cap = None
    ss.last_event = "Monitoring stopped."
    st.rerun()

with st.sidebar.expander("Per-class thresholds"):
    for name, thr in CLASS_THRESHOLDS.items():
        st.write(f"{name}: **{thr:.2f}**")

# --- main layout ---------------------------------------------------------------
st.title("🖨️ 3D Print Defect Monitor")
feed_col, status_col = st.columns([3, 1])

with feed_col:
    frame_ph = st.empty()
with status_col:
    st.subheader("Status")
    state_ph = st.empty()
    alert_ph = st.empty()
    hits_ph = st.empty()

    st.subheader("Printer controls")
    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("⏸ Pause", use_container_width=True):
        ok = printer.pause_print()
        ss.last_event = "Pause sent ✓" if ok else "Pause failed ✗"
    if pc2.button("▶ Resume", use_container_width=True):
        ok = printer.resume_print()
        ss.system_paused = False
        ss.last_event = "Resume sent ✓" if ok else "Resume failed ✗"
    if pc3.button("⏹ Stop", use_container_width=True):
        ok = printer.cancel_print()
        ss.last_event = "Cancel sent ✓" if ok else "Cancel failed ✗"
    event_ph = st.empty()


def render_status(detector, detections):
    """Fill the status placeholders for the current frame."""
    printing = printer.is_printing()
    state_ph.metric("Printer", "PRINTING" if printing else "IDLE/PAUSED")

    if ss.system_paused:
        alert_ph.error("⛔ DEFECT CONFIRMED — printer auto-paused.")
    elif detector.triggered_classes:
        alert_ph.warning("⚠️ Defect persisting: " + ", ".join(detector.triggered_classes))
    else:
        alert_ph.success("✅ Clean")

    with hits_ph.container():
        if detector.class_hits:
            for cls, hits in sorted(detector.class_hits.items(), key=lambda kv: -kv[1]):
                st.write(f"**{cls}** {hits}/{min_hits}")
                st.progress(min(hits / min_hits, 1.0))
        else:
            st.caption("No defects in the current window.")
    if ss.last_event:
        event_ph.caption(ss.last_event)


# --- one analysed frame per rerun ----------------------------------------------
if ss.monitoring and ss.cap is not None:
    detector = load_detector(model_path, min_hits)
    detector.min_hits = min_hits  # honour live slider changes

    ret, frame = ss.cap.read()
    if not ret:
        # Video file ended → loop it so the demo keeps running; webcams won't hit this.
        ss.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = ss.cap.read()

    if ret:
        should_pause, detections = detector.trigger(frame)
        frame_ph.image(draw(frame.copy(), detections), use_container_width=True)
        render_status(detector, detections)

        if should_pause and not ss.system_paused and printer.is_printing():
            if printer.pause_print():
                ss.system_paused = True
                ss.last_event = f"AUTO-PAUSE: {', '.join(detector.triggered_classes)}"
                detector.reset()

        time.sleep(ANALYSE_INTERVAL)
        st.rerun()
    else:
        ss.monitoring = False
        frame_ph.warning("Stream ended / source unreadable.")
else:
    frame_ph.info("Configure a source in the sidebar and press **Start**.")
    state_ph.metric("Printer", "—")
