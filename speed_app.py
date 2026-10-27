import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

st.set_page_config(page_title="Speed Detection Local", layout="wide")
st.title("Detecție viteză cu ROI-uri (video local sau upload)")

# Sidebar
conf_threshold = st.sidebar.slider("YOLO confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("YOLO IOU threshold", 0.0, 1.0, 0.70, 0.01)
start_btn = st.sidebar.button("Start procesare")
reset_btn = st.sidebar.button("Resetează")

uploaded_file = st.sidebar.file_uploader("Încarcă un video mic (mp4/avi)", type=["mp4", "avi"])
local_path = st.sidebar.text_input("Sau pune calea video locală", "videos/traffic_vid2.mp4")

# ROI-uri cu distanțe diferite
ROIs = [
    {"name": "ROI1", "pts": np.array([(460, 1300), (840, 800), (1520, 800), (1440, 1380)], np.int32), "color": (255, 0, 0), "distance": 12.0},
    {"name": "ROI2", "pts": np.array([(1660, 1160), (1660, 840), (2000, 840), (2100, 1160)], np.int32), "color": (0, 255, 0), "distance": 6.0},
    {"name": "ROI3", "pts": np.array([(2980, 1260), (3410, 1010), (3740, 1200), (3460, 1530)], np.int32), "color": (0, 165, 255), "distance": 6.0},
]

# Load model (caching)
@st.cache_resource
def load_vehicle_model(path: str):
    return YOLO(path)

veh_model = load_vehicle_model("runs/train/vehicles_custom_run2/weights/best.pt")

# Session state
if "entry_frame" not in st.session_state:
    st.session_state.entry_frame = {roi["name"]: {} for roi in ROIs}
if "exit_frame" not in st.session_state:
    st.session_state.exit_frame = {roi["name"]: {} for roi in ROIs}
if "speeds" not in st.session_state:
    st.session_state.speeds = {roi["name"]: {} for roi in ROIs}
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "fps" not in st.session_state:
    st.session_state.fps = None

if reset_btn:
    st.session_state.entry_frame = {roi["name"]: {} for roi in ROIs}
    st.session_state.exit_frame = {roi["name"]: {} for roi in ROIs}
    st.session_state.speeds = {roi["name"]: {} for roi in ROIs}
    st.session_state.frame_idx = 0
    st.session_state.fps = None
    st.success("Stare resetată.")

placeholder = st.empty()
info_placeholder = st.sidebar.empty()

def point_in_roi(pt, roi_pts):
    return cv2.pointPolygonTest(roi_pts, pt, False) >= 0

# Decide sursa video: upload sau local
video_source = None
if uploaded_file:
    tfile = st.sidebar.button("Folosește upload-ul")
    if tfile:
        vf = uploaded_file
        vid_path = f"temp_{uploaded_file.name}"
        with open(vid_path, "wb") as f:
            f.write(vf.read())
        video_source = vid_path
elif local_path:
    if os.path.exists(local_path):
        video_source = local_path
    else:
        st.sidebar.error("Fișierul local nu există.")

if video_source and start_btn:
    cap = cv2.VideoCapture(video_source)
    st.session_state.fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dt = 1.0 / st.session_state.fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        st.session_state.frame_idx += 1
        frame_idx = st.session_state.frame_idx

        results = veh_model.track(
            source=frame,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            iou=iou_threshold,
            persist=True,
            stream=True,
        )

        for result in results:  # one per frame
            display = result.orig_img.copy()

            # desen ROI-uri
            for roi in ROIs:
                cv2.polylines(display, [roi["pts"]], isClosed=True, color=roi["color"], thickness=2)
                cv2.putText(display, roi["name"], (roi["pts"][0][0], roi["pts"][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi["color"], 2)

            # procesare track-uri
            for box in result.boxes:
                tid = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                for roi in ROIs:
                    name = roi["name"]
                    inside = point_in_roi((cx, cy), roi["pts"])

                    if inside and tid not in st.session_state.entry_frame[name]:
                        st.session_state.entry_frame[name][tid] = frame_idx

                    if (not inside) and (tid in st.session_state.entry_frame[name]) and (tid not in st.session_state.exit_frame[name]):
                        st.session_state.exit_frame[name][tid] = frame_idx
                        elapsed = (st.session_state.exit_frame[name][tid] - st.session_state.entry_frame[name][tid]) * dt
                        if elapsed > 0:
                            distance_m = roi["distance"]
                            speed_m_s = distance_m / elapsed
                            speed_kmh = speed_m_s * 3.6
                            st.session_state.speeds[name][tid] = speed_kmh

                    # afișează când e în ROI
                    if (tid in st.session_state.entry_frame[name]) and (tid not in st.session_state.exit_frame[name]) and inside:
                        label = f"car{tid}@{name}"
                        cv2.rectangle(display, (x1, y1), (x2, y2), roi["color"], 2)
                        cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi["color"], 2)
                        cv2.circle(display, (cx, cy), 4, roi["color"], -1)

                # afișează după ieșire cu viteza
                for roi in ROIs:
                    name = roi["name"]
                    if tid in st.session_state.speeds[name]:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(display, f"car{tid}@{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        cv2.putText(display, f"{st.session_state.speeds[name][tid]:.1f} km/h", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # afișare resize
            resized = cv2.resize(display, (1400, 700))
            placeholder.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # sumar în sidebar
            summary = []
            for roi in ROIs:
                name = roi["name"]
                for tid, v in st.session_state.speeds[name].items():
                    summary.append(f"{name} car{tid}: {v:.1f} km/h")
            info_placeholder.text("\n".join(summary) if summary else "Nicio viteză calculată încă.")

    cap.release()
else:
    st.info("Fie încarcă un video mic prin uploader, fie furnizează o cale locală validă, apoi apasă Start procesare.")
