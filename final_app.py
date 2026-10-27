import streamlit as st
import tempfile
import cv2
import numpy as np
import os
from ultralytics import YOLO
import pytesseract
import math

# Tesseract config (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(layout="wide", page_title="Suite: Comparatie / ANPR / Viteza")

# Model loading (cached)
@st.cache_resource
def load_model(path):
    return YOLO(path)

@st.cache_resource
def load_vehicle_model_for_anpr():
    return YOLO('runs/train/vehicles_custom_run2/weights/best.pt')

@st.cache_resource
def load_plate_model():
    return YOLO('runs/train/plates_custom_run/weights/best.pt')

@st.cache_resource
def load_vehicle_model_for_speed():
    return YOLO("runs/train/vehicles_custom_run2/weights/best.pt")

# Utility
def draw_detections(frame, results, color):
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{results.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Stop flags
if "stop_compare" not in st.session_state:
    st.session_state.stop_compare = False
if "stop_anpr" not in st.session_state:
    st.session_state.stop_anpr = False
if "stop_speed" not in st.session_state:
    st.session_state.stop_speed = False

# Tabs
tabs = st.tabs(["1. Comparatie YOLO", "2. ANPR & Vehicule", "3. Detectie Viteza"])

# Tab 1: Comparatie YOLO
with tabs[0]:
    st.title("Comparatie YOLOv8-nano vs modelul meu")
    st.write("R  YOLOv8n (pretrained)    G  Modelul meu")
    video_file = st.file_uploader("Incarca un video (mp4/avi)", type=["mp4", "avi"], key="comp_video")
    start_btn = st.button("Porneste procesarea")
    stop_btn = st.button("Opreste procesarea")
    if stop_btn:
        st.session_state.stop_compare = True

    if video_file and start_btn:
        st.session_state.stop_compare = False
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        model_ext = load_model("yolov8n.pt")
        model_own = load_model("runs/train/vehicles_split_run/weights/best.pt")

        col1, col2 = st.columns(2)
        slot1 = col1.empty()
        slot2 = col2.empty()

        while cap.isOpened():
            if st.session_state.stop_compare:
                st.info("Procesarea a fost oprita."); break

            ret, frame = cap.read()
            if not ret:
                break

            f_ext = frame.copy()
            f_own = frame.copy()

            r_ext = model_ext(f_ext, conf=0.25, iou=0.45)[0]
            r_own = model_own(f_own, conf=0.25, iou=0.45)[0]

            out_ext = draw_detections(f_ext, r_ext, (0, 0, 255))
            out_own = draw_detections(f_own, r_own, (0, 255, 0))

            slot1.image(cv2.cvtColor(out_ext, cv2.COLOR_BGR2RGB), use_container_width=True)
            slot2.image(cv2.cvtColor(out_own, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
    elif not video_file:
        st.info("Incarca un video si apasa 'Porneste procesarea'.")
    else:
        st.info("Apasa butonul de start pentru a procesa video-ul.")

# Tab 2: ANPR si detectie vehicule
with tabs[1]:
    st.title("ANPR si Detectie Vehicule")
    st.write("Alege functionalitatile si incarca un video pentru procesare in timp real.")
    enable_vehicles = st.checkbox("Detecteaza vehicule", value=True)
    enable_plates = st.checkbox("Detecteaza placute", value=True)
    uploaded_file = st.file_uploader("Incarca un video (mp4, avi) pentru ANPR", type=["mp4", "avi"], key="anpr_video")

    start_anpr = st.button("Porneste ANPR")
    stop_anpr = st.button("Opreste ANPR")
    if stop_anpr:
        st.session_state.stop_anpr = True

    vehicle_model = load_vehicle_model_for_anpr()
    plate_model = load_plate_model()

    if uploaded_file and start_anpr:
        st.session_state.stop_anpr = False
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        placeholder = st.empty()

        while cap.isOpened():
            if st.session_state.stop_anpr:
                st.info("Procesarea ANPR a fost oprita."); break

            ret, frame = cap.read()
            if not ret:
                break

            if enable_vehicles:
                results_v = vehicle_model(frame, conf=0.25, iou=0.70, max_det=200)[0]
                for box, cls_id, conf in zip(results_v.boxes.xyxy,
                                              results_v.boxes.cls,
                                              results_v.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{results_v.names[int(cls_id)]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if enable_plates:
                results_p = plate_model(frame, conf=0.25, iou=0.45)[0]
                for box in results_p.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(
                        gray,
                        config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )
                    text = ''.join(filter(str.isalnum, text)).upper()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, text, (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()
    else:
        st.info("Incarca un video si apasa 'Porneste ANPR' in acest tab.")

# Tab 3: Detectie viteza
with tabs[2]:
    st.title("Detectie viteza cu ROI-uri")
    st.write("Video local sau upload, cu calculul vitezei pe ROI-uri definite.")

    start_btn_speed = st.button("Porneste procesarea viteza")
    stop_btn_speed = st.button("Opreste viteza")
    if stop_btn_speed:
        st.session_state.stop_speed = True
    reset_btn = st.button("Reseteaza viteza")
    uploaded_file_speed = st.file_uploader("Incarca un video mic (mp4/avi) pentru viteza", type=["mp4", "avi"], key="speed_video")
    local_path = st.text_input("Sau pune calea video locala", "videos/traffic_vid2.mp4")

    ROIs = [
        {"name": "ROI1", "pts": np.array([(460, 1300), (840, 800), (1520, 800), (1440, 1380)], np.int32), "color": (255, 0, 0), "distance": 12.0},
        {"name": "ROI2", "pts": np.array([(1660, 1160), (1660, 840), (2000, 840), (2100, 1160)], np.int32), "color": (0, 255, 0), "distance": 6.0},
        {"name": "ROI3", "pts": np.array([(2980, 1260), (3410, 1010), (3740, 1200), (3460, 1530)], np.int32), "color": (0, 165, 255), "distance": 6.0},
    ]

    veh_model_speed = load_vehicle_model_for_speed()

    if "entry_frame_speed" not in st.session_state:
        st.session_state.entry_frame_speed = {roi["name"]: {} for roi in ROIs}
    if "exit_frame_speed" not in st.session_state:
        st.session_state.exit_frame_speed = {roi["name"]: {} for roi in ROIs}
    if "speeds_speed" not in st.session_state:
        st.session_state.speeds_speed = {roi["name"]: {} for roi in ROIs}
    if "frame_idx_speed" not in st.session_state:
        st.session_state.frame_idx_speed = 0
    if "fps_speed" not in st.session_state:
        st.session_state.fps_speed = None

    if reset_btn:
        st.session_state.entry_frame_speed = {roi["name"]: {} for roi in ROIs}
        st.session_state.exit_frame_speed = {roi["name"]: {} for roi in ROIs}
        st.session_state.speeds_speed = {roi["name"]: {} for roi in ROIs}
        st.session_state.frame_idx_speed = 0
        st.session_state.fps_speed = None
        st.success("Stare resetata pentru detectia viteza.")

    placeholder_speed = st.empty()
    info_placeholder = st.empty()

    def point_in_roi(pt, roi_pts):
        return cv2.pointPolygonTest(roi_pts, pt, False) >= 0

    video_source = None
    if uploaded_file_speed:
        if st.button("Folosește upload-ul pentru viteza"):
            vf = uploaded_file_speed
            vid_path = f"temp_speed_{uploaded_file_speed.name}"
            with open(vid_path, "wb") as f:
                f.write(vf.read())
            video_source = vid_path
    elif local_path:
        if os.path.exists(local_path):
            video_source = local_path
        else:
            st.error("Fisierul local nu exista pentru viteza.")

    if video_source and start_btn_speed:
        st.session_state.stop_speed = False
        cap = cv2.VideoCapture(video_source)
        st.session_state.fps_speed = cap.get(cv2.CAP_PROP_FPS) or 30
        dt = 1.0 / st.session_state.fps_speed

        while cap.isOpened():
            if st.session_state.stop_speed:
                st.info("Procesarea viteza a fost oprita."); break

            ret, frame = cap.read()
            if not ret:
                break
            st.session_state.frame_idx_speed += 1
            frame_idx = st.session_state.frame_idx_speed

            results = veh_model_speed.track(
                source=frame,
                tracker="bytetrack.yaml",
                conf=0.25,
                iou=0.70,
                persist=True,
                stream=True,
            )

            for result in results:
                display = result.orig_img.copy()

                for roi in ROIs:
                    cv2.polylines(display, [roi["pts"]], isClosed=True, color=roi["color"], thickness=2)
                    cv2.putText(display, roi["name"], (roi["pts"][0][0], roi["pts"][0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi["color"], 2)

                for box in result.boxes:
                    tid = int(box.id)
                    coords = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    for roi in ROIs:
                        name = roi["name"]
                        inside = point_in_roi((cx, cy), roi["pts"])

                        if inside and tid not in st.session_state.entry_frame_speed[name]:
                            st.session_state.entry_frame_speed[name][tid] = frame_idx

                        if (not inside) and (tid in st.session_state.entry_frame_speed[name]) and (tid not in st.session_state.exit_frame_speed[name]):
                            st.session_state.exit_frame_speed[name][tid] = frame_idx
                            elapsed = (st.session_state.exit_frame_speed[name][tid] - st.session_state.entry_frame_speed[name][tid]) * dt
                            if elapsed > 0:
                                distance_m = roi["distance"]
                                speed_m_s = distance_m / elapsed
                                speed_kmh = speed_m_s * 3.6
                                st.session_state.speeds_speed[name][tid] = speed_kmh

                        if (tid in st.session_state.entry_frame_speed[name]) and (tid not in st.session_state.exit_frame_speed[name]) and inside:
                            label = f"car{tid}@{name}"
                            cv2.rectangle(display, (x1, y1), (x2, y2), roi["color"], 2)
                            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi["color"], 2)
                            cv2.circle(display, (cx, cy), 4, roi["color"], -1)

                    for roi in ROIs:
                        name = roi["name"]
                        if tid in st.session_state.speeds_speed[name]:
                            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(display, f"car{tid}@{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(display, f"{st.session_state.speeds_speed[name][tid]:.1f} km/h", (x1, y2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                resized = cv2.resize(display, (1400, 700))
                placeholder_speed.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                summary = []
                for roi in ROIs:
                    name = roi["name"]
                    for tid, v in st.session_state.speeds_speed[name].items():
                        summary.append(f"{name} car{tid}: {v:.1f} km/h")
                info_placeholder.text("\n".join(summary) if summary else "Nicio viteza calculata inca.")

        cap.release()
    else:
        st.info("Incarca un video pentru viteza si apasa 'Porneste procesarea viteza'.")