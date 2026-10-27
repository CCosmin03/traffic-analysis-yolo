import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import tempfile

# ---------- CONFIGURARE Tesseract ----------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------- Încărcare modele ----------
@st.cache_resource
def load_vehicle_model():
    return YOLO('runs/train/vehicles_custom_run2/weights/best.pt')

@st.cache_resource
def load_plate_model():
    return YOLO('runs/train/plates_custom_run/weights/best.pt')

vehicle_model = load_vehicle_model()
plate_model   = load_plate_model()

st.title("ANPR și Detecție Vehicule Streamlit")
st.write("Alege funcționalitățile și încarcă un video pentru procesare în timp real.")

# --- Sidebar controale ---
enable_vehicles = st.sidebar.checkbox("Detectează vehicule", value=True)
enable_plates   = st.sidebar.checkbox("Detectează plăcuțe", value=True)

uploaded_file = st.sidebar.file_uploader("Încarcă un video (mp4, avi)", type=["mp4", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesare vehicule
        if enable_vehicles:
            results_v = vehicle_model(frame, conf=0.25, iou=0.70, max_det=200)[0]
            for box, cls_id, conf in zip(results_v.boxes.xyxy,
                                          results_v.boxes.cls,
                                          results_v.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"{results_v.names[int(cls_id)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Procesare plăcuțe
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, text, (x1, y2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # Conversie BGR->RGB și afișare
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("Aștept un video pentru procesare.")
