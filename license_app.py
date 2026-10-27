import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import tempfile

# ---------- Configurare ----------
# Configurați calea către executabilul Tesseract dacă este necesar
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)

model = load_model('runs/train/plates_custom_run/weights/best.pt')

st.title("ANPR YOLOv8 Streamlit")
st.write("Detecție și recunoaștere plăcuțe de înmatriculare în video")

# Sidebar pentru configurare
def explain_threshold():
    st.sidebar.write(
        "**Threshold încredere YOLO**: nivelul minim de încredere (0–1) pe care detectarea YOLO trebuie să-l aibă pentru a fi afișată."
    )

explain_threshold()
conf_threshold = st.sidebar.slider(
    "Threshold încredere YOLO", 0.0, 1.0, 0.25
)

uploaded_file = st.file_uploader("Încarcă un fișier video (mp4, avi)", type=["mp4", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    run_button = st.button("Rulează detecția și OCR")
    placeholder = st.empty()

    if run_button:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detecție YOLO cu prag de încredere și IoU fixat
            results = model(frame, conf=conf_threshold, iou=0.45)[0]
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]

                # OCR cu Tesseract
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                cfg = (
                    "--oem 3 --psm 7 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                text = pytesseract.image_to_string(gray, config=cfg)
                text = ''.join(filter(str.isalnum, text)).upper()

                # Desenare boxă și text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2
                )

            # Conversie BGR->RGB și afișare
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame, channels="RGB", use_container_width=True)
        cap.release()
else:
    st.info("Încarcă un fișier video pentru procesare")
