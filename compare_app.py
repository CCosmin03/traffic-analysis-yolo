# compare_app.py
import tempfile, cv2, streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model(path):
    return YOLO(path)

def draw_detections(frame, results, color):
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1,y1,x2,y2 = map(int, box)
        label = f"{results.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

st.set_page_config(layout="wide")
st.title("Comparatie YOLOv8‑nano vs. modelul meu")
st.write("🔴 YOLOv8n (pretrained) 🟢 Modelul meu")

video_file = st.file_uploader("Incarca un video", type=["mp4","avi"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000/fps)

    model_ext = load_model("yolov8n.pt")
    model_own = load_model("runs/train/vehicles_split_run/weights/best.pt")

    col1, col2 = st.columns(2)  # două coloane egale pe toată lățimea
    slot1 = col1.empty()
    slot2 = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        f_ext = frame.copy()
        f_own = frame.copy()

        r_ext = model_ext(f_ext, conf=0.25, iou=0.45)[0]
        r_own = model_own(f_own, conf=0.25, iou=0.45)[0]

        out_ext = draw_detections(f_ext, r_ext, (0,0,255))
        out_own = draw_detections(f_own, r_own, (0,255,0))

        # afișează imaginea ocupând toată lățimea coloanei
        slot1.image(cv2.cvtColor(out_ext, cv2.COLOR_BGR2RGB),
                    use_container_width=True)
        slot2.image(cv2.cvtColor(out_own, cv2.COLOR_BGR2RGB),
                    use_container_width=True)

    cap.release()
