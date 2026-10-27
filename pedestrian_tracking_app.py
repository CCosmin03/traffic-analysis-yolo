import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile, os

st.title('Real-Time Pedestrian Tracking')

uploaded = st.file_uploader("Upload a video", type=['mp4','avi','webm'])
local_path = st.text_input("…or enter a local video path")
video_source = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read())
    video_source = tmp.name
elif local_path:
    video_source = local_path

if video_source:
    @st.cache_resource
    def load_model():
        return YOLO('runs/train/pedestrian_custom_run3/weights/best.pt')
    model = load_model()

    cap = cv2.VideoCapture(video_source)
    placeholder = st.empty()

    trails = defaultdict(list)
    colors = {}
    CONF, IOU = 0.5, 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            tracker='bytetrack.yaml',
            persist=True,
            conf=CONF,
            iou=IOU,
            stream=False
        )
        r = results[0]
        if r.boxes is not None and len(r.boxes.xyxy) and getattr(r.boxes, 'id', None) is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids   = r.boxes.id.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), tid in zip(boxes, ids):
                cx, cy = (x1+x2)//2, (y1+y2)//2
                trails[tid].append((cx, cy))
                if tid not in colors:
                    rng = np.random.RandomState(int(tid))
                    colors[tid] = tuple(map(int, rng.randint(0,256,3)))
                c = colors[tid]
                cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
                cv2.putText(frame, f'ID:{tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        for tid, pts in trails.items():
            if len(pts) > 1:
                arr = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [arr], False, colors[tid], 2)
                cv2.circle(frame, tuple(arr[-1]), 4, colors[tid], -1)

        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
else:
    st.info("Upload a video or enter a local path to start.")
