import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

VIDEO_SOURCE = 'videos/ped_vid3.mov'
MODEL_PATH   = 'runs/train/pedestrian_custom_run3/weights/best.pt'
CONF_THRESH  = 0.5
IOU_THRESH   = 0.5

model = YOLO(MODEL_PATH)

# Dicționar pt. trasee și culori per ID
trails = defaultdict(list)
colors = {}  # track_id -> (B, G, R)

cap = cv2.VideoCapture(VIDEO_SOURCE)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Tracking cu ByteTrack
    results = model.track(
        frame,
        tracker='bytetrack.yaml',
        persist=True,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        stream=False
    )

    r = results[0]
    xyxy = r.boxes.xyxy
    if xyxy is None or len(xyxy) == 0:
        cv2.imshow('ByteTrack + Trasee Color', cv2.resize(frame, (1200, 500)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = xyxy.cpu().numpy().astype(int)
    ids_tensor = getattr(r.boxes, 'id', None)
    if ids_tensor is None:
        cv2.imshow('ByteTrack + Trasee Color', cv2.resize(frame, (1200, 500)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    ids = ids_tensor.cpu().numpy().astype(int)

    # 2) Desenam boxes si salvam centre + culoare per ID
    for (x1, y1, x2, y2), tid in zip(boxes, ids):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        trails[tid].append((cx, cy))

        if tid not in colors:
            # culori random reproducibile per tid
            rng = np.random.RandomState(int(tid))
            colors[tid] = tuple(map(int, rng.randint(0, 256, size=3)))

        color = colors[tid]
        # desenam bounding-box si ID cu culoarea aferentă
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID:{tid}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3) Desenam traseele fiecarui ID cu aceeasi culoare
    for tid, pts in trails.items():
        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [pts_arr], False, colors[tid], 2)
            # punctul curent
            cv2.circle(frame, pts_arr[-1], 4, colors[tid], -1)

    cv2.imshow('ByteTrack + Trasee Color', cv2.resize(frame, (1200, 500)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
