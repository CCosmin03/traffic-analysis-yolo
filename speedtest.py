import cv2
import numpy as np
from ultralytics import YOLO
import math

# ---------- CONFIGURARE ROI-URI ----------
# Fiecare ROI are un nume, punctele și o culoare
ROIs = [
    {
        "name": "ROI1",
        "pts": np.array([(460, 1300), (840, 800), (1520, 800), (1440, 1380)], dtype=np.int32),
        "color": (255, 0, 0),  # albastru
    },
    {
        "name": "ROI2",
        "pts": np.array([(1660, 1160), (1660, 840), (2000, 840), (2100, 1160)], dtype=np.int32),
        "color": (0, 255, 0),  # verde
    },
    {
        "name": "ROI3",
        "pts": np.array([(2980, 1260), (3410, 1010), (3740, 1200), (3460, 1530)], dtype=np.int32),
        "color": (0, 165, 255),  # portocaliu
    },
]

# Distanța reală pentru toate ROI-urile (poți ajusta per ROI dacă vrei)
ROI_DISTANCES = {
    "ROI1": 12.0,
    "ROI2": 6.0,
    "ROI3": 6.0,
}

# ---------- ÎNCĂRCARE MODEL CU TRACKING YOLOv8 ----------
veh_model = YOLO('runs/train/vehicles_custom_run2/weights/best.pt')

# ---------- VIDEO ȘI FPS ----------
source = 'videos/traffic_vid2.mp4'
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
dt = 1.0 / fps

# ---------- STRUCTURI DE URMĂRIRE ----------
# Pentru fiecare ROI: track_id -> frame intrare / ieșire / speed
entry_frame = {roi["name"]: {} for roi in ROIs}
exit_frame = {roi["name"]: {} for roi in ROIs}
speeds = {roi["name"]: {} for roi in ROIs}

frame_idx = 0

# helper: verifică dacă un punct e în interiorul unui poligon ROI
def point_in_roi(pt, roi_pts):
    return cv2.pointPolygonTest(roi_pts, pt, False) >= 0

# ---------- STREAMING CU TRACK YOLOv8 ----------
track_stream = veh_model.track(
    source=source,
    tracker='bytetrack.yaml',
    conf=0.25,
    iou=0.70,
    persist=True,
    stream=True
)

for result in track_stream:
    frame_idx += 1
    frame = result.orig_img.copy()

    # desenează fiecare ROI
    for roi in ROIs:
        cv2.polylines(frame, [roi["pts"]], isClosed=True, color=roi["color"], thickness=2)
        cv2.putText(
            frame,
            roi["name"],
            (roi["pts"][0][0], roi["pts"][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            roi["color"],
            2,
        )

    # procesează obiectele urmărite
    for box in result.boxes:
        tid = int(box.id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # verifică pentru fiecare ROI
        for roi in ROIs:
            name = roi["name"]
            inside = point_in_roi((cx, cy), roi["pts"])

            # intrare
            if inside and tid not in entry_frame[name]:
                entry_frame[name][tid] = frame_idx

            # ieșire: era în ROI și acum nu mai este
            if (not inside) and (tid in entry_frame[name]) and (tid not in exit_frame[name]):
               exit_frame[name][tid] = frame_idx
               elapsed = (exit_frame[name][tid] - entry_frame[name][tid]) * dt
               if elapsed > 0:
                  distance_m = ROI_DISTANCES.get(name, 12.0)  # fallback
                  speed_m_s = distance_m / elapsed
                  speed_kmh = speed_m_s * 3.6
                  speeds[name][tid] = speed_kmh

            # afișează vehiculul dacă e în ROI (și nu a ieșit încă)
            if (tid in entry_frame[name]) and (tid not in exit_frame[name]) and inside:
                label = f"car{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), roi["color"], 2)
                cv2.putText(frame, label + f" @{name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi["color"], 2)
                cv2.circle(frame, (cx, cy), 4, roi["color"], -1)

        # afișează vehiculul după ce a ieșit din oricare ROI cu viteza lui pentru fiecare ROI în care a traversat
        for roi in ROIs:
            name = roi["name"]
            if tid in speeds[name]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, f"car{tid} @{name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"{speeds[name][tid]:.1f} km/h", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow('Multiple ROI Speed Detection', cv2.resize(frame, (1400,700)))
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# la final, afișează toate vitezele
print("Viteze pe ROI-uri:")
for roi in ROIs:
    name = roi["name"]
    for tid, v in speeds[name].items():
        print(f"{name} - car{tid}: {v:.1f} km/h")
