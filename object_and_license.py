import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

# ---------- CONFIGURĂRI ----------
# Calea către tesseract (dacă nu e în PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cât „micșorăm” cutiile de vehicule (pentru a evita marginile)
SHRINK = 0.7

# Culori per clasă vehicul
CLASS_COLORS = {
    'car':        (50,205, 50),
    'bus':        (  0,  0,255),
    'truck':      (255,  0,  0),
    'motorcycle': (255,165,  0),
}

# OCR simplu cu Tesseract
def ocr_plate_tess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(gray, config=cfg)
    return ''.join(filter(str.isalnum, text)).upper()

def main():
    # 1) Încarcă modelele
    veh_model   = YOLO('runs/train/vehicles_custom_run2/weights/best.pt')
    plate_model = YOLO('runs/train/plates_custom_run/weights/best.pt')

    # 2) Deschide video
    cap   = cv2.VideoCapture('videos/traffic_vid3.mp4')
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. Detectează vehicule ---
        results_veh = veh_model(frame, conf=0.25, iou=0.70, max_det=200)[0]

        for box, cls_id, conf in zip(results_veh.boxes.xyxy,
                                      results_veh.boxes.cls,
                                      results_veh.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2, y1 + h/2
            nw, nh = w * SHRINK, h * SHRINK
            x1s = int(cx - nw/2);  y1s = int(cy - nh/2)
            x2s = int(cx + nw/2);  y2s = int(cy + nh/2)

            label = f"{results_veh.names[int(cls_id)]} {conf:.2f}"
            color = CLASS_COLORS.get(results_veh.names[int(cls_id)], (0,255,0))

            # Desenează vehiculul
            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1s, y1s-th-8), (x1s+tw+4, y1s), color, -1)
            cv2.putText(frame, label, (x1s+2, y1s-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # --- 2. În interiorul fiecărui vehicul, caută plăcuțe ---
            veh_roi = frame[y1s:y2s, x1s:x2s]
            plate_res = plate_model(veh_roi, conf=0.25, iou=0.45)[0]
            for pbox in plate_res.boxes.xyxy:
                px1, py1, px2, py2 = map(int, pbox)
                # Ajustează coordonatele la întregul cadru
                gx1, gy1 = px1 + x1s, py1 + y1s
                gx2, gy2 = px2 + x1s, py2 + y1s

                # OCR pe plăcuță
                plate_roi = frame[gy1:gy2, gx1:gx2]
                text = ocr_plate_tess(plate_roi)

                # Desenare plăcuță + text
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0,255,255), 2)
                cv2.putText(frame, text, (gx1, gy2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow("Vehicule + ANPR Combinat", cv2.resize(frame, (1400,800)))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
