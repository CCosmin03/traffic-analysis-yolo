import cv2
from ultralytics import YOLO

# procentajul cu care reduci box-ul (1.0 = dimensiune originală, <1.0 = mai mic)
SHRINK = 0.7  

CLASS_COLORS = {
    'car':        ( 50,205, 50),
    'bus':        (  0,  0,255),
    'truck':      (255,  0,  0),
    'motorcycle': (255,165,  0),
}

def main():
    model = YOLO('runs/train/vehicles_custom_run2/weights/best.pt')
    cap   = cv2.VideoCapture('videos/traffic_vid1.mp4')
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Ajustează conf și iou după nevoie
        results = model(frame, conf=0.25, iou=0.70, max_det=200)[0]

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Calculează box-ul „micșorat” în jurul aceluiași centru
            cx, cy = x1 + w/2, y1 + h/2
            nw, nh = w * SHRINK, h * SHRINK
            x1s = int(cx - nw/2);  y1s = int(cy - nh/2)
            x2s = int(cx + nw/2);  y2s = int(cy + nh/2)

            label = f"{results.names[int(cls)]} {conf:.2f}"
            color = CLASS_COLORS.get(results.names[int(cls)], (0,255,0))

            # Desenează box-ul micșorat
            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), color, 2)

            # Fundal+text la fel
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1s, y1s-th-8), (x1s+tw+4, y1s), color, -1)
            cv2.putText(frame, label, (x1s+2, y1s-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Detectie YOLOv8 Antrenat", cv2.resize(frame, (1400,700)))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
