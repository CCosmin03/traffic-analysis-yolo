import cv2
import pytesseract
from ultralytics import YOLO

def ocr_plate_tess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(gray, config=cfg)
    return ''.join(filter(str.isalnum, text)).upper()


model = YOLO('runs/train/plates_custom_run/weights/best.pt')
cap   = cv2.VideoCapture('videos/traffic_vid3.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, iou=0.45)[0]
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]

        text = ocr_plate_tess(roi)
        #text = ocr_plate_easy(roi)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("OCR Test", cv2.resize(frame, (1400, 700)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
