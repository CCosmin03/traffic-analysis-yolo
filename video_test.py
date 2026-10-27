import cv2

def main():
    # Deschide camera implicită (0). Înlocuiește 0 cu calea către un fișier video, dacă vrei.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eroare: nu pot deschide camera.")
        return

    print("Camera deschisă. Apasă 'q' pentru a închide fereastra.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Eroare: nu pot citi frame-ul.")
            break

        # Afișează cadrul
        cv2.imshow('Frame', frame)

        # Așteaptă 1 ms; dacă s-a apăsat 'q', oprește
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
