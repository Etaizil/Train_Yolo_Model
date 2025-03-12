import cv2

video_path = "C:\\Users\\etai4\\Desktop\\Computer Science\\experimental\\yolo\\WIN_20250313_00_28_47_Pro.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Failed to open video file.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(50) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()