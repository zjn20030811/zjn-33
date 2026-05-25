import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 q 退出，按 s 截图保存")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("截图已保存为 snapshot.jpg")

cap.release()
cv2.destroyAllWindows()
