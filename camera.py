import cv2
import time
from datetime import datetime

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头已打开: {width}x{height}, {fps:.0f}fps")

recording = False
writer = None
prev_time = time.time()
fps_display = 0

print("q-退出 | s-截图 | r-开始/停止录像")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    frame = cv2.flip(frame, 1)

    now = time.time()
    fps_display = 0.9 * fps_display + 0.1 / (now - prev_time) if now != prev_time else fps_display
    prev_time = now

    cv2.putText(frame, f"FPS: {fps_display:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if recording:
        cv2.putText(frame, "REC", (width - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(frame, (width - 20, 25), 8, (0, 0, 255), -1)
        writer.write(frame)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, frame)
        print(f"截图已保存: {filename}")
    elif key == ord('r'):
        if recording:
            recording = False
            writer.release()
            writer = None
            print("录像已停止")
        else:
            filename = datetime.now().strftime("video_%Y%m%d_%H%M%S.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            if writer.isOpened():
                recording = True
                print(f"开始录像: {filename}")
            else:
                print("无法创建录像文件")

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
