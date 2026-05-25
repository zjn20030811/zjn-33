import cv2
import time
from datetime import datetime

# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开: {width}x{height}")

    recording = False
    writer = None
    paused = False
    show_help = True
    prev_time = time.time()
    fps_display = 0

    print("h-帮助 | 空格-暂停 | s-截图 | r-录像 | q-退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面")
                break

            if not paused:
                frame = cv2.flip(frame, 1)

                # 人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # FPS
                now = time.time()
                fps_display = 0.9 * fps_display + 0.1 / (now - prev_time) if now != prev_time else fps_display
                prev_time = now

                if recording:
                    writer.write(frame)
            else:
                frame = frame.copy()

            # HUD 叠加层
            draw_hud(frame, fps_display, recording, paused, show_help, width, len(faces))

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s') and not paused:
                filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(filename, frame)
                print(f"截图已保存: {filename}")
            elif key == ord('r') and not paused:
                recording, writer = toggle_recording(recording, writer, width, height)
            elif key == ord(' '):
                paused = not paused
                print("已暂停" if paused else "已恢复")
            elif key == ord('h'):
                show_help = not show_help

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("摄像头已释放")


def draw_hud(frame, fps_display, recording, paused, show_help, width, face_count):
    cv2.putText(frame, f"FPS: {fps_display:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if recording:
        cv2.putText(frame, "REC", (width - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(frame, (width - 20, 25), 8, (0, 0, 255), -1)

    if paused:
        cv2.putText(frame, "PAUSED", (width // 2 - 60, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if show_help:
        cv2.putText(frame, "h:hide  space:pause  s:shot  r:rec  q:quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def toggle_recording(recording, writer, width, height):
    if recording:
        writer.release()
        print("录像已停止")
        return False, None
    else:
        filename = datetime.now().strftime("video_%Y%m%d_%H%M%S.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        if writer.isOpened():
            print(f"开始录像: {filename}")
            return True, writer
        else:
            print("无法创建录像文件")
            return False, None


if __name__ == "__main__":
    main()
