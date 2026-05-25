import cv2
import os
import pickle
import face_recognition


KNOWN_DIR = "known_faces"
DATA_FILE = "face_data.pkl"


def load_known_faces():
    """从 known_faces 目录加载已知人脸，支持增量缓存"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings, known_names = data.get("encodings", []), data.get("names", [])
    else:
        known_encodings, known_names = [], []

    existing_files = set()
    for name in os.listdir(KNOWN_DIR):
        person_dir = os.path.join(KNOWN_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            if filepath in existing_files:
                continue
            existing_files.add(filepath)

            img = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    if known_encodings:
        with open(DATA_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    return known_encodings, known_names


def register_face(frame, name):
    """从当前画面提取人脸并注册"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        print("未检测到人脸，请重试")
        return False

    person_dir = os.path.join(KNOWN_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    filepath = os.path.join(person_dir, f"{count + 1}.jpg")
    cv2.imwrite(filepath, frame)
    print(f"已注册 {name} -> {filepath}")

    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    return True


def main():
    os.makedirs(KNOWN_DIR, exist_ok=True)

    print("正在加载已知人脸...")
    known_encodings, known_names = load_known_faces()
    print(f"已加载 {len(known_encodings)} 张人脸")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("c-注册新人脸 | q-退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                name = "Unknown"
                color = (0, 0, 255)

                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                    if any(matches):
                        idx = matches.index(True)
                        name = known_names[idx]
                        color = (0, 255, 0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                name = input("输入姓名: ").strip()
                if name:
                    register_face(frame, name)
                    known_encodings, known_names = load_known_faces()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
