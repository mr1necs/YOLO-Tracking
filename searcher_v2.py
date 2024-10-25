from argparse import ArgumentParser
from ultralytics import YOLO
from collections import deque
from torch.backends import mps
from torch import cuda
import numpy as np
import cv2
import imutils


# Настройка аргументов командной строки
def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default='mps', help="тип обработки cpu или mps")
    ap.add_argument("-v", "--video", help="путь к (необязательному) видеофайлу")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="максимальный размер буфера для траектории")
    return vars(ap.parse_args())


# Загрузка модели
def get_model(device):
    model = YOLO('yolo11n.pt')
    device = (
        'mps' if device == 'mps' and mps.is_available() else
        'cuda' if device == 'cuda' and cuda.is_available() else
        'cpu'
    )
    model.to(device)
    print(f'Model use {device}')

    return model


# Захват видео
def get_video(video_path=None):
    camera = cv2.VideoCapture(0 if not video_path else video_path)
    if not camera.isOpened():
        print("Ошибка при открытии видеопотока.")
        exit()

    return camera


# Обработка детекций
def process_detections(model, classes, pts, frame):
    results = model(frame)

    for r in results:
        detections = r.boxes

        for det in detections:
            xyxy = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = int(det.cls.cpu().numpy()[0])
            class_name = model.names[cls]

            if class_name.lower() in classes and conf >= 0.3:
                x1, y1, x2, y2 = xyxy
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)
                radius = int(abs((y1 - y2) / 2))

                pts.appendleft(center)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                break
        else:
            pts.appendleft(None)



# Отрисовка траектории
def draw_trace(frame, pts):
    for i in range(1, len(pts)):
        if pts[i - 1] is not None and pts[i] is not None:
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


# Тело основной программы
def main():
    args = get_arguments()
    model = get_model(args["model"])
    camera = get_video(args["video"])
    pts = deque(maxlen=args["buffer"])  # Буфер для отслеживания траектории

    classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}  # Нужные классы

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=800)

        process_detections(model, classes, pts, frame)
        draw_trace(frame, pts)

        # Отображаем кадр
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()