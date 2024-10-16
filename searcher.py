from argparse import ArgumentParser
import torch
import cv2
import os
from collections import deque
import numpy as np
import imutils


# Настройка аргументов командной строки
def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-v", "--video", help="путь к (необязательному) видеофайлу")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="максимальный размер буфера для траектории")
    ap.add_argument("-w", "--weights", type=str, default='yolov5n.pt', help="путь к файлу весов YOLOv5")
    ap.add_argument("-r", "--repo", type=str, default='yolov5', help="путь к локально клонированному репозиторию YOLOv5")
    ap.add_argument("-m", "--model", type=str, default='cpu', help="тип обработки cpu или gpu")
    return vars(ap.parse_args())

# Загрузка модели YOLOv5 из локального репозитория
def get_model(yolov5_repo, weight_path, device):
    try:
        # После загрузки модели
        model = torch.hub.load(yolov5_repo, 'custom', path=weight_path, source='local')
        
        # Перенос модели на GPU
        model.to('mps' if device == 'mps' and torch.backends.mps.is_available() else 'cpu')  
        print(f'Model use {device}')
        
    except Exception as e:
        print('Ошибка при загрузке модели YOLOv5:', e)
        exit()
    
    return model


def main():
    # Настройка аргументов командной строки
    args = get_arguments()
    
    # Настройка библиотеки удовлетворяющийх классов
    classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}

    # Проверка наличия файла весов
    weight_path = args["weights"]
    if not os.path.isfile(weight_path):
        print(f'Файл весов не найден по пути: {weight_path}')
        exit()

    # Загрузка модели YOLOv5 из локального репозитория
    model = get_model(args["repo"], args["weights"], args["model"])
    
    # Настройка для отслеживания траектории
    pts = deque(maxlen=args["buffer"])

    # Захват видео
    video_path = args.get("video", False)
    camera = cv2.VideoCapture(0 if not video_path else video_path)

    # Проверка успешности захвата видео
    if not camera.isOpened():
        print("Ошибка при открытии видеопотока.")
        exit()

    while True:
        # Захват текущего кадра
        grabbed, frame = camera.read()

        # Если видео и кадр не захвачен, выйти из цикла
        if video_path and not grabbed:
            break
        
        # Изменяем размер кадра
        frame = imutils.resize(frame, width=800)
        
        # Выполнение предсказания с помощью YOLOv5
        try:
            results = model(frame)

            # Получение результатов детекции
            detections = results.xyxy[0].cpu().numpy()  # Формат: [x1, y1, x2, y2, confidence, class]

            # Фильтрация детекций по классу и порогу доверия
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = model.names[int(cls)]
                
                # Проверка, соответствует ли класс целевому объекту
                if (class_name.lower() in classes) and conf >= 0.5:
                    # Вычисление центра объекта
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    center = (center_x, center_y)
                    radius = int(abs((y1 - y2) / 2))

                    # Добавление точки в очередь
                    pts.appendleft(center)

                    # Рисование прямоугольника вокруг объекта
                    cv2.circle(frame, center, radius, (0, 255, 255), 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Так как предполагается один объект, можно выйти из цикла детекций
                    break
            else:
                # Если объект не найден, добавляем None в очередь
                pts.appendleft(None)

        except Exception as e:
            print('Ошибка при обработке кадра с помощью YOLOv5:', e)
            pts.appendleft(None)

        # Отрисовка траектории
        for i in range(1, len(pts)):
            # Если обе точки определены, вычисляем толщину линии и рисуем соединяющую линию
            if pts[i - 1] is not None and pts[i] is not None:
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)        

        # Отображение кадра
        cv2.imshow("YOLOv5 Ball Detection and Tracking", frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()