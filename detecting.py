import torch
import cv2
import argparse
import os

# Путь к локально клонированному репозиторию YOLOv5
yolov5_repo = 'yolov5'

# Путь к файлу весов модели
weight_path = '/Users/mr1necs/Documents/GitHub/Ball-tracker/yolov5/yolov5s.pt'

# Проверка наличия файла весов
if not os.path.isfile(weight_path):
    print(f'Файл весов не найден по пути: {weight_path}')
    exit()

# Загрузка модели YOLOv5 из локального репозитория    
try:
    model = torch.hub.load(yolov5_repo, 'custom', path=weight_path, source='local')
    
except Exception as e:
    print('Ошибка при загрузке модели:', e)
    exit()

# Создаем парсер аргументов и разбираем аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="путь к (необязательному) видеофайлу")
args = vars(ap.parse_args())

# Если путь к видео не был передан, захватываем ссылку на веб-камеру, иначе — ссылку на видеофайл
camera = cv2.VideoCapture(0 if not args.get("video", False) else args["video"])

while True:
    # Захватываем текущий кадр
    grabbed, frame = camera.read()
    
    # Если мы просматриваем видео и не удалось захватить кадр, значит, мы дошли до конца видео
    if args.get("video") and not grabbed:
        break

    try:
        results = model(frame)

        # Обработка результатов
        results.render()

        # Отображение кадра
        cv2.imshow('Ball Detection', frame)
        
    except Exception as e:
        print('Ошибка при обработке кадра:', e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()