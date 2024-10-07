from argparse import ArgumentParser
from collections import deque
import numpy as np
import imutils
import cv2

# Создаем парсер аргументов и разбираем аргументы
ap = ArgumentParser()
ap.add_argument("-v", "--video", help="путь к (необязательному) видеофайлу")
ap.add_argument("-b", "--buffer", type=int, default=32, help="максимальный размер буфера")
args = vars(ap.parse_args())

# Определяем нижние и верхние границы зеленого шара в цветовом пространстве HSV, затем инициализируем список отслеживаемых точек
greenLower, greenUpper = (29, 86, 6), (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# Если путь к видео не был передан, захватываем ссылку на веб-камеру, иначе — ссылку на видеофайл
camera = cv2.VideoCapture(0 if not args.get("video", False) else args["video"])

# Продолжаем цикл
while True:
    # Захватываем текущий кадр
    grabbed, frame = camera.read()

    # Если мы просматриваем видео и не удалось захватить кадр, значит, мы дошли до конца видео
    if args.get("video") and not grabbed:
        break

    # Изменяем размер кадра, размываем его и конвертируем в цветовое пространство HSV
    frame = imutils.resize(frame, width=1080)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем маску для цвета "зелёный", затем выполняем серию дилатаций и эрозий, чтобы удалить любые мелкие артефакты, оставшиеся в маске
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Находим контуры в маске и инициализируем текущий центр шара (x, y)
    cnts, center = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], None
    
    # Продолжаем только если найден хотя бы один контур
    if len(cnts) > 0:
        # Находим самый большой контур в маске, затем используем его для вычисления минимальной описывающей окружности и центра масс
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moment = cv2.moments(c)
        center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

        # Продолжаем только если радиус достаточно велик
        if radius > 10:
            # Рисуем окружность и центр на кадре, затем обновляем список отслеживаемых точек
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Обновляем очередь точек
    pts.appendleft(center)

    # Проходим по списку отслеживаемых точек
    for i in range(1, len(pts)):
        # Если обе точки определены, вычисляем толщину линии и рисуем соединяющую линию
        if pts[i - 1] is not None and pts[i] is not None:
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Показываем кадр на экране
    cv2.imshow("Ball Tracking", frame)

    # Если нажата клавиша 'q', выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы камеры и закрываем все окна
camera.release()
cv2.destroyAllWindows()
