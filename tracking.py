from collections import deque
import numpy as np
import argparse
import imutils
import cv2


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

# Define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
greenLower, greenUpper = (29, 86, 6), (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# If a video path was not supplied, grab the reference to the webcam otherwise, grab a reference to the video file
camera = cv2.VideoCapture(0 if not args.get("video", False) else args["video"])


# Keep looping
while True:
    # Grab the current frame
    (grabbed, frame) = camera.read()

    # If we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # Resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=1080)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Construct a mask for the color "green", then perform a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    cnts, center = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], None
    
    # Only proceed if at least one contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moment = cv2.moments(c)
        center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Update the points queue
    pts.appendleft(center)

    # Loop over the set of tracked points
    for i in range(1, len(pts)):
        # Если обе точки определены, вычисляем толщину линии и рисуем соединяющую линию
        if pts[i - 1] is not None and pts[i] is not None:
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
