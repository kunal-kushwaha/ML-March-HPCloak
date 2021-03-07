import cv2
import numpy as np

cap = cv2.VideoCapture(1)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', hsv)
        # BGR: np.unit8([[[0, 0, 255]]])
        # red = np.uint8([[[0, 0, 255]]])
        # hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', hsv_red)
        # print(hsv_red)
        # [H-10, 100, 100] and [H+10, 255, 255]

        l_red = np.array([0, 120, 120])
        u_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, l_red, u_red)
        # cv2.imshow('mask', mask)
        part1 = cv2.bitwise_and(back, back, mask=mask)
        # cv2.imshow('part1', part1)

        mask = cv2.bitwise_not(mask)
        # cv2.imshow('mask', mask)
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow('part2', part2)

        cv2.imshow('final', part1 + part2)

        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()