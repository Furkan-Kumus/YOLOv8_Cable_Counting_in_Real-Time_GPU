import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(0)  # For Video
cap.set(3, 1920)
cap.set(4, 1080)

model = YOLO("../Yolo-Weights/best.pt")

classNames = ['Black_Cable']

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [1470, 550, 1470, 1080]
limits_ = [1530, 550, 1530, 1080]
limit = [1430, 550, 1430, 1080]
limit_ = [1570, 550, 1570, 1080]
totalCount = []
nok_counter = 0

limits2 = [1470, 0, 1470, 530]
limits2_ = [1530, 0, 1530, 530]
limit2 = [1430, 0, 1430, 530]
limit2_ = [1570, 0, 1570, 530]
totalCount2 = []
ok_counter = 0

nok_control = False
ok_control = False

nok_control2 = False
ok_control2 = False


while True:
    success, img = cap.read()

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "Black_Cable" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits_[0], limits_[1]), (limits_[2], limits_[3]), (0, 0, 255), 5)
    #cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (200, 0, 255), 5)
    #cv2.line(img, (limit_[0], limit_[1]), (limit_[2], limit_[3]), (200, 0, 255), 5)

    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)
    cv2.line(img, (limits2_[0], limits2_[1]), (limits2_[2], limits2_[3]), (0, 255, 0), 5)
    # cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (200, 0, 255), 5)
    # cv2.line(img, (limit2_[0], limit2_[1]), (limit2_[2], limit2_[3]), (200, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)} {currentClass} {conf}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # -------------------------------------------NOK-------------------------------------------------------------------------------------------------

        # NOK ARTAN SAYAC
        if limits[1] < cy < limits[3] and limits[0] - 15 < cx < limits[0] + 15:
            nok_control = True

        if limit[1] < cy < limit[3] and limit[0] - 15 < cx < limit[0] + 15:
            nok_control = False

        if nok_control and limits_[1] < cy < limits_[3] and limits_[0] - 15 < cx < limits_[0] + 15:
            nok_control = False
            nok_counter += 1  # Sayaç artırılır
            cv2.line(img, (limits_[0], limits_[1]), (limits_[2], limits_[3]), (200, 200, 0), 5)

        # NOK AZALAN SAYAC
        if limits_[1] < cy < limits_[3] and limits_[0] - 15 < cx < limits_[0] + 15:
            nok_control2 = True

        if limit_[1] < cy < limit_[3] and limit_[0] - 15 < cx < limit_[0] + 15:
            nok_control2 = False

        if nok_control2 and limits[1] < cy < limits[3] and limits[0] - 15 < cx < limits[0] + 15:
            nok_control2 = False
            nok_counter -= 1  # Sayaç azaltılır
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (200, 200, 0), 5)

        # ---------------------------------------------OK-----------------------------------------------------------------------------------------------

        # OK ARTAN SAYAC
        if limits2[1] < cy < limits2[3] and limits2[0] - 15 < cx < limits2[0] + 15:
            ok_control = True

        if limit2[1] < cy < limit2[3] and limit2[0] - 15 < cx < limit2[0] + 15:
            ok_control = False

        if ok_control and limits2_[1] < cy < limits2_[3] and limits2_[0] - 15 < cx < limits2_[0] + 15:
            ok_control = False
            ok_counter += 1  # Sayaç artırılır
            cv2.line(img, (limits2_[0], limits2_[1]), (limits2_[2], limits2_[3]), (200, 200, 0), 5)


        # OK AZALAN SAYAC
        if limits2_[1] < cy < limits2_[3] and limits2_[0] - 15 < cx < limits2_[0] + 15:
            ok_control2 = True

        if limit2_[1] < cy < limit2_[3] and limit2_[0] - 15 < cx < limit2_[0] + 15:
            ok_control2 = False

        if ok_control2 and limits2[1] < cy < limits2[3] and limits2[0] - 15 < cx < limits2[0] + 15:
            ok_control2 = False
            ok_counter -= 1  # Sayaç azaltılır
            cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (200, 200, 0), 5)

    # --------------------------------------------------------------------------------------------------------------------------------------------

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, f'NOK Counter {str(nok_counter)}', (50, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5)
    cv2.putText(img, f'OK Counter {str(ok_counter)}', (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)