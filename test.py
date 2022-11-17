import os
import time

import cv2
import numpy as np

from src.dnnmodels import PersonDetector, FaceDetector

PERSON_MODEL = os.path.join(os.curdir, 'Model', 'PersonDetect', 'yolov4-tiny-person.cfg')
PERSON_WEIGHTS = os.path.join(os.curdir, 'Model', 'PersonDetect', 'yolov4-tiny-person_last.weights')

FACE_MODEL = os.path.join(os.curdir, 'Model', 'FaceDetect', 'deploy.prototxt')
FACE_WEIGHTS = os.path.join(os.curdir, 'Model', 'FaceDetect', 'res10_300x300_ssd_iter_140000.caffemodel')

p_detector = PersonDetector(PERSON_WEIGHTS, PERSON_MODEL)
f_detector = FaceDetector(FACE_WEIGHTS, FACE_MODEL)

if __name__ == '__main__' :
 
    # Start default camera
    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0
    while True:
        ret, frame = cap.read()
        if (ret == False):
            break
            
        #Frame processing
        people = p_detector.detect(frame)
        # for person in people:
        #     box, _ = person
        #     x, y, w, h = box
        #     person_frame = frame[y : y + h, x : x + w]
        #     face = f_detector.detect(person_frame)

        frame = p_detector.draw_boxes(frame, people)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Live", frame)

        if (cv2.waitKey(1) & 0xff==ord('q')):
            break

    cap.release()