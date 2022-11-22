import os
import time

import cv2
import numpy as np

from src.dnnmodels import PersonDetector, FaceDetector
from src.recognizer import Recognizer
from src.tracker import Tracker
from utils.box_limit import limited_box

PERSON_MODEL = os.path.join(os.curdir, 'Model', 'PersonDetect', 'yolov4-tiny-person.cfg')
PERSON_WEIGHTS = os.path.join(os.curdir, 'Model', 'PersonDetect', 'yolov4-tiny-person_last.weights')

FACE_MODEL = os.path.join(os.curdir, 'Model', 'FaceDetect', 'deploy.prototxt')
FACE_WEIGHTS = os.path.join(os.curdir, 'Model', 'FaceDetect', 'res10_300x300_ssd_iter_140000.caffemodel')

#Models dirs for the recognizer
EMBEDDER_MODEL = os.path.join(os.curdir, 'Model', 'FaceRecog', 'FaceNet_Keras_converted.h5')
CLF_MODEL = os.path.join(os.curdir, 'Model', 'svc_classifier.sav')
LABEL_JSON = os.path.join(os.curdir, 'Model', 'decode.json')

p_detector = PersonDetector(PERSON_WEIGHTS, PERSON_MODEL)
f_detector = FaceDetector(FACE_WEIGHTS, FACE_MODEL)
recognizer = Recognizer(EMBEDDER_MODEL)
tracker = Tracker('kcf')

tracking_target = 'hieu'
tracking = False
button_pressed = False

if __name__ == '__main__' :

    # Start default camera
    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0
    while True:
        ret, frame = cap.read()
        if (ret == False):
            break
        
        if not tracking:
            #Frame processing
            people = p_detector.detect(frame)
            for person in people:
                box, _ = person
                x1, y1, w, h = box
                x1, y1, x2, y2 = limited_box(x1, y1, x1 + w, y1 + h,)
                person_frame = frame[y1 : y2, x1 : x2]
                
                faces = f_detector.detect(person_frame)
                for face in faces:
                    f_box, _ = face
                    f_box = [a + b for a, b in zip(f_box, [x1, y1, 0, 0])]

                    crop = recognizer.crop_face(frame, f_box)
                    acc, name = recognizer.identify(CLF_MODEL, LABEL_JSON, crop)
                    
                    if name == tracking_target:
                        if button_pressed:
                            tracker.start_tracking(frame, (x1, y1, w, h))
                            tracking = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                    else:
                        cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'UNKNOWN', (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        else:
            ret, (x, y, w, h) = tracker.tracking(frame)
            if ret:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                cv2.putText(frame, 'ERROR !!!', (200, 20), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 4)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Live", frame)

        if (cv2.waitKey(1) & 0xff==ord('q')):
            button_pressed = True

    cap.release()