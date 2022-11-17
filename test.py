import os
import time

import cv2
import numpy as np

from src.dnnmodels import PersonDetector, FaceDetector
from src.recognizer import Recognizer
from utils.named_draw import draw_with_name

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

tracking_target = 'hieu'

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
        for person in people:
            box, _ = person
            x, y, w, h = box
            person_frame = frame[y : y + h, x : x + w]
            
            faces = f_detector.detect(person_frame)
            for face in faces:
                f_box, _ = face
                f_box = [a + b for a, b in zip(f_box, [x, y, 0, 0])]

                crop = recognizer.crop_face(frame, f_box)
                acc, name = recognizer.identify(CLF_MODEL, LABEL_JSON, crop)
                
                if name==tracking_target:
                    frame = draw_with_name(frame, box, name)
                    
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