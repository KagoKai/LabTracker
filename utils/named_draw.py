import cv2
import numpy as np

def draw_with_name(frame, person_box, name):
    x, y, w, h = person_box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
    return frame