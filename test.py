import os
import time

import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

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

tolerance = 50
x_deviation = 0
y_deviation = 0
track_data = [0, 0, 0, 0, 0, 0]
frame_width = 640
frame_height = 480

def main():
    global x_deviation, y_deviation, tolerance, button_pressed
    #---------------Initialize the devices--------------- 
    openni2.initialize()    
    dev = openni2.Device.open_any()
    global tracking
    # Start the depth camera
    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, 
                                resolutionX = 640, resolutionY = 480, fps = 30))
    depth_stream.start()


    # Start the default camera
    cap = cv2.VideoCapture(1)

    objs = list()

    while True:
        objs.clear()
        start_time = time.time()
        ret, frame = cap.read()
        if (ret == False):
            break

        frame_height, frame_width, _ = frame.shape
        frame_x_center = round(frame_width/2, 1)
        frame_y_center = round(frame_height/2, 1)

        # Grab a new depth frame
        dframe = depth_stream.read_frame()
        frame_data = dframe.get_buffer_as_uint16()
        dmap = np.frombuffer(frame_data, dtype=np.uint16)
        dmap.shape = (1, 480, 640)
        dmap = np.swapaxes(dmap, 0, 2)
        dmap = np.swapaxes(dmap, 0, 1)
        dmap = np.fliplr(dmap)

        #---------------Inference---------------
        if not tracking:
            # Detect people in the current frame
            people = p_detector.detect(frame)
            for person in people:
                box, _ = person
                x1, y1, w, h = box
                x1, y1, x2, y2 = limited_box(x1, y1, x1 + w, y1 + h,)
                # Cut the bounding boxs around people (if there's any) into person frames
                person_frame = frame[y1 : y2, x1 : x2]
                
                # Detect the face in the person frames
                faces = f_detector.detect(person_frame)
                for face in faces:
                    f_box, _ = face
                    # Offset by the upperleft corner of the person frame to get the corresponding
                    # location in the camera frame
                    f_box = [a + b for a, b in zip(f_box, [x1, y1, 0, 0])]

                    crop = recognizer.crop_face(frame, f_box)
                    acc, name = recognizer.identify(CLF_MODEL, LABEL_JSON, crop)
                    
                    if name == tracking_target:
                        if button_pressed:
                            tracker.start_tracking(frame, (x1, y1, w, h))
                            tracking = True

                        objs.append([name, f_box])

                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #cv2.putText(frame, name, (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                    else:
                        objs.append(['UNKNOWN', f_box])
                        
                        #cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 0, 255), 2)
                        #cv2.putText(frame, 'UNKNOWN', (x1, y1+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        else:
            ret, (x, y, w, h) = tracker.tracking(frame)
            if ret:
                obj_x_center = int(x + w/2)
                obj_y_center = int(y + h/2)
                x_deviation = frame_x_center - obj_x_center
                y_deviation = frame_y_center - obj_y_center

                depth = dmap[int(obj_x_center), int(obj_y_center)]
                cv2.putText(frame, f'{depth*100e-6}m', (x, y + 20), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

                track_data[0] = obj_x_center
                track_data[1] = obj_y_center
                track_data[2] = x_deviation
                track_data[3] = y_deviation
                track_data[4], track_data[5] = move_command()

                objs.append([tracking_target, [x, y, w, h]])
            else:
                cv2.putText(frame, 'ERROR !!!', (100, 240), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 4)
                track_data[4] = 0

        end_time = time.time()
        duration = end_time - start_time

        button_pressed = False

        frame = draw_overlays(frame, objs, duration, track_data, tracking)
        cv2.imshow('Giao dien', frame)

        if (cv2.waitKey(1) & 0xff==ord('q')):
            button_pressed = True
        
        if (cv2.waitKey(1) & 0xff==ord('c')):
            break
    cap.release()

def move_command():
    #Output direction for the robot, possible value: Stop/Left/Right
    global x_deviation, y_deviation, tolerance
    speed = 0

    if (abs(x_deviation) < tolerance and abs(y_deviation) < tolerance):
        cmd = 'Stop'
    elif (abs(x_deviation)>abs(y_deviation)):
        if(x_deviation>=tolerance):
            cmd="Move Left"
        if(x_deviation<=-1*tolerance):
            cmd="Move Right"

        direction = 'lr'
        speed = speed_command(x_deviation, direction)

    else:
        if(y_deviation>=tolerance):
            cmd="Move Forward"   
        if(y_deviation<=-1*tolerance):
            cmd="Move Backward"

        direction = 'fb'
        speed = speed_command(y_deviation, direction)

    return cmd, speed

def speed_command(deviation, direction):
    #Output speed command based on direction and deviation from the frame center point
    global x_deviation, y_deviation, tolerance

    deviation=abs(deviation)
    if (direction == 'fb'):
        if(deviation >= 0.8 * frame_height/2):
            speed = 100
        elif(deviation >= 0.6 * frame_height/2):
            speed = 75
        elif(deviation >= 0.4 * frame_height/2):
            speed = 45
        else:
            speed = 35
    elif (direction == 'lr'):
        if(deviation >= 0.8 * frame_width/2):
            speed = 80
        elif(deviation >= 0.7 * frame_width/2):
            speed = 70
        elif(deviation >= 0.6 * frame_width/2):
            speed = 60
        elif(deviation >= 0.5 * frame_width/2):
            speed = 50
        elif(deviation >= 0.4 * frame_width/2):
            speed = 40
        else:
            speed = 30
    
    return speed

def draw_overlays(image, objs, duration, track_data, tracking):
    global x_deviation, y_deviation, tolerance

    height, width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #draw black rectangle on top
    image = cv2.rectangle(image, (0,0), (width, 24), (0,0,0), -1)
    
     
    #write processing durations
    processing_time = round(duration*1000, 0)
    text_dur = f'Processing time: {processing_time}ms'
    image = cv2.putText(image, text_dur, (int(width/4)-30, 16), font, 0.4, (255, 255, 255), 1)
    
    #write FPS 
    fps=round(1/duration, 1)
    text1 = 'FPS: {}'.format(fps)
    image = cv2.putText(image, text1, (10, 20),font, 0.7, (150, 150, 255), 2)
   
    
    #draw black rectangle at bottom
    image = cv2.rectangle(image, (0,height-24), (width, height), (0,0,0), -1)
    
    if tracking:
        #write deviations and tolerance
        str_tol='Tol : {}'.format(tolerance)
        image = cv2.putText(image, str_tol, (10, height-8),font, 0.55, (150, 150, 255), 2)
    
    
        x_dev = track_data[2]
        str_x='X: {}'.format(x_dev)
        if(abs(x_dev)<tolerance):
            color_x=(0,255,0)
        else:
            color_x=(0,0,255)
        image = cv2.putText(image, str_x, (110, height-8),font, 0.55, color_x, 2)
        
        y_dev = track_data[3]
        str_y='Y: {}'.format(y_dev)
        if(abs(y_dev)<tolerance):
            color_y=(0,255,0)
        else:
            color_y=(0,0,255)
        image = cv2.putText(image, str_y, (220, height-8),font, 0.55, color_y, 2)
        
        
        #write direction, speed, tracking status
        cmd = track_data[4]
        image = cv2.putText(image, str(cmd), (int(width/2) + 10, height-8),font, 0.68, (0, 255, 255), 2)
        
        speed = track_data[5]
        str_sp='Speed: {}%'.format(speed)
        image = cv2.putText(image, str_sp, (int(width/2) + 185, height-8),font, 0.55, (150, 150, 255), 2)
        
        if(cmd==0):
            str1="No object"
        elif(cmd=='Stop'):
            str1='Acquired'
        else:
            str1='Tracking'
        image = cv2.putText(image, str1, (width-140, 18),font, 0.7, (0, 255, 255), 2)
        
        #draw center cross lines
        image = cv2.rectangle(image, (0,int(height/2)-1), (width, int(height/2)+1), (255,0,0), -1)
        image = cv2.rectangle(image, (int(width/2)-1,0), (int(width/2)+1,height), (255,0,0), -1)
        
        #draw the center red dot on the object
        image = cv2.circle(image, (int(track_data[0]),int(track_data[1])), 7, (0,0,255), -1)

        #draw the tolerance box
        image = cv2.rectangle(image, (int(width/2-tolerance),int(height/2-tolerance)),
                                     (int(width/2+tolerance),int(height/2+tolerance)), (0,255,0), 2)
        
    #draw bounding boxes
    for obj in objs:
        name, [x, y, w, h] = obj
        
        box_color, text_color, thickness=(0,150,255), (0,255,0),2
        image = cv2.rectangle(image, (x, y), (x + w, y + h), box_color, thickness)
        image = cv2.putText(image, name, (x, y-5),font, 0.5, text_color, thickness)
        
    return image

if __name__ == '__main__':
    main()