import os

from abc import ABC, abstractmethod
import cv2
import numpy as np

class DnnModel(ABC):
    @abstractmethod
    def preprocess(self, image, scale, input_size, ch_means):
        '''
            Create input blob from image. 
            
            Input parameters:
                -image: The image fed to the network.
                -scale: Multiplier for image value.
                -input_size: Required network input shape.
                -ch_means: Mean values to be subtracted from each channels of the image (before scaling).
            Output:
                -Original height and width of the image.
                -Processed blob.
        '''
        pass

    @abstractmethod
    def detect(self, image, threshold):
        '''
        Get output of the network from input image.

        Input parameters:
            -image: The input image.
            -threshold: The threshold for class confidences of each box.
        Output:
            -A tuple in the format of (box locations, class confidences) 
            for every "good" prediction.
        '''
        pass

    @abstractmethod()
    def draw_boxes(self, frame, net_outputs):
        '''
            Draw predictions onto the displaying frame after further post-processing.

            Input parameters: 
                -frame: Current frame.
                -net_output: Output of the "detect" method.
            Output:
                -The frame with necessary information drawn.
        '''
        pass

class PersonDetector(DnnModel):
    def __init__(self, weight_file, model_file):
        self.net = cv2.readNet(weight_file, model_file)

    def preprocess(self, image, scale=1.0/255, input_size=(416,416), ch_means=(0,0,0)):
        height, width, _ = image.shape

        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, 
                size=input_size, mean=ch_means, swapRB=True, crop=False) 

        return [(height, width), blob] 

    def detect(self, image, threshold=0.5):
        (h,w), blob = self.preprocess(image)

        self.net.setInput(blob)
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        
        outputs = self.net.forward(output_layers_names)
        '''
            Net forward output shape: (1, 2, N, 5+C)
                N: The number of detection.
                5+C: [x_cent, y_cent, w, h, box_conf, class_0_conf, class_1_conf, ...]
        '''

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > threshold:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
        
        return (boxes, confidences)

    def draw_boxes(self, frame, net_outputs):
        boxes, confidences = net_outputs
        indexes = cv2.dnn.NMSboxes(boxes, confidences, 0.6, 0.4)
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                confidence = str(round(confidences[i],2))

                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "person" + confidence, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        return frame

class FaceDetector(DnnModel):
    def __init__(self, weight_file, model_file):
        self.net = cv2.readNet(weight_file, model_file)

    def preprocess(self, image, scale=1.0, input_size=(300,300), ch_means=(104.0, 177.0, 123.0)):
        height, width, _ = image.shape

        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, 
                size=input_size, mean=ch_means, swapRB=True, crop=False) 

        return [(height, width), blob] 

    def detect(self, image, threshold):
        (h,w), blob = self.preprocess(image)

        self.net.setInput(blob)
        detections = self.net.forward()
        '''
            Net forward output shape: (1, 1, 200, 7)
                200: The number of detections.
                7: [_, _, conf, left(x1), top(y1), right(x2), bottom(y2)]
        '''

        boxes = []
        confidences = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                bounding_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                left, top, right, bottom = bounding_box.astype('int')
                width = right - left
                height = top - bottom
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))

        return (boxes, confidences)
    
    def draw_boxes(self, frame, net_outputs):
        boxes, confidences = net_outputs
        indexes = cv2.dnn.NMSboxes(boxes, confidences, 0.6, 0.4)
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                #label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, confidence, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        return frame

    