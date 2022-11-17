import os
import json

import numpy as np
import tensorflow as tf
import cv2
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class Recognizer(object):
    def __init__(self, model_path):
        self.net = tf.keras.models.load_model(model_path)

    def preprocess(self, image):
        """
		Arguments:
            image = Face crop of the image
		Output:
            return preprocessed(normalized) version of image
		"""
        # resize image and converty to recommended data type
        img = cv2.resize(image, (160,160))
        img = np.asarray(img, 'float32')

        axis = (0,1,2)
        size = img.size

        mean = np.mean(img, axis=axis, keepdims=True)
        std = np.std(img, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        processed_img = (img-mean) / std_adj

        return processed_img

    # l2 normalize embeddindgs
    def l2_normalize(self, embed, axis=-1, epsilon=1e-10):
        """
		Arguments:
            embed - 128 number long embeddind 
            axis - axis of the embedding default to -1
            epsilon - a small number to avoid division by zero 
		Output:
            normalized version of embeddings

		"""
        output = embed / np.sqrt(np.maximum(np.sum(np.square(embed), axis=axis, keepdims=True), epsilon))
        return output
    
    # method for getting face embeddings using model 
    def get_face_embedding(self, face):
        """
		Arguments:
            face - face crop drom an image
		Output:
            face embedding with 128 parameters

		"""
        # preprocess image and expand the dimension 
        processed_face = self.preprocess(face)
        processed_face = np.expand_dims(processed_face, axis=0)

        # predict using model and l2 normalize embedding
        model_pred = self.net.predict(processed_face)
        face_embedding = model_pred
        #face_embedding = self.l2_normalize(model_pred)
        return face_embedding
    
    # calculate euclidain distance between the true and predicted 
    # face embeddings
    def calculate_distance(self, embd_real, embd_candidate):
        """
		Arguments:
            embd_embd - embedding from database
            embd_candidate - model predicted embedding
		Output:
            euclidian distance between the two embeddings

		"""
        return distance.euclidean(embd_real, embd_candidate)
    
    def crop_face(self, frame, box):
        x, y, w, h = box
        return frame[y : y + h, x : x + w]

    def identify(self, clf_dir, decode_json_dir, face_image):
        """
		Arguments:
            clf_dir - model directory for svc_classifier
            decode_json_dir - directory for json file containing data to decode svc 
                                classifier output 
            face_image - face on which to predict
		Output:
            a tuple with format (confidence, person_name) for the face image
		"""
        
        # load svc classifier
        with open(clf_dir, "rb") as file:
            svc_clf = joblib.load(file) 

        # load decode json
        with open(decode_json_dir, "r") as file:
            class_decode = json.load(file) 

        # get face embedding for the face crop
        face_embd = self.get_face_embedding(face_image)
        # get svc_classifier output
        prediction = svc_clf.predict_proba(face_embd)
        # pass
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]*100
        if confidence >= 70:
            person_name = class_decode[str(class_id)]
        else:
            person_name = 'UNKNOWN'
            confidence = 0

        return (confidence, person_name)

    