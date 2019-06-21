from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import argparse
import cv2
import base64

dirpath = os.getcwd()
scriptpath = os.path.realpath(__file__)
base_dir = os.path.dirname(scriptpath)


def detect(path):
    """Detect if picture has a face, returns false or emotion detection prediction (happy, sad, angry, etc)"""
    # img = Image.open(path)
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    # If more than one result are a lot of faces
    # If none result is not a face image
    if len(face_locations) == 1:
        # print(os.path.splitext(path))
        # imgname = os.path.splitext(path)[0]
        extension = os.path.splitext(path)[1]

        app_dir = base_dir + "/outs"
        try:
            os.mkdir(app_dir)
        except OSError:
            None

        output = app_dir + "/output" + extension
        emotion = app_dir + "/emotion" + extension

        top, right, bottom, left = face_locations[0]
        sample_top = int(top - top * 0.45)
        sample_bottom = int((bottom * 0.15) + bottom)
        sample_left = int(left - left * 0.35)
        sample_right = int((right * 0.15) + right)

        face_image1 = image[sample_top:sample_bottom, sample_left:sample_right]
        image_save = Image.fromarray(face_image1)
        print(output)
        image_save.save(output)

        # Emotion
        emotion_image = image[top:bottom, left:right]
        emotion_image_save = Image.fromarray(emotion_image)
        emotion_image_save.save(emotion)

        emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

        face_image = cv2.imread(emotion)

        with open(output, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        # resizing the image
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

        # Train model
        # https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/src/EmotionDetector_v2.ipynb
        model = load_model(base_dir + "/model_v6_23.hdf5")
        predicted_class = np.argmax(model.predict(face_image))
        label_map = dict((v, k) for k, v in emotion_dict.items())
        predicted_label = label_map[predicted_class]
        # return predicted_label, encoded_string

        # os.remove(output)
        # os.remove(emotion)

        return encoded_string
        # return output
        # return True
    else:
        return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path', help='Picture file path')
    # todo argument parser predict emotion
    argparser.add_argument('--e', metavar='emotion', required=False, help='emotion')
    args = argparser.parse_args()
    path = args.path
    is_face = detect(path)
    print(False if False else is_face)
    # print('Extracted Text', captcha_text)
