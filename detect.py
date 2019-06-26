import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image, ImageFile
import warnings
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import argparse
import cv2
import base64

# suppress warnings because mobile wont work
tf.logging.set_verbosity(tf.logging.ERROR)

dirpath = os.getcwd()
scriptpath = os.path.realpath(__file__)
base_dir = os.path.dirname(scriptpath)

__all__ = ['fix_orientation']

# PIL's Error "Suspension not allowed here" work around:
# s. http://mail.python.org/pipermail/image-sig/1999-August/000816.html
ImageFile.MAXBLOCK = 1024 * 1024

# The EXIF tag that holds orientation data.
EXIF_ORIENTATION_TAG = 274

# Obviously the only ones to process are 3, 6 and 8.
# All are documented here for thoroughness.
ORIENTATIONS = {
    1: ("Normal", 0),
    2: ("Mirrored left-to-right", 0),
    3: ("Rotated 180 degrees", 180),
    4: ("Mirrored top-to-bottom", 0),
    5: ("Mirrored along top-left diagonal", 0),
    6: ("Rotated 90 degrees", -90),
    7: ("Mirrored along top-right diagonal", 0),
    8: ("Rotated 270 degrees", -270)
}


def fix_orientation(img, save_over=False):
    """
    `img` can be an Image instance or a path to an image file.
    `save_over` indicates if the original image file should be replaced by the new image.
    * Note: `save_over` is only valid if `img` is a file path.
    """
    path = None
    if not isinstance(img, Image.Image):
        path = img
        img = Image.open(path)
    elif save_over:
        raise ValueError("You can't use `save_over` when passing an Image instance.  Use a file path instead.")
    try:
        orientation = img._getexif()[EXIF_ORIENTATION_TAG]
    except (TypeError, AttributeError, KeyError):
        raise ValueError("Image file has no EXIF data.")
    if orientation in [3, 6, 8]:
        degrees = ORIENTATIONS[orientation][1]
        img = img.rotate(degrees)
        if save_over and path is not None:
            try:
                img.save(path, quality=95, optimize=1)
            except IOError:
                # Try again, without optimization (PIL can't optimize an image
                # larger than ImageFile.MAXBLOCK, which is 64k by default).
                # Setting ImageFile.MAXBLOCK should fix this....but who knows.
                img.save(path, quality=95)
        return (img, degrees)
    else:
        return (img, 0)


def detect(path):
    """Detect if picture has a face, returns false or emotion detection prediction (happy, sad, angry, etc)"""

    # Rotate image
    fix_orientation(path, path)

    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    # If more than one result are a lot of faces
    # If none result is not a face image
    if len(face_locations) == 1:
        # print(os.path.splitext(path))
        # imgname = os.path.splitext(path)[0]
        # extension = os.path.splitext(path)[1]
        extension = ".jpg"

        app_dir = base_dir + "/outs"
        try:
            os.mkdir(app_dir)
        except OSError:
            None

        output = app_dir + "/output" + extension
        emotion = app_dir + "/emotion" + extension

        top, right, bottom, left = face_locations[0]
        sample_top = int(top - top * 0.55)
        sample_bottom = int((bottom * 0.25) + bottom)
        sample_left = int(left - left * 0.45)
        sample_right = int((right * 0.25) + right)

        face_image1 = image[sample_top:sample_bottom, sample_left:sample_right]
        image_save = Image.fromarray(face_image1)
        image_save.save(output)

        # Emotion
        emotion_image = image[top:bottom, left:right]
        emotion_image_save = Image.fromarray(emotion_image)
        emotion_image_save.save(emotion)

        emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

        face_image = cv2.imread(emotion)

        with open(output, "rb") as image_file:
            es = base64.b64encode(image_file.read())
            encoded_string = es.decode('utf-8')

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

        os.remove(output)
        os.remove(emotion)

        return "data:image/jpeg;base64," + encoded_string
        # return output
        # return True
    elif len(face_locations) > 1:
        return 2

    else:
        return 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path', help='Picture file path')
    # todo argument parser predict emotion
    argparser.add_argument('--e', metavar='emotion', required=False, help='emotion')
    args = argparser.parse_args()
    path = args.path
    is_face = detect(path)
    print(0 if False else is_face)
    # print('Extracted Text', captcha_text)
