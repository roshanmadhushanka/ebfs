import numpy as np
import cv2
from constants import NUMPY_IMAGES_DATA_PATH, NUMPY_LABEL_DATA_PATH, FACE_SIZE, EMOTIONS, MODEL_CHECKPOINT, MODEL_PATH, FACE_CASCADE_PATH
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from os.path import isfile
import sys


class FaceMe:
    def __init__(self):
        # Defining the model
        network = input_data(shape=[None, FACE_SIZE, FACE_SIZE, 1])
        network = conv_2d(network, 64, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 64, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = conv_2d(network, 128, 4, activation='relu')
        network = dropout(network, 0.3)
        network = fully_connected(network, 3072, activation='relu')
        network = fully_connected(
            network, len(EMOTIONS), activation='softmax')

        network = regression(
            network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )

        self.model = tflearn.DNN(
            network,
            checkpoint_path=MODEL_CHECKPOINT,
            max_checkpoints=1,
            tensorboard_verbose=2
        )

        if isfile(MODEL_PATH):
            self.model.load(MODEL_PATH)
            print("[:)] Model loaded from ", MODEL_PATH)

    def predict(self, image):
        if self.model is None:
            print("[:(] Cannot find the model")
            return None

        if image is None:
            print("[:(] Image is none")
            return None

        image = image.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        return self.model.predict(image)[0]


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    fm = FaceMe()

    emotion_offsets = (20, 40)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():  # True:
        ret, bgr_image = cap.read()

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (48, 48))
            except:
                continue

            prediction = fm.predict(gray_face)
            prediction_text = EMOTIONS[np.argmax(prediction)]
            max_val = np.max(prediction)

            print(EMOTIONS[0], prediction[0] / max_val)
            print(EMOTIONS[1], prediction[1] / max_val)
            print(EMOTIONS[2], prediction[2] / max_val)
            print(EMOTIONS[3], prediction[3] / max_val)
            print(EMOTIONS[4], prediction[4] / max_val)
            print(EMOTIONS[5], prediction[5] / max_val)
            print(EMOTIONS[6], prediction[6] / max_val)
            print("--------------------------------")



            draw_bounding_box(face_coordinates, bgr_image, 1)
            draw_text(face_coordinates, bgr_image, prediction_text,
                      200, 0, -45, 1, 1)

        cv2.namedWindow('window_frame')
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


