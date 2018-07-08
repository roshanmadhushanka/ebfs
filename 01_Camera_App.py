import cv2
from constants import FACE_CASCADE_PATH

image_offsets = (30, 30)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)