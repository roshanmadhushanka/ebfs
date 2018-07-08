import pandas as pd
import numpy as np
import cv2
from constants import INPUT_CSV_PATH, FACE_CASCADE_PATH, EYE_CASCADE_PATH, FACE_SIZE, EMOTIONS, NUMPY_IMAGES_DATA_PATH, \
    NUMPY_LABEL_DATA_PATH


def extract_face(pixel_data):
    """
    Extract faces from the given pixel data upon availability
    :param pixel_data: row pixel data separated by space
    :return: Extracted face in numpy array format
    """

    # Create 2D image array from the string data
    image_data = np.fromstring(str(pixel_data), dtype=np.uint8, sep=' ').reshape((FACE_SIZE, FACE_SIZE))

    # Apply Haar-cascade to detect faces
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    # faces = face_cascade.detectMultiScale(image_data, scaleFactor=1.1, minNeighbors=5)

    faces = face_cascade.detectMultiScale(image_data, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # primary_face = None
    if len(faces) == 0:
        eyes = eye_cascade.detectMultiScale(image_data)
        if len(eyes) == 0:
            return None
        else:
            primary_face = image_data
    else:
        # Detecting primary face
        primary_face_meta_info = faces[0]
        for face in faces:
            if face[2] * face[3] > primary_face_meta_info[2] * primary_face_meta_info[3]:
                primary_face_meta_info = face

        # Chop image
        primary_face = image_data[primary_face_meta_info[1]:(primary_face_meta_info[1] + primary_face_meta_info[2]),
                     primary_face_meta_info[0]:(primary_face_meta_info[0] + primary_face_meta_info[3])]

    # Resize the image to match network input
    try:
        primary_face = cv2.resize(primary_face, (FACE_SIZE, FACE_SIZE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[:(] Problem occurred during resizing")
        return None

    return primary_face


def extract_emotion(emotion_data):
    """
    Create emotion classification to one hot encoding
    :param emotion_data: Emotion index
    :return: One hot encoded vector
    """
    d = np.zeros(len(EMOTIONS))
    d[emotion_data] = 1.0
    return d


def main():
    data_set = pd.read_csv(INPUT_CSV_PATH)
    if data_set is not None and type(data_set) == pd.core.frame.DataFrame:
        print("[:)] Data set loaded successfully")
        print("--- Data Set ---")
        print("Shape :", data_set.shape)

    labels = []
    images = []

    data_set_size = data_set.shape[0]

    print("[:)] Conversion started ...")
    drop_count = 0
    stat = {
        0: [0, 0],
        1: [0, 0],
        2: [0, 0],
        3: [0, 0],
        4: [0, 0],
        5: [0, 0],
        6: [0, 0]
    }

    for index, row in data_set.iterrows():
        face_image = extract_face(row["pixels"])
        if face_image is not None:
            images.extend(face_image)
            labels.extend(extract_emotion(row["emotion"]))
        else:
            stat[row["emotion"]][0] += 1
            drop_count += 1
        stat[row["emotion"]][1] += 1
        print("[:)] Working : {} / {} {:.2f}% Dropped : {}".format(index, data_set_size, index * 100.0 / data_set_size, drop_count))

    images = np.array(images)
    labels = np.array(labels)

    np.save(NUMPY_IMAGES_DATA_PATH, images.reshape([-1, FACE_SIZE, FACE_SIZE, 1]))
    np.save(NUMPY_LABEL_DATA_PATH, labels.reshape([-1, len(EMOTIONS)]))
    print("[:)] Data converted and saved successfully")
    # Drop count in original 21869
    print("[:)] Drop data count :", drop_count)
    print("[:)] Drop summary")
    for i in range(len(EMOTIONS)):
        print("[:)] {} : {} / {}".format(EMOTIONS[i], stat[i][0], stat[i][1]))


if __name__ == "__main__":
    print("[:)] Application started ...")
    main()

    # 35102
    # 35192 image_data, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    # 27555 image_data, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE