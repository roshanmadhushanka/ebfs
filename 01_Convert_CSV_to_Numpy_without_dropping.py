import pandas as pd
import numpy as np
from constants import INPUT_CSV_PATH, EMOTIONS, NUMPY_IMAGES_DATA_PATH, \
    NUMPY_LABEL_DATA_PATH, FACE_SIZE


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

    for index, row in data_set.iterrows():
        face_data = np.fromstring(str(row["pixels"]), dtype=np.uint8, sep=' ')
        images.extend(face_data)
        labels.extend(extract_emotion(row["emotion"]))

        print("[:)] Working : {} / {} {:.2f}%".format(index, data_set_size, index * 100.0 / data_set_size))

    images = np.array(images)
    labels = np.array(labels)

    np.save(NUMPY_IMAGES_DATA_PATH, images.reshape([-1, FACE_SIZE, FACE_SIZE, 1]))
    np.save(NUMPY_LABEL_DATA_PATH, labels.reshape([-1, len(EMOTIONS)]))
    print("[:)] Data converted and saved successfully")


if __name__ == "__main__":
    print("[:)] Application started ...")
    main()