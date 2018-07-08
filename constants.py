from os.path import join

DATA_DIRECTORY_PATH = './data/'
FACE_CASCADE_PATH = './cascade/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = './cascade/haarcascade_eye.xml'
INPUT_CSV_PATH = join(DATA_DIRECTORY_PATH, 'fer2013.csv')
NUMPY_IMAGES_DATA_PATH = join(DATA_DIRECTORY_PATH, "image_data.npy")
NUMPY_LABEL_DATA_PATH = join(DATA_DIRECTORY_PATH, "label_data.npy")

FACE_SIZE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

MODEL_CHECKPOINT = join(DATA_DIRECTORY_PATH, "checkpoint_emr")
MODEL_PATH = join(DATA_DIRECTORY_PATH, "face_me_model")