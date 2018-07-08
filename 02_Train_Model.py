import numpy as np
from constants import NUMPY_IMAGES_DATA_PATH, NUMPY_LABEL_DATA_PATH, FACE_SIZE, EMOTIONS, MODEL_CHECKPOINT, MODEL_PATH
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Load data
image_data = np.load(NUMPY_IMAGES_DATA_PATH)
label_data = np.load(NUMPY_LABEL_DATA_PATH)
print("[:)] Data loaded")
print("[:)] Input shape ", image_data.shape,"Label shape ", label_data.shape)

# Defining network
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
print("[:)] Network created")

model = tflearn.DNN(
    network,
    checkpoint_path=MODEL_CHECKPOINT,
    max_checkpoints=1,
    tensorboard_verbose=2
)
print("[:)] Model is set")
print("[:)] Start training ...")

model.fit(
    image_data,
    label_data,
    n_epoch=50,
    batch_size=1000,
    shuffle=True,
    show_metric=True,
    snapshot_step=200,
    snapshot_epoch=True,
    run_id='emotion_recognition'
)
print("[:)] Training completed")

model.save(MODEL_PATH)
print("[:)] Model saved to ", MODEL_PATH)
