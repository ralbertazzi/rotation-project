import tensorflow as tf    
import keras.backend as K
import numpy as np

def set_keras_session(gpu_memory_fraction=0.4):
    
    
    from keras.backend.tensorflow_backend import set_session
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))

def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true, 1), K.argmax(y_pred, 1))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def cos_error(y_true, y_pred):
    """
    Calculate the mean cosine between the true angles
    and the predicted angles
    """
    degrees_to_radians = K.cast(np.pi / 180, K.floatx())
    diff = angle_difference(K.argmax(y_true, 1), K.argmax(y_pred, 1))
    diff = K.cast(diff, K.floatx())
    return -K.mean(K.cos(diff * degrees_to_radians))