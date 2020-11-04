import numpy as np
import tensorflow as tf
from dualstudent.datasets import timit
from tensorflow.keras.preprocessing.sequence import pad_sequences


PADDING_VALUE = np.inf


def prepare_data_for_model(dataset, n_classes, one_hot=True):
    # take just features and labels
    x = np.array([u['features'] for u in dataset])
    y = np.array([u['labels'] for u in dataset])

    # pad sequences
    x = pad_sequences(x, padding='post', value=PADDING_VALUE, dtype='float32')
    y = pad_sequences(y, padding='post', value=n_classes)

    # one-hot encode labels
    if one_hot:
        y = tf.one_hot(y, depth=n_classes).numpy()

    return x, y


def get_number_of_classes():
    return len(timit.get_phone_mapping()[1])
