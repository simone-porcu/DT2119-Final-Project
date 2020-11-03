import argparse
import numpy as np
import tensorflow as tf
from dualstudent.datasets import timit
from dualstudent.speech import normalize
from dualstudent.models import DualStudent
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student on TIMIT dataset for automatic speech recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path to the model.')
    return parser.parse_args()


def main():
    # command-line arguments
    args = get_command_line_arguments()
    dataset_path = args.data
    model_path = args.model

    # load dataset
    train_set, test_set = timit.load_data(dataset_path)
    _, test_set = normalize(train_set, test_set)
    x_test = np.array([u['features'] for u in test_set])
    y_test = np.array([u['labels'] for u in test_set])

    # pad sequences
    padding_value = np.inf
    n_classes = len(timit.get_phone_mapping()[1])
    x_train = pad_sequences(x_test, padding='post', value=padding_value, dtype='float32')
    y_train = pad_sequences(y_test, padding='post', value=n_classes)
    n_features = x_train.shape[-1]

    # one-hot encode labels
    y_train = tf.one_hot(y_train, depth=n_classes).numpy()

    # predict with model
    model = DualStudent(n_classes, n_features, padding_value=padding_value, student_version='mono_directional',
                        n_layers=2, n_units=3)  # TODO: modify
    model.compile(metrics=['accuracy'])
    model.load_weights(model_path)

    # TODO: problem: if we pad the test sequences, accuracy etc. are wrong... they consider also the padding
    # y_pred = model.predict(x_test)
    # y_true = np.argmax(y_train, axis=-1)

    # accuracy (in lab 3: frame-by-frame at phoneme level)

    # phone error rate (PER) (in lab 3: edit distance at phoneme level)
    probabilities = model.predict(x_test)
    y_pred = np.argmax

    # TODO: compute PER, confusion matrix


if __name__ == '__main__':
    main()
