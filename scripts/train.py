import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from dualstudent.datasets import timit
from dualstudent.speech import normalize, unlabel
from dualstudent.models import DualStudent
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student on TIMIT dataset for automatic speech recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path where to save trained model, checkpoints and logs. Must be a '
                                                'non-existing directory.')
    return parser.parse_args()


def main():
    # command-line arguments
    args = get_command_line_arguments()
    dataset_path = args.data
    model_path = Path(args.model)

    # prepare paths
    # if model_path.is_dir():
    #     raise FileExistsError(str(model_path) + ' already exists')
    # model_path.mkdir(parents=True)
    logs_path = model_path / 'logs'
    checkpoints_path = model_path / 'checkpoints'
    # checkpoints_path.mkdir()
    model_path = str(model_path / 'model.h5')

    # load dataset
    train_set, _ = timit.load_data(dataset_path)
    train_set = normalize(train_set)
    x_train = np.array([u['features'] for u in train_set])
    y_train = np.array([u['labels'] for u in train_set])

    # TODO: split in training + validation set

    # pad sequences
    padding_value = np.inf
    n_classes = len(timit.get_phone_mapping()[1])
    x_train = pad_sequences(x_train, padding='post', value=padding_value, dtype='float32')
    y_train = pad_sequences(y_train, padding='post', value=n_classes)
    n_features = x_train.shape[-1]

    # one-hot encode labels
    y_train = tf.one_hot(y_train, depth=n_classes).numpy()

    # unlabel some sequences
    x_labeled, x_unlabeled, y = unlabel(x_train, y_train, .2)

    # train model
    model = DualStudent(n_classes, n_features, padding_value=padding_value, student_version='mono_directional', n_layers=2, n_units=3)  # TODO: modify
    model.compile(optimizer=SGD(learning_rate=0.01))        # TODO: set momentum
    model.train(x_labeled, x_unlabeled, y, shuffle=False)   # TODO: remove shuffle=False
    model.save_weights(model_path)


if __name__ == '__main__':
    main()
