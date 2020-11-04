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

    # split training in training and validation sets (version 1: unique validation and training utterances)
    # train_set, valid_set, complete_valid_set = timit.split_val(train_set)
    # print(f'original train_set dimension: \t {train_set.shape}')
    # print(f'splitted train_set dimension: \t\t {train_set.shape}')
    # print(f'valid: \t\t {valid_set.shape}')
    # print(f'complete_valid:  {complete_valid_set.shape}')

    # split training in training and validation sets (version 2: random validation)
    train_set, val_set = timit.split_val_random(train_set)
    print(train_set.shape)
    print(val_set.shape)

    # normalize dataset
    train_set, val_set = normalize(train_set, val_set)

    # take just features and labels
    x_train = np.array([u['features'] for u in train_set])
    y_train = np.array([u['labels'] for u in train_set])
    x_val = np.array([u['features'] for u in val_set])
    y_val = np.array([u['labels'] for u in val_set])

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
    model.train(x_labeled, x_unlabeled, y, x_val=x_val, y_val=y_val, shuffle=False)   # TODO: remove shuffle=False
    model.save_weights(model_path)


if __name__ == '__main__':
    main()
