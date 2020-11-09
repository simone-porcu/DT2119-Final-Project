import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
from utils import Config, get_number_of_classes
from dualstudent.datasets import timit
from dualstudent.preprocess import normalize, unlabel
from dualstudent.models import DualStudent


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student on TIMIT dataset for automatic preprocess recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='base path where to save models and checkpoints')
    parser.add_argument('logs', type=str, help='base path where to save the logs for TensorBoard')
    return parser.parse_args()


def get_data(dataset_path, seed=None):
    train_set, _ = timit.load_data(dataset_path)
    train_set, val_set = timit.split_validation(train_set, seed=seed)
    train_set, val_set = normalize(train_set, val_set)
    train_set = unlabel(train_set, 0.7, seed=seed)

    x_train_labeled = np.array([utterance['features'] for utterance in train_set if 'labels' in utterance])
    x_train_unlabeled = np.array([utterance['features'] for utterance in train_set if 'labels' not in utterance])
    y_train_labeled = np.array([utterance['labels'] for utterance in train_set if 'labels' in utterance])
    x_val = np.array([utterance['features'] for utterance in val_set])
    y_val = np.array([utterance['labels'] for utterance in val_set])

    return x_train_labeled, x_train_unlabeled, y_train_labeled, x_val, y_val


def get_optimizer(optimizer):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=0.01)
    elif optimizer == 'adam_w':
        return tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=0.01)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer == 'sgd_w':
        return tfa.optimizers.SGDW(weight_decay=1e-4, learning_rate=0.01, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Invalid optimizer version')


def main():
    # prepare paths
    args = get_command_line_arguments()
    config = Config()
    model_name = str(config)
    dataset_path = Path(args.data)
    model_path = Path(args.model) / model_name
    logs_path = str(Path(args.logs) / model_name)
    checkpoints_path = model_path / 'checkpoints'
    if model_path.is_dir():
        raise FileExistsError(str(model_path) + ' already exists')
    model_path.mkdir(parents=True)
    checkpoints_path.mkdir()
    model_path = str(model_path / 'model.h5')

    # prepare data
    x_train_labeled, x_train_unlabeled, y_train_labeled, x_val, y_val = get_data(dataset_path, seed=config.seed)

    # prepare model
    model = DualStudent(
        n_classes=get_number_of_classes(),
        n_hidden_layers=config.n_hidden_layers,
        n_units=config.n_units,
        consistency_scale=config.consistency_scale,
        stabilization_scale=config.stabilization_scale,
        epsilon=config.epsilon,
        padding_value=config.padding_value,
        sigma=config.sigma,
        version=config.version
    )

    # train model
    model.compile(optimizer=get_optimizer(config.optimizer))
    model.train(
        x_labeled=x_train_labeled,
        x_unlabeled=x_train_unlabeled,
        y_labeled=y_train_labeled,
        x_val=x_val,
        y_val=y_val,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        checkpoints_path=checkpoints_path,
        logs_path=logs_path,
        seed=config.seed
    )
    model.save_weights(model_path)


if __name__ == '__main__':
    main()
