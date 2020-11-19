import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
from utils import Config, get_number_of_classes, N_HIDDEN_LAYERS, N_UNITS, PADDING_VALUE
from dualstudent.datasets import timit
from dualstudent.preprocess import normalize, unlabel
from dualstudent.models import DualStudent

# tune here!
N_EPOCHS = 1
BATCH_SIZE = 100
VERSION = 'mono_directional'        # one of 'mono_directional', 'bidirectional', 'imbalanced'
NORMALIZATION = 'speaker'           # one of 'full', 'speaker', 'utterance'
OPTIMIZER = 'adam_w'                # one of 'adam', 'adam_w', 'sgd', 'sgd_w'
UNLABELED_PERCENTAGE = 0.7

CONSISTENCY_LOSS = 'mse'            # one of 'mse', 'kl'
CONSISTENCY_SCALE = 0               # weight of consistency constraint
STABILIZATION_SCALE = 0             # weight of stabilization constraint
XI = 0.6                            # confidence threshold
SIGMA = 0.01                        # standard deviation for noisy augmentation
SCHEDULE = 'triangular_cycling'     # one of 'rampup', 'triangular_cycling', 'sinusoidal_cycling'
SCHEDULE_LENGTH = 5                 # length of rampup or half cycle

SEED = 1


def get_data(dataset_path, normalization, unlabeled_percentage, seed=None):
    train_set, _ = timit.load_data(dataset_path)
    train_set, val_set = timit.split_validation(train_set, seed=seed)
    train_set, val_set = normalize(train_set, val_set, mode=normalization)
    train_set = unlabel(train_set, unlabeled_percentage, seed=seed)

    x_train_labeled = np.array([utterance['features'] for utterance in train_set if 'labels' in utterance])
    x_train_unlabeled = np.array([utterance['features'] for utterance in train_set if 'labels' not in utterance])
    y_train_labeled = np.array([utterance['labels'] for utterance in train_set if 'labels' in utterance])
    x_val = np.array([utterance['features'] for utterance in val_set])
    y_val = np.array([utterance['labels'] for utterance in val_set])

    return x_train_labeled, x_train_unlabeled, y_train_labeled, x_val, y_val


def get_optimizer(optimizer):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam()
    elif optimizer == 'adam_w':
        return tfa.optimizers.AdamW(weight_decay=1e-2)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD()
    elif optimizer == 'sgd_w':
        return tfa.optimizers.SGDW(weight_decay=1e-2, nesterov=True)
    else:
        raise ValueError('Invalid optimizer version')


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student for Automatic Speech Recognition on TIMIT dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path where to save model, checkpoints and logs')
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    dataset_path = Path(args.data)
    model_path = Path(args.model)
    logs_path = model_path / 'logs'
    checkpoints_path = model_path / 'checkpoints'
    if model_path.is_dir():
        raise FileExistsError(str(model_path) + ' already exists')
    model_path.mkdir(parents=True)
    checkpoints_path.mkdir()
    logs_path.mkdir()
    model_path = str(model_path / 'model.h5')
    logs_path = str(logs_path)

    config = Config(
        version=VERSION,
        n_hidden_layers=N_HIDDEN_LAYERS,
        n_units=N_UNITS,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        unlabeled_percentage=UNLABELED_PERCENTAGE,
        optimizer=OPTIMIZER,
        consistency_loss=CONSISTENCY_LOSS,
        consistency_scale=CONSISTENCY_SCALE,
        stabilization_scale=STABILIZATION_SCALE,
        xi=XI,
        sigma=SIGMA,
        schedule=SCHEDULE,
        schedule_length=SCHEDULE_LENGTH,
        normalization=NORMALIZATION,
        seed=SEED
    )

    x_train_labeled, x_train_unlabeled, y_train_labeled, x_val, y_val = get_data(
        dataset_path=dataset_path,
        normalization=config.normalization,
        unlabeled_percentage=config.unlabeled_percentage,
        seed=config.seed
    )
    _, evaluation_mapping, _ = timit.get_phone_mapping()

    model = DualStudent(
        n_classes=get_number_of_classes(),
        n_hidden_layers=config.n_hidden_layers,
        n_units=config.n_units,
        consistency_loss=config.consistency_loss,
        consistency_scale=config.consistency_scale,
        stabilization_scale=config.stabilization_scale,
        xi=config.xi,
        padding_value=PADDING_VALUE,
        sigma=config.sigma,
        schedule=config.schedule,
        schedule_length=config.schedule_length,
        version=config.version
    )

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
        evaluation_mapping=evaluation_mapping,
        seed=config.seed
    )

    model.save_weights(model_path)


if __name__ == '__main__':
    main()
