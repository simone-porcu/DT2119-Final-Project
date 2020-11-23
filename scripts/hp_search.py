import argparse
import numpy as np
import itertools as it
import tensorflow as tf
from pathlib import Path
from tensorboard.plugins.hparams import api as hp
from utils import Config, get_number_of_classes, N_HIDDEN_LAYERS, N_UNITS, PADDING_VALUE
from train import get_data, get_optimizer
from dualstudent.datasets import timit
from dualstudent.models import DualStudent

# tune here!
GROUP_MEMBERS = ['Franco Ruggeri', 'Andrea Caraffa', 'Kevin Dalla Torre Castillo', 'Simone Porcu']
WHO_AM_I = 'Franco Ruggeri'

CONSISTENCY_LOSSES = ['mse', 'kl']
SCHEDULES = ['rampup', 'sinusoidal_cycling', 'triangular_cycling']

SIGMA_N_VALUES = 4                  # only for grid search
SIGMA_MIN = 0.01                    # linear space
SIGMA_MAX = 1

CONSISTENCY_SCALE_N_VALUES = 4      # only for grid search
CONSISTENCY_SCALE_MIN = 0           # log space
CONSISTENCY_SCALE_MAX = 4

STABILIZATION_SCALE_N_VALUES = 4    # only for grid search
STABILIZATION_SCALE_MIN = 0         # log space
STABILIZATION_SCALE_MAX = 4

XI_N_VALUES = 4                     # only for grid search
XI_MIN = 0.02                       # linear space
XI_MAX = 0.8

N_TRIALS = 500                      # only for random search
N_EPOCHS = 20

# fixed, do not touch here!
VERSION = 'mono_directional'        # one of 'mono_directional', 'bidirectional', 'imbalanced'
NORMALIZATION = 'speaker'           # one of 'full', 'speaker', 'utterance'
OPTIMIZER = 'adam_w'                # one of 'adam', 'adam_w', 'sgd', 'sgd_w'
UNLABELED_PERCENTAGE = 0.7
BATCH_SIZE = 100
SCHEDULE_LENGTH = 5
SEED = 1


def generate_grid_possibilities():
    # generate values for each hyper-parameter
    consistency_loss = CONSISTENCY_LOSSES
    schedule = SCHEDULES
    sigma = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_N_VALUES)
    consistency_scale = np.logspace(CONSISTENCY_SCALE_MIN, CONSISTENCY_SCALE_MAX, CONSISTENCY_SCALE_N_VALUES)
    stabilization_scale = np.logspace(STABILIZATION_SCALE_MIN, STABILIZATION_SCALE_MAX, STABILIZATION_SCALE_N_VALUES)
    xi = np.linspace(XI_MIN, XI_MAX, XI_N_VALUES)

    # generate all possibilities
    possibilities = list(it.product(consistency_loss, schedule, sigma, consistency_scale, stabilization_scale, xi))
    possibilities = sorted(possibilities)       # to have same order for different runs (needed to split up the work)
    return possibilities


def generate_random_possibilities(seed=None):
    if seed is not None:
        np.random.seed(seed)

    # generate values for each hyper-parameter
    consistency_loss = np.random.choice(CONSISTENCY_LOSSES, N_TRIALS)
    schedule = np.random.choice(SCHEDULES, N_TRIALS)
    sigma = np.random.uniform(SIGMA_MIN, SIGMA_MAX, N_TRIALS)
    consistency_scale = np.random.uniform(CONSISTENCY_SCALE_MIN, CONSISTENCY_SCALE_MAX, N_TRIALS)
    consistency_scale = np.power(10, consistency_scale)             # log space
    stabilization_scale = np.random.uniform(STABILIZATION_SCALE_MIN, STABILIZATION_SCALE_MAX, N_TRIALS)
    stabilization_scale = np.power(10, stabilization_scale)         # log space
    xi = np.random.uniform(XI_MIN, XI_MAX, N_TRIALS)

    # generate possibilities
    possibilities = zip(consistency_loss, schedule, sigma, consistency_scale, stabilization_scale, xi)
    return possibilities


def generate_possibilities(mode, seed=None):
    if mode == 'grid':
        possibilities = generate_grid_possibilities()
    elif mode == 'random':
        possibilities = generate_random_possibilities(seed)
    else:
        raise ValueError('Invalid mode')
    return possibilities


def get_my_part(possibilities):
    n = len(possibilities) / len(GROUP_MEMBERS)
    idx = GROUP_MEMBERS.index(WHO_AM_I)
    idx_start = int(idx * n)
    idx_end = int((idx + 1) * n)
    return possibilities[idx_start:idx_end]


def run_possibilities(dataset_path, logs_path, possibilities):
    x_train_labeled, x_train_unlabeled, y_train_labeled, x_val, y_val = get_data(
        dataset_path=dataset_path,
        normalization=NORMALIZATION,
        unlabeled_percentage=UNLABELED_PERCENTAGE,
        seed=SEED
    )
    _, evaluation_mapping, _ = timit.get_phone_mapping()
    n_classes = get_number_of_classes()

    for consistency_loss, schedule, sigma, consistency_scale, stabilization_scale, xi in possibilities:
        hparams = {
            'consistency_loss': consistency_loss,
            'schedule': schedule,
            'sigma': sigma,
            'consistency_scale': consistency_scale,
            'stabilization_scale': stabilization_scale,
            'xi': xi
        }

        for k, v in hparams.items():
            print(f'{k}={v}, ', end='')
        print()

        config = Config(
            version='mono_directional',
            n_hidden_layers=N_HIDDEN_LAYERS,
            n_units=N_UNITS,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            unlabeled_percentage=UNLABELED_PERCENTAGE,
            optimizer=OPTIMIZER,
            consistency_loss=consistency_loss,
            consistency_scale=consistency_scale,
            stabilization_scale=stabilization_scale,
            xi=xi,
            sigma=sigma,
            schedule=schedule,
            schedule_length=SCHEDULE_LENGTH,
            normalization=NORMALIZATION,
            seed=SEED
        )

        logs_path_ = logs_path / str(config)
        if logs_path_.is_dir():     # skip what already done (e.g. in case of crashes)
            print('already done, skipping...')
            continue
        logs_path_.mkdir(parents=True)
        logs_path_ = str(logs_path_)

        model = DualStudent(
            n_classes=n_classes,
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
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            seed=config.seed
        )

        results = model.test(
            x=x_val,
            y=y_val,
            batch_size=config.batch_size,
            evaluation_mapping=evaluation_mapping
        )

        with tf.summary.create_file_writer(logs_path_).as_default():
            hp.hparams(hparams)
            for k, v in results.items():
                tf.summary.scalar(k, v, step=N_EPOCHS)


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Hyper-parameter search for Dual Student on TIMIT dataset for Automatic Speech Recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('mode', type=str, choices=['grid', 'random'], help='mode of hyper-parameter search')
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('logs', type=str, help='path where to logs')
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    dataset_path = Path(args.data)
    logs_path = Path(args.logs)

    possibilities = generate_possibilities(args.mode, seed=SEED)
    possibilities = get_my_part(possibilities)      # comment this if you want to run everything
    run_possibilities(dataset_path, logs_path, possibilities)


if __name__ == '__main__':
    main()
