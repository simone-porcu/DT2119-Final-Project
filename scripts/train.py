import argparse
import numpy as np
from pathlib import Path
from utils import prepare_data_for_model, get_number_of_classes, PADDING_VALUE
from dualstudent.datasets import timit
from dualstudent.speech import normalize, unlabel
from dualstudent.models import DualStudent
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
    np.random.seed(1)

    # command-line arguments
    args = get_command_line_arguments()
    dataset_path = args.data
    model_path = Path(args.model)

    # prepare paths
    if model_path.is_dir():
        raise FileExistsError(str(model_path) + ' already exists')
    model_path.mkdir(parents=True)
    logs_path = model_path / 'logs'
    checkpoints_path = model_path / 'checkpoints'
    checkpoints_path.mkdir()
    model_path = str(model_path / 'model.h5')

    # prepare data
    train_set, _ = timit.load_data(dataset_path)
    train_set, val_set = timit.split_validation(train_set)
    train_set, val_set = normalize(train_set, val_set)
    n_classes = get_number_of_classes()
    x_train, y_train = prepare_data_for_model(train_set, n_classes)
    x_val, y_val = prepare_data_for_model(val_set, n_classes)
    x_labeled, x_unlabeled, y = unlabel(x_train, y_train, .7)
    n_features = x_train.shape[-1]

    # train model
    model = DualStudent(n_classes, n_features, padding_value=PADDING_VALUE)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.1))
    model.train(x_labeled, x_unlabeled, y, x_val=x_val, y_val=y_val)
    model.save_weights(model_path)


if __name__ == '__main__':
    main()
