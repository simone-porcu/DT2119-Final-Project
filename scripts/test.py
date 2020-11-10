import argparse
from pathlib import Path
from utils import *
from dualstudent.datasets import timit
from dualstudent.preprocess import normalize
from dualstudent.models import DualStudent


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student on TIMIT dataset for automatic preprocess recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path to the model.')
    parser.add_argument('output', type=str, help='path where to save the evaluation.')
    return parser.parse_args()


def get_data(dataset_path):
    train_set, test_set = timit.load_data(dataset_path)
    _, test_set = normalize(train_set, test_set)
    x_test = np.array([utterance['features'] for utterance in test_set])
    y_test = np.array([utterance['labels'] for utterance in test_set])
    return x_test, y_test


def main():
    # prepare paths
    args = get_command_line_arguments()
    dataset_path = args.data
    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    config = Config(model_name=model_path.name)
    model_path = str(model_path / 'model.h5')

    # prepare data
    x_test, y_test = get_data(dataset_path)
    _, evaluation_mapping, _ = timit.get_phone_mapping()

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
    model.load_weights(model_path)

    # evaluate model
    results = model.test(x_test, y_test, evaluation_mapping=evaluation_mapping)
    with open(output_path / 'performance.txt') as f:
        for k, v in results.items():
            output = f'{k}: {v}'
            print(output)
            f.write(output + '\n')


if __name__ == '__main__':
    main()
