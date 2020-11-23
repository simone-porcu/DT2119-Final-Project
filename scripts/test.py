import argparse
import numpy as np
from pathlib import Path
from utils import get_number_of_classes, N_HIDDEN_LAYERS, N_UNITS, PADDING_VALUE
from dualstudent.datasets import timit
from dualstudent.preprocess import normalize
from dualstudent.models import DualStudent

# tune here!
NORMALIZATION = 'speaker'


def get_data(dataset_path, normalization):
    train_set, test_set = timit.load_data(dataset_path)
    _, test_set = normalize(train_set, test_set, mode=normalization)
    x_test = np.array([utterance['features'] for utterance in test_set])
    y_test = np.array([utterance['labels'] for utterance in test_set])
    return x_test, y_test


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Test Dual Student for Automatic Speech Recognition on TIMIT dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path to the model.')
    parser.add_argument('output', type=str, help='path where to save the evaluation.')
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    dataset_path = Path(args.data)
    model_path = args.model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    x_test, y_test = get_data(dataset_path, NORMALIZATION)
    _, evaluation_mapping, _ = timit.get_phone_mapping()

    for version in ['mono_directional', 'bidirectional', 'imbalanced']:
        model = DualStudent(
            n_classes=get_number_of_classes(),
            n_hidden_layers=N_HIDDEN_LAYERS,
            n_units=N_UNITS,
            padding_value=PADDING_VALUE,
            version=version
        )
        model.build(input_shape=(None,) + x_test[0].shape)      # necessary, otherwise load_weights() fails

        try:
            model.load_weights(model_path)
            print(f'model version: {version}')
            break
        except ValueError:
            print(f'not {version}, retrying...')

    results = model.test(x_test, y_test, evaluation_mapping=evaluation_mapping)
    with open(output_path / 'performance.txt', mode='w') as f:
        for k, v in results.items():
            output = f'{k}: {v}'
            print(output)
            f.write(output + '\n')


if __name__ == '__main__':
    main()
