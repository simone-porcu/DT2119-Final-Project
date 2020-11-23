import argparse
import tensorflow as tf
from pathlib import Path
from utils import get_number_of_classes, N_HIDDEN_LAYERS, N_UNITS, PADDING_VALUE
from dualstudent.datasets import timit
from dualstudent.models import DualStudent


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Save a checkpoint to a saved model loadable with load_weights().',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('version', type=str, help="version of the model. One of 'mono_directional', 'bidirectional', "
                                                  "'imbalanced'.")
    parser.add_argument('output', type=str, help='path where to save the model')
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    dataset_path = args.data
    checkpoint_path = args.checkpoint
    version = args.version
    model_path = Path(args.output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = str(model_path)

    train_set, _ = timit.load_data(dataset_path)

    model = DualStudent(
        n_classes=get_number_of_classes(),
        n_hidden_layers=N_HIDDEN_LAYERS,
        n_units=N_UNITS,
        padding_value=PADDING_VALUE,
        version=version
    )

    model.build(input_shape=(None,) + train_set[0]['features'].shape)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path)
    model.save_weights(model_path)


if __name__ == '__main__':
    main()
