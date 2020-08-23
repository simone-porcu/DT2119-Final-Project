import argparse
import tensorflow as tf
from pathlib import Path
from dualstudent.datasets import timit
from dualstudent.speech.preprocess import normalize

BUFFER_SIZE = 1024
BATCH_SIZE = 32


def _get_tf_dataset(dataset, padding_values, shuffle=False):
    features = [utterance['features'] for utterance in dataset]
    labels = [utterance['labels'] for utterance in dataset]
    n_features = features[0].shape[1]

    x_dataset = tf.data.Dataset.from_generator(lambda: features, output_types=tf.float64)
    y_dataset = tf.data.Dataset.from_generator(lambda: labels, output_types=tf.int32, output_shapes=(None,))
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)      # before padding, less memory used!
    dataset = dataset.padded_batch(batch_size=BATCH_SIZE, padding_values=padding_values,
                                   padded_shapes=((None, n_features), (None,)))
    return dataset


if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train dual student.')
    parser.add_argument('--dataset', type=str, help='Path to TIMIT dataset', required=True)
    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    # load dataset
    train_set, test_set = timit.load_data(dataset_path)
    train_set, test_set = normalize(train_set, test_set)

    # define input pipeline
    # TODO: choose padding values not present in the dataset
    padding_values = (tf.constant(-50, dtype=tf.float64), tf.constant(50, dtype=tf.int32))
    train_ds = _get_tf_dataset(train_set, padding_values, shuffle=True)
    test_ds = _get_tf_dataset(train_set, padding_values)

    # train model
    # TODO

    # evaluate model
    # TODO
