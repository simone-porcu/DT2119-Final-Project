import argparse
from pathlib import Path
from utils import *
from dualstudent.datasets import timit
from dualstudent.preprocess import normalize, map_labels
from dualstudent.models import DualStudent
from dualstudent.metrics import PhoneErrorRate, plot_confusion_matrix, make_classification_report


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
    model_path = str(model_path)

    # prepare data
    x_test, y_test = get_data(dataset_path)
    _, _, test_label_to_phone_name = timit.get_phone_mapping()

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

    # predict with model
    y_pred, mask = model.pad_and_predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)

    # phone error rate
    per = PhoneErrorRate()
    for i in range(len(y_pred)):
        # drop padding
        x_utterance = x_test[i]
        length = len(x_utterance)
        y_utterance_true = y_test[i]
        y_utterance_pred = y_pred[i][:length]

        # map training phones to test phones
        y_utterance_true = list(map_labels(test_label_to_phone_name, y_utterance_true))
        y_utterance_pred = list(map_labels(test_label_to_phone_name, y_utterance_pred))

        # update phone error rate
        per.update_state(y_utterance_true, y_utterance_pred)

    print('Phone Error Rate: {:.2f}%'.format(per * 100))
    with open(output_path / 'per.txt', mode='w') as f:
        f.write('Phone Error Rate: {:.2f}%'.format(per * 100))

    # classification report and confusion matrix
    y_test = y_test.flatten()   # flatten
    y_pred = y_pred[mask]       # flatten removing padding
    class_names = [name for _, name in sorted(test_label_to_phone_name.items())]
    make_classification_report(y_test, y_pred, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names)


if __name__ == '__main__':
    main()
