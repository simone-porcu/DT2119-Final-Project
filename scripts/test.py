import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import prepare_data_for_model, get_number_of_classes, PADDING_VALUE
from edit_distance import SequenceMatcher
from dualstudent.datasets import timit
from dualstudent.speech import normalize
from dualstudent.models import DualStudent
from sklearn.metrics import confusion_matrix, classification_report


def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Train Dual Student on TIMIT dataset for automatic speech recognition.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str, help='path to the TIMIT dataset')
    parser.add_argument('model', type=str, help='path to the model.')
    parser.add_argument('output', type=str, help='path where to save the evaluation.')
    return parser.parse_args()


def map_labels(mapping, labels):
    return np.array([mapping[label] for label in labels])


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Plots the confusion matrix. The confusion matrix is annotated with the absolute counts, while colored according
    to the normalized values by row. In this way, even for imbalanced datasets, the diagonal is highlighted well if the
    predictions are good.

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels)
    :param predictions: numpy array of shape (n_samples,), predictions
    :param class_names: list of class names to use as ticks
    :param save_path: path to the directory where to save the figure (with name 'confusion_matrix.png')
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure()
    sns.heatmap(cm, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Reds')
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path / 'confusion_matrix.pdf')
    plt.show()


def make_classification_report(labels, predictions, class_names, save_path=None):
    """
    Generates a classification report including: precision, recall, f1-score, accuracy.

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels)
    :param predictions: numpy array of shape (n_samples,), predictions
    :param class_names: list of class names to use as ticks
    :param save_path: path to the directory where to save the report (with name 'classification_report.txt')
    """
    save_path = Path(save_path)
    cr = classification_report(labels, predictions, target_names=class_names)
    print(cr)
    if save_path is not None:
        save_path = Path(save_path)
        with open(save_path / 'classification_report.txt', mode='w') as f:
            f.write(cr)


def main():
    # command-line arguments
    args = get_command_line_arguments()
    dataset_path = args.data
    model_path = args.model
    output_path = Path(args.output)

    # prepare paths
    output_path.mkdir(parents=True, exist_ok=True)

    # prepare data
    train_set, test_set = timit.load_data(dataset_path)
    _, test_set = normalize(train_set, test_set)
    n_train_classes = get_number_of_classes()
    origin_to_train_label, train_label_to_test_label, test_label_to_test_name = timit.get_phone_mapping()
    n_test_classes = len(set(train_label_to_test_label.values()))
    x_test, y_test = prepare_data_for_model(test_set, n_train_classes, one_hot=False)
    n_features = x_test.shape[-1]

    # predict with model
    model = DualStudent(n_train_classes, n_features, padding_value=PADDING_VALUE)
    # model.load_weights(model_path)
    model.compile(metrics=['accuracy'])
    probabilities = model.predict(x_test)
    y_pred = np.argmax(probabilities, axis=-1)

    # evaluate
    y_true_concat = []
    y_pred_concat = []
    edit_distance = 0
    tot_frames = 0
    for i in range(y_test.shape[0]):
        # drop padding
        mask = np.array([(True if label < n_train_classes else False) for label in y_test[i]])
        sentence_true = y_test[i][mask]
        sentence_pred = y_pred[i][mask]

        # map training phones to test phones
        sentence_true = list(map_labels(train_label_to_test_label, sentence_true))
        sentence_pred = list(map_labels(train_label_to_test_label, sentence_pred))

        # update edit distance
        sm = SequenceMatcher(a=sentence_true, b=sentence_pred)
        tot_frames = tot_frames + len(sentence_true)
        edit_distance = edit_distance + sm.distance()

        # concatenate (for confusion matrix and classification report)
        y_true_concat += sentence_true
        y_pred_concat += sentence_pred

    # compute PER (phone error rate)
    per = edit_distance / tot_frames
    print('Phone Error Rate: {:.2f}%'.format(per * 100))
    with open(output_path / 'per.txt', mode='w') as f:
        f.write('Phone Error Rate: {:.2f}%'.format(per * 100))

    # compute confusion matrix and classification report
    y_true_concat = np.array(y_true_concat)
    y_pred_concat = np.array(y_pred_concat)
    class_names = [name for _, name in sorted(test_label_to_test_name.items())]
    plot_confusion_matrix(y_true_concat, y_pred_concat, class_names, save_path=output_path)
    make_classification_report(y_true_concat, y_pred_concat, class_names, save_path=output_path)


if __name__ == '__main__':
    main()
