import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


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
