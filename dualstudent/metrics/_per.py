import numpy as np
from tensorflow.keras.metrics import Metric
from edit_distance import SequenceMatcher


def _merge_consequent_states(y):
    cur_y = y[0]
    merged_y = [cur_y]
    for i in y:
        if i == cur_y:
            continue
        cur_y = i
        merged_y += [cur_y]
    return merged_y


def _map_labels(mapping, labels):
    return np.array([mapping[label] for label in labels])


class PhoneErrorRate(Metric):
    """
    Phone Error Rate (PER), i.e. the length-normalized edit distance between the predicted sequence of phones and the
    correct transcription.
    """

    def __init__(self, name='phone_error_rate', mapping=None, **kwargs):
        """
        Constructs the Phone Error Rate metric.

        :param name: name of the metric
        :param mapping: phone mapping for the evaluation, dictionary {training label -> test label}
        """
        super(PhoneErrorRate, self).__init__(name=name, **kwargs)
        self.edit_distance = self.add_weight(name='edit_distance', initializer='zeros')
        self.length = self.add_weight(name='length', initializer='zeros')
        self.mapping = mapping

    def update_state(self, y_true, y_pred):
        """
        Compute the edit distance and update the statistics (edit distance and length).

        :param y_true: numpy array of shape (n_frames,), ground truth for an utterance
        :param y_pred: numpy array of shape (n_frames,), predictions for an utterance
        """
        y_true = _map_labels(self.mapping, y_true)
        y_true = _merge_consequent_states(y_true)
        y_pred = _map_labels(self.mapping, y_true)
        y_pred = _merge_consequent_states(y_pred)
        sm = SequenceMatcher(a=y_true, b=y_pred)
        self.edit_distance.assign_add(sm.distance())
        self.length.assign_add(len(y_true))

    def result(self):
        return self.edit_distance / self.length
