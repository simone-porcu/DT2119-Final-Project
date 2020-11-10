import tensorflow as tf
from tensorflow.keras.metrics import Metric
from edit_distance import SequenceMatcher


def _merge_consequent_states(y):
    cur_state = y[0]
    merged_y = [cur_state]
    for state in y:
        if state == cur_state:
            continue
        cur_state = state
        merged_y.append(cur_state)
    return merged_y


class PhoneErrorRate(Metric):
    """
    Phone Error Rate (PER), i.e. the length-normalized edit distance between the predicted sequence of phones and the
    correct transcription.
    """

    def __init__(self, name='phone_error_rate', **kwargs):
        """
        Constructs the Phone Error Rate metric.

        :param name: name of the metric
        """
        super(PhoneErrorRate, self).__init__(name=name, **kwargs)
        self.edit_distance = self.add_weight(name='edit_distance', initializer='zeros')
        self.length = self.add_weight(name='length', initializer='zeros')

    def _update_state(self, y_true, y_pred, mask=None):
        for i in range(len(y_true)):
            assert len(y_true[i]) == len(y_pred[i])

            # select utterance
            y_true_ = y_true[i]
            y_pred_ = y_pred[i]

            # remove padding
            y_true_ = y_true_[mask[i]]
            y_pred_ = y_pred_[mask[i]]

            # merge consequence states
            y_true_ = _merge_consequent_states(y_true_)
            y_pred_ = _merge_consequent_states(y_pred_)

            # compute edit distance
            sm = SequenceMatcher(a=y_true_, b=y_pred_)
            edit_distance = sm.distance()

            # update state
            self.edit_distance.assign_add(edit_distance)
            self.length.assign_add(len(y_true_))

    def update_state(self, y_true, y_pred, mask=None):
        """
        Compute the edit distances and update the statistics (edit distance and length).

        :param y_true: tensor of shape (n_utterances, n_frames), ground truth (not one-hot)
        :param y_pred: tensor of shape (n_utterances, n_frames), predictions for an utterance (not one-hot)
        :param mask: mask for padding values
        """
        tf.numpy_function(self._update_state, [y_true, y_pred, mask], [])   # to work in graph mode

    def result(self):
        return self.edit_distance / self.length
