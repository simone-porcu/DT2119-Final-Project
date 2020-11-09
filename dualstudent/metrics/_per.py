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
        Compute the edit distances and update the statistics (edit distance and length).

        :param y_true: numpy array of variable-length numpy arrays, ground truth for an utterance (not one-hot)
        :param y_pred: numpy array of variable-length numpy arrays, predictions for an utterance (not one-hot)
        """
        assert len(y_true) == len(y_pred)
        for i in range(len(y_true)):
            assert y_true[i] == y_pred[i]

            y_true = _merge_consequent_states(y_true)
            y_pred = _merge_consequent_states(y_pred)
            sm = SequenceMatcher(a=y_true, b=y_pred)

            self.edit_distance.assign_add(sm.distance())
            self.length.assign_add(len(y_true))

    def result(self):
        return self.edit_distance / self.length
