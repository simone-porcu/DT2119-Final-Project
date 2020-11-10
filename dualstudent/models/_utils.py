import numpy as np


def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup.

    Source: https://github.com/ZHKKKe/DualStudent/blob/5e0c010c4cb7cafe0aff76f6511a22511c8e8ae8/third_party/mean_teacher/ramps.py#L19
    Original proposal: https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def select_batch(data, batch_idx, batch_size):
    """
    Selects batch of data selecting from the first dimension.

    :param data: tensor with more than 1 dimension, complete dataset
    :param batch_idx: integer, index of batch to select
    :param batch_size: integer, batch size
    :return: tensor, selected batch
    """
    idx_start = batch_idx * batch_size
    idx_end = (batch_idx+1) * batch_size
    batch = data[idx_start:idx_end]
    return batch


def map_labels(labels, mapping=None):
    """
    Maps labels according to a mapping.

    :param labels: numpy array of shape (n_utterances, n_frames), labels
    :param mapping: numpy array containing just one element, i.e. a dictionary {original label -> mapped label}.
        This is supposed to be a numpy array in order for this function to be called using tf.numpy_function().
    :return: numpy array of shape (n_utterances, n_frames), mapped labels
    """
    if mapping is not None:
        mapping = mapping[0]
        labels = np.copy(labels)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label = labels[i, j]
                if label in mapping:
                    label = mapping[label]
                labels[i, j] = label
    return labels
