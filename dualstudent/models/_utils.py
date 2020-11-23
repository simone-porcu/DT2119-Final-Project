import numpy as np


def sigmoid_rampup(current, length):
    """
    Exponential rampup.

    Source: https://github.com/ZHKKKe/DualStudent/
    Original proposal: https://arxiv.org/abs/1610.02242

    :param current: integer or numpy array, current epoch
    :param length: integer or numpy array, length of ramp (in epochs)
    :return: float, sigmoid value in range [0,1]
    """
    if length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, length)
        phase = 1.0 - current / length
        return np.exp(-5.0 * phase * phase)


def sinusoidal_cycling(current, length):
    """
    Sinusoidal schedule. In the first half cycle, the values increase from 0 to 1, while after that the values follow a
    periodic sinusoidal trend in the range [0.5, 1].

    :param current: integer, current epoch
    :param length: integer, length of half cycle (in epochs)
    :return: float, sinusoidal value as described above
    """
    length *= 2
    return np.where(
        current < length/2,
        1 / 2 + np.cos(current / length * 2 * np.pi + np.pi) / 2,
        3 / 4 + np.cos((current % length) / length * 2 * np.pi + np.pi) / 4
    )


def triangular_cycling(current, length):
    """
    Triangular schedule. In the first half cycle, the values increase from 0 to 1, while after that the values follow a
    periodic linear trend in the range [0.5, 1].

    :param current: integer, current epoch
    :param length: integer, length of half cycle (in epochs)
    :return: float, linear value as described above
    """
    length *= 2
    return np.where(
        np.remainder(current, length) < length/2,
        np.where(
            current < length / 2,
            np.remainder(current, length / 2) / (length / 2),
            np.remainder(current, length / 2) / length + 0.5
        ),
        (1 - np.remainder(current, length) / length) + 0.5,
    )


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
        mapping = mapping.tolist()
        labels = np.copy(labels)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label = labels[i, j]
                if label in mapping:
                    label = mapping[label]
                labels[i, j] = label
    return labels
