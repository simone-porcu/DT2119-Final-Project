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
    idx_start = batch_idx * batch_size
    idx_end = (batch_idx+1) * batch_size
    batch = data[idx_start:idx_end]
    return batch
