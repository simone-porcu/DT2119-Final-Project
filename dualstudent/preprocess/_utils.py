def get_number_of_frames(n_samples, sample_rate, win_len, win_shift):
    """
    Returns the number of frames for which the window is fully contained.

    :param n_samples: number of samples
    :param sample_rate: sampling rate
    :param win_len: window length (in seconds)
    :param win_shift: window shift (in seconds)
    :return: number of frames
    """
    win_len = round(win_len * sample_rate)
    win_shift = round(win_shift * sample_rate)
    return 1 + int((n_samples - win_len) / win_shift)
