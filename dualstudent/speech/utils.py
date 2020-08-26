def load_transcription(filepath):
    """
    Loads the phonetic transcription of an utterance from a file.

    :param filepath: path to the transcription file (.phn)
    :return: list of tuples (begin_sample, end_sample, phone)
    """
    with filepath.open() as f:
        lines = f.read().split('\n')
    transcription = map(lambda line: line.split(' '), lines)
    transcription = filter(lambda segment: len(segment) == 3, transcription)    # remove invalid lines
    transcription = map(lambda segment: (int(segment[0]), int(segment[1]), segment[2]), transcription)
    transcription = list(transcription)
    return transcription


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
