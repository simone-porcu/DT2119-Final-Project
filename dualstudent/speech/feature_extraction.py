import numpy as np
import python_speech_features as pss
from intervaltree import IntervalTree


def extract_features(samples, sample_rate, win_len, win_shift, win_fun):
    """
    Computes 13 MFCC + delta + delta-delta features for an utterance.

    :param samples: samples of the utterance
    :param sample_rate: sampling rate
    :param win_len: window size (length)
    :param win_shift: window shift
    :param win_fun: window function (e.g. np.hamming)
    :return: np.array of shape (39,)
    """
    mfcc = pss.mfcc(samples, sample_rate, winlen=win_len, winstep=win_shift, winfunc=win_fun)
    delta = pss.delta(mfcc, 3)
    delta_delta = pss.delta(delta, 3)
    return np.hstack((mfcc, delta, delta_delta))


def extract_labels(transcription, n_frames, win_len, win_shift):
    """
    Extracts the labels from a phone transcription. The mid-point is used to solve ambiguities (frames with more than
    one label).

    :param transcription: phone transcription, tuple (begin_sample, end_sample, phone)
    :param n_frames: number of frames of the utterance
    :param win_len: window size (length)
    :param win_shift: window shift
    :return: np.array of shape (n_frames,)
    """
    # fill interval tree
    phones = IntervalTree()
    for segment in transcription:
        begin_sample = segment[0]
        end_sample = segment[1]
        label = segment[2]
        assert len(phones[begin_sample:end_sample]) == 0    # no overlaps
        phones[begin_sample:end_sample] = label

    # find labels
    labels = np.zeros(n_frames)
    mid_point = round(win_len / 2)      # use mid point to solve ambiguity (i.e. frame with multiple labels)
    for i in range(n_frames):
        aux = phones[mid_point]
        if len(aux) == 0:               # some samples can be removed (segment ignored)
            continue
        labels[i] = aux.pop().data
        mid_point += win_shift
    return labels
