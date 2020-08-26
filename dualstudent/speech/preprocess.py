import numpy as np
import python_speech_features as pss
from intervaltree import IntervalTree
from sklearn.preprocessing import StandardScaler


def extract_features(samples, sample_rate, win_len, win_shift, win_fun=np.hamming):
    """
    Computes 13 MFCC + delta + delta-delta features for an utterance.

    :param samples: samples of the utterance, numpy array of shape (n_samples,)
    :param sample_rate: sampling rate
    :param win_len: window length (in seconds)
    :param win_shift: window shift (in seconds)
    :param win_fun: window function
    :return: numpy array of shape (n_frames, n_features), where n_features=39
    """
    mfcc = pss.mfcc(samples, sample_rate, winlen=win_len, winstep=win_shift, winfunc=win_fun)
    delta = pss.delta(mfcc, 3)
    delta_delta = pss.delta(delta, 3)
    return np.concatenate((mfcc, delta, delta_delta), axis=1)


def extract_labels(transcription, sample_rate, n_frames, win_len, win_shift):
    """
    Extracts the phone labels from a phone transcription. The mid-point is used to solve ambiguities (frames with more
    than one label).

    :param transcription: phone transcription, list of tuples (begin_sample, end_sample, label)
    :param sample_rate: sampling rate
    :param n_frames: number of frames of the utterance
    :param win_len: window length (in seconds)
    :param win_shift: window shift (in seconds)
    :return: list of length n_frames
    """
    # fill interval tree
    segments = IntervalTree()
    for segment in transcription:
        begin_sample = segment[0]
        end_sample = segment[1]
        label = segment[2]
        assert len(segments[begin_sample:end_sample]) == 0    # no overlaps
        segments[begin_sample:end_sample] = label

    # seconds -> samples
    win_len = round(win_len * sample_rate)
    win_shift = round(win_shift * sample_rate)

    # find labels of middle samples
    labels = []
    mid_sample = transcription[0][0] + int(win_len/2)
    for i in range(n_frames):
        labels.append(segments[mid_sample].pop().data)
        mid_sample += win_shift
    return labels


def stack_acoustic_context(features, n):
    """
    For each feature vector (frame), stack feature vectors on the left and on the right to get an acoustic context
    (dynamic features).

    :param features: original features, numpy array of shape (n_frames, n_features)
    :param n: how many features on the left and on the right to stack (acoustic context or dynamic features)
    :return: features with acoustic context, numpy array of shape (n_frames, context*n_features)
    """
    if n < 0 or n > features.shape[0]:
        raise ValueError('Invalid context size')
    if n == 0:
        return features
    length = features.shape[0]
    idx_list = list(range(length))
    idx_list = idx_list[1:1+n][::-1] + idx_list + idx_list[-1-n:-1][::-1]
    features = [features[idx_list[i:i+1+2*n]].reshape(-1) for i in range(length)]
    return np.array(features)


def normalize(train_set, test_set, mode='full'):
    """
    Normalizes the dataset according to the specified mode.

    :param train_set: list of utterances, each utterance is a dictionary containing utterance info useful for
        normalization, feature vectors, and phone labels.
    :param test_set: list of utterances, each utterance is a dictionary containing utterance info useful for
        normalization, feature vectors, and phone labels.
    :param mode: normalization mode. Support for: 'full', 'speaker', 'utterance'.
    :return: (train_set, test_set), normalized
    """
    if mode == 'full':
        # fit scaler
        x_train = np.concatenate([utterance['features'] for utterance in train_set])
        ss = StandardScaler()
        ss.fit(x_train)

        # normalize
        for utterance in train_set:
            utterance['features'] = ss.transform(utterance['features'])
        for utterance in test_set:
            utterance['features'] = ss.transform(utterance['features'])

    elif mode == 'speaker':
        # TODO
        raise NotImplementedError('Normalization mode ' + mode + ' not yet supported')
    elif mode == 'utterance':
        # TODO
        raise NotImplementedError('Normalization mode ' + mode + ' not yet supported')
    else:
        raise ValueError('Invalid normalization mode')

    return train_set, test_set
