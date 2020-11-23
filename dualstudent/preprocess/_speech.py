import numpy as np
import python_speech_features as pss
from intervaltree import IntervalTree
from sklearn.preprocessing import StandardScaler
from operator import itemgetter      # for speaker normalization
from itertools import groupby        # for speaker normalization


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


def normalize(train_set, test_set=None, mode='full'):
    """
    Normalizes the dataset according to the specified mode.

    :param train_set: numpy array of utterances, each utterance is a dictionary containing utterance info useful for
        normalization, feature vectors, and phone labels.
    :param test_set: numpy array of utterances, each utterance is a dictionary containing utterance info useful for
        normalization, feature vectors, and phone labels.
    :param mode: normalization mode. Support for: 'full', 'speaker', 'utterance'.
    :return: tuple (train_set, test_set) if test_set is provided, otherwise train_set. The results are normalized.
    """
    if mode == 'full':
        # fit scaler
        x_train = np.concatenate([utterance['features'] for utterance in train_set])
        ss = StandardScaler()
        ss.fit(x_train)

        # normalize
        for utterance in train_set:
            utterance['features'] = ss.transform(utterance['features'])
        if test_set is not None:
            for utterance in test_set:
                utterance['features'] = ss.transform(utterance['features'])

    elif mode == 'speaker':
        for index, dataset in enumerate([train_set, test_set]):
            # split the set according to the speaker
            set_split = []
            grouper = itemgetter("speaker_id")
            for _, v in groupby(dataset, grouper):
                set_split.append(list(v))  # list of lists of dict, each sublist represent a speaker

            for speaker_set in set_split:
                # fit scaler
                x_train = np.concatenate([utterance['features'] for utterance in speaker_set])
                ss = StandardScaler()
                ss.fit(x_train)

                # normalize
                for utterance in speaker_set:
                    utterance['features'] = ss.transform(utterance['features'])

            # from the normalized list of lists recover a single list containing all the utterances
            aux = np.array([item for sublist in set_split for item in sublist])
            if index == 0:
                train_set = aux
            else:
                test_set = aux

    elif mode == 'utterance':
        for dataset in [train_set, test_set]:
            for utterance in dataset:
                # fit scaler
                x_train = utterance['features']
                ss = StandardScaler()
                ss.fit(x_train)

                # normalize
                utterance['features'] = ss.transform(utterance['features'])

    else:
        raise ValueError('Invalid normalization mode')

    return train_set if test_set is None else train_set, test_set


def unlabel(train_set, percentage, seed=None):
    """
    Removes the labels from a percentage of training frames. The percentage is computed at sample-level, not
    at utterance-level.

    :param train_set: numpy array of utterances, each utterance is a dictionary containing utterance info useful for
        normalization, feature vectors, and phone labels.
    :param percentage: percentage of samples to unlabel
    :return: train_set, with some utterances without labels (i.e. dictionary not containing 'labels')
    """
    if seed is not None:
        np.random.seed(seed)

    train_set = np.copy(train_set)                  # copy to avoid modifying original
    total = sum([len(utterance['labels']) for utterance in train_set])
    total_unlabeled = int(round((total * percentage)))
    shuffled_idx = np.arange(len(train_set))
    np.random.shuffle(shuffled_idx)

    n_unlabeled = 0
    for i in shuffled_idx:
        idx = shuffled_idx[i]                       # random index of utterance chosen to unlabel
        n_samples = len(train_set[idx]['labels'])
        del train_set[idx]['labels']                # unlabel
        n_unlabeled += n_samples
        if n_unlabeled >= total_unlabeled:          # percentage reached
            break

    return train_set
