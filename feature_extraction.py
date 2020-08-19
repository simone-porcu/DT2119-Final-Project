import os
import numpy as np
import tensorflow as tf
import python_speech_features as pss
from pysndfile import sndio
from intervaltree import IntervalTree

# TODO: convert to package (?), global variable -> __all__ mechanism (see PEP8)

# paths
root_dir = 'data/'
dataset_dir = os.path.join(root_dir, 'timit')

# TODO: set values
win_len = 0.025
win_shift = 0.01


def path_to_info(path):
    """
    Extracts the information about an utterance of the TIMIT dataset starting from its path.

    Path format: /<CORPUS>/<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>
    Example: timit/train/dr1/mwar0/sx415.wav

    See timit/readme.doc for an explanation of each field.

    :param path: path the utterance
    :return: dictionary with utterance information
    """
    path, aux = os.path.split(path)
    sentence_id, file_type = aux.split(sep='.')
    path, aux = os.path.split(path)
    sex = aux[:1]
    speaker_id = aux[1:]
    path, dialect = os.path.split(path)
    path, usage = os.path.split(path)
    return {'usage': usage, 'dialect': dialect, 'sex': sex, 'speaker_id': speaker_id, 'sentence_id': sentence_id,
            'file_type': file_type}


def load_audio(path):
    """
    Loads the utterance samples from a file.

    Source: lab3 of DT2119 Speech and Speaker Recognition at KTH, by prof. Giampiero Salvi (slightly modified)

    :param path: path to the file
    :return: (samples, sample rate)
    """
    snd_obj = sndio.read(path, dtype=np.int16)
    samples = np.array(snd_obj[0])
    sample_rate = snd_obj[1]
    return samples, sample_rate


def extract_features(path):
    """
    Computes MFCC + delta + delta-delta features for an utterance.

    :param path: path to the file
    :return: features
    """
    # load audio
    samples, sample_rate = load_audio(path)

    # compute features
    mfcc = pss.mfcc(samples, sample_rate, winlen=win_len, winstep=win_shift, winfunc=np.hamming)
    delta = pss.delta(mfcc, 3)
    delta_delta = pss.delta(delta, 3)

    # stack features
    features = np.hstack((mfcc, delta, delta_delta))

    return features


def extract_labels(path, n_windows):
    # TODO: add mapping 61->48
    # TODO: add mapping phone string -> int

    # load transcription
    with open(path, 'r') as file:
        lines = file.readlines()

    # fill interval tree
    phones = IntervalTree()
    for line in lines:
        fields = line[:-1].split(sep=' ')   # :-1 to remove ending new line
        begin_sample = int(fields[0])
        end_sample = int(fields[1])
        phone = fields[2]
        assert len(phones[begin_sample:end_sample]) == 0    # no overlaps
        phones[begin_sample:end_sample] = phone

    # find labels
    labels = np.zeros(n_windows)
    mid_point = round(win_len / 2)      # use mid point to solve ambiguity (i.e. frame with multiple labels)
    for i in range(n_windows):
        aux = phones[mid_point]
        assert len(aux) == 1            # one and only one interval
        labels[i] = aux.pop().data      # TODO: now crashes here, needed mapping string -> int for labels
        mid_point += win_shift

    return labels


def generate_dataset(dataset_dir, core_test=False):
    dataset = {}

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            filepath = os.path.join(root, file)

            if core_test:
                pass    # TODO: more filtering

            # TODO: are we sure that SA ones are not used even in training set?
            if file.endswith('.wav') and not file.startswith('sa'):
                dataset[filepath] = {}

                # features
                features = extract_features(filepath)
                dataset[filepath]['features'] = features

                # labels
                filepath = filepath.split('.')[0] + '.phn'
                dataset[filepath]['labels'] = extract_labels(filepath, len(features))

    return dataset


def get_datasets():
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    train_file = 'train_data.npz'
    test_file = 'test_data.npz'

    # training data
    if os.path.exists(train_file):
        train_set = np.load(train_file)
    else:
        train_set = generate_dataset(train_dir)
        np.savez(train_file, train_set)

    # test data
    if os.path.exists(test_file):
        test_set = np.load(test_file)
    else:
        test_set = generate_dataset(test_dir)
        np.savez(test_file, test_set)

    train_set = tf.data.Dataset.from_tensors([])    # TODO
    test_set = tf.data.Dataset.from_tensors([])     # TODO
    # TODO: validation set (with tensorflow!)
    return train_set, test_set


if __name__ == '__main__':
    get_datasets()









# # # phone_map_tsv is the 60-48-39.map file
# # def phoneMapping(phone_map_tsv):
# #     df = pd.read_csv(phone_map_tsv, sep="\t", index_col=0)
# #     df = df.dropna()
# #     df = df.drop('eval', axis=1)
# #     train_phn_idx = {k: i for i, k in enumerate(df['train'].unique())}
# #     df['train_idx'] = df['train'].map(train_phn_idx)
# #     phone_to_idx = df['train_idx'].to_dict()
# #
# #     # train_phn_idx: mapping from 48 phonemes to 48 idx
# #     # phone_to_idx:  mapping from 61 phonemes to 48 idx
# #     return train_phn_idx, phone_to_idx
