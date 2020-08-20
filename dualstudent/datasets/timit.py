import os
import json
import pandas as pd
import numpy as np
from dualstudent.utils import get_root_dir
from dualstudent.speech.feature_extraction import extract_features, extract_labels
from dualstudent.speech.sphere import load_audio

WIN_LEN = 0.03
WIN_SHIFT = 0.01


def path_to_info(path):
    """
    Extracts the information about an utterance starting from its path.

    Path format: .../<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<TEXT_TYPE><SENTENCE_NUMBER>.<FILE_TYPE>
    Example: .../train/dr1/mwar0/sx415.wav

    See timit/readme.doc for an explanation of each field.

    :param path: path the utterance file
    :return: dictionary with utterance information
    """
    path = path.lower()
    path, aux = os.path.split(path)
    aux, file_type = aux.split(sep='.')
    text_type = aux[0:2]
    path, aux = os.path.split(path)
    sex = aux[:1]
    speaker_id = aux[1:]
    path, dialect = os.path.split(path)
    path, usage = os.path.split(path)
    return {'usage': usage, 'dialect': dialect, 'sex': sex, 'speaker_id': speaker_id, 'text_type': text_type,
            'file_type': file_type}


def get_core_test_speakers():
    """
    Returns a dictionary (dialect -> list of speaker_id) for the core test set.

    :return: dictionary (dialect -> list of speaker_id)
    """
    filepath = os.path.join(get_root_dir(), 'data', 'timit_core_test_set.json')
    with open(filepath) as json_file:
        core_test_speakers = json.load(json_file)
    core_test_speakers = {dialect.lower(): [speaker_id.lower() for speaker_id in speaker_ids]
                          for dialect, speaker_ids in core_test_speakers.items()}
    return core_test_speakers


def _get_phone_mapping(data_frame, origin, new):
    phone_mapping = {op: tp for op, tp in zip(data_frame[origin], data_frame[new])}
    phone_labels = {phone: label for label, phone in enumerate(data_frame[new].unique())}
    for origin_phone in data_frame[origin]:
        new_phone = phone_mapping[origin_phone]
        phone_labels[origin_phone] = phone_labels[new_phone]
    return phone_labels


def get_phone_mapping():
    """
    Generates training and test labels. The generated labels are integers. The full set of phones of size 60 is
    reduced to 48 for training and 39 for evaluation, according to standard recipes.

    :return: tuple of dictionaries (phone -> label) for training and test set
    """
    filepath = os.path.join(get_root_dir(), 'data', 'timit_phones_60-48-39.map')
    with open(filepath) as csv_file:
        data_frame = pd.read_csv(csv_file, sep="\t")
    data_frame = data_frame.dropna()
    train_phone_labels = _get_phone_mapping(data_frame, 'origin', 'train')
    test_phone_labels = _get_phone_mapping(data_frame, 'origin', 'test')
    return train_phone_labels, test_phone_labels


def _extract_labels(filepath, n_frames, phone_labels):
    with open(filepath) as f:
        lines = f.readlines()
    transcription = map(lambda line: line[:-1].split(sep=' '), lines)   # :-1 to remove ending new line
    transcription = filter(lambda segment: segment[2] in phone_labels, transcription)   # filter out 'q' (glottal stop)
    transcription = map(lambda segment: (int(segment[0]), int(segment[1]), phone_labels[segment[2]]), transcription)
    return extract_labels(transcription, n_frames, WIN_LEN, WIN_SHIFT)


def _load_and_preprocess_data(dataset_path, core_test=True, normalization='full'):
    if core_test:
        core_test_speakers = get_core_test_speakers()
    train_phone_labels = get_phone_mapping()[0]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            filepath = os.path.join(root, file)
            info = path_to_info(filepath)

            # check speaker
            if info['usage'] == 'test' and core_test and not info['speaker_id'] in core_test_speakers[info['dialect']]:
                continue

            # check file
            if info['file_type'] != 'wav' or info['text_type'] == 'sa':
                continue

            print('processing ', filepath, '...', sep='', end=' ')

            # extract features
            samples, sample_rate = load_audio(filepath)
            x = extract_features(samples, sample_rate, WIN_LEN, WIN_SHIFT, np.hamming)

            # extract labels
            filepath = filepath.split('.')[0] + '.phn'
            y = _extract_labels(filepath, len(x), train_phone_labels)

            # append
            if info['usage'] == 'train':
                x_train.append(x)
                y_train.append(y)
            elif info['usage'] == 'test':
                x_test.append(x)
                y_test.append(y)
            else:
                raise ValueError('TIMIT dataset contains an invalid path')

            print('done')

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    # TODO: normalization (here)
    # TODO: tf.data.Dataset and stacking features (not here, in models/)

    return (x_train, y_train), (x_test, y_test)


def get_preprocessed_data(dataset_path, core_test=True, normalization='full'):
    """
    Returns the preprocessed dataset as features (39 coefficients, MFCC + delta + delta-delta) and labels (integers).
    The split in training and test sets is the recommended one (see timit/readme.doc and timit/doc/testset.doc).

    :param dataset_path: path to the dataset. Since the TIMIT dataset is protected by copyright, it is not distributed
        with the package.
    :param core_test: whether to use the core test set (see timit/doc/testset.doc) instead of the complete test set
    :param normalization: type of normalization. Support for: 'full', 'speaker', 'utterance'
    :return: (x_train, y_train), (x_test, y_test)
    """
    preprocessed_file = os.path.join(get_root_dir(), 'data', 'preprocessed_' + normalization + '_normalized_data.npz')
    if os.path.exists(preprocessed_file):
        print(preprocessed_file, 'found, loading...', end=' ')
        with np.load(preprocessed_file) as preprocessed_data:
            x_train = preprocessed_data['x_train']
            y_train = preprocessed_data['y_train']
            x_test = preprocessed_data['x_test']
            y_test = preprocessed_data['y_test']
        print('done')
    else:
        print(preprocessed_file, 'not found, starting to preprocess...')
        (x_train, y_train), (x_test, y_test) = _load_and_preprocess_data(dataset_path, core_test, normalization)
        np.savez(preprocessed_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return (x_train, y_train), (x_test, y_test)
