import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dualstudent import get_root_dir
from dualstudent.preprocess import extract_features, extract_labels, get_number_of_frames
from dualstudent.io import load_sphere, load_transcription

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
    return {
        'file_type': path.suffix,
        'text_type': path.stem[:2],
        'sentence_number': int(path.stem[2:]),
        'sex': path.parts[-2][0],
        'speaker_id': path.parts[-2][1:],
        'dialect': path.parts[-3],
        'usage': path.parts[-4]
    }


def get_core_test_speakers():
    """
    Returns a dictionary (dialect -> list of speaker_id) for the core test set.

    :return: dictionary (dialect -> list of speaker_id)
    """
    filepath = get_root_dir() / 'data' / 'timit_core_test.json'
    with filepath.open() as json_file:
        return json.load(json_file)


def get_phone_mapping():
    """
    Generates:
    - dictionary (origin phone -> train label), to load targets for the model from transcriptions. Different phones can
        be mapped to the same label, as a subset of phones is used for training (48 phones).
    - dictionary (train label -> test label), to evaluate the model on a subset of the training phones (39 phones).
    - dictionary (test label -> test phone), to print the names (e.g. in confusion matrix)

    The training and test phone subsets are chosen according to standard recipes for TIMIT.

    :return: tuple (phone_labels, evaluation_mapping, test_label_to_test_phone), containing the described dictionaries.
    """
    # read file
    filepath = get_root_dir() / 'data' / 'timit_phones_60-48-39.map'
    with filepath.open() as csv_file:
        data_frame = pd.read_csv(csv_file, sep='\t')
    data_frame = data_frame.dropna()

    # load phone mappings
    origin_to_train_phone = {op: tp for op, tp in zip(data_frame['origin'], data_frame['train'])}
    origin_to_test_phone = {op: tp for op, tp in zip(data_frame['origin'], data_frame['test'])}

    # generate labels (sorting in order to be sure that multiple calls generate always the same dictionaries)
    train_labels = {phone: label for label, phone in enumerate(sorted(data_frame['train'].unique()))}
    test_labels = {phone: label for label, phone in enumerate(sorted(data_frame['test'].unique()))}

    # get phone labels (origin phone -> train label, to generate targets from transcriptions)
    origin_phone_to_train_label = {}
    for origin_phone in data_frame['origin']:
        train_phone = origin_to_train_phone[origin_phone]
        origin_phone_to_train_label[origin_phone] = train_labels[train_phone]

    # get evaluation mapping (train label -> test label, to evaluate the model using a subset of phones)
    train_label_to_test_label = {}
    for origin_phone in data_frame['origin']:
        test_phone = origin_to_test_phone[origin_phone]
        train_label = origin_phone_to_train_label[origin_phone]
        train_label_to_test_label[train_label] = test_labels[test_phone]

    # get test class names (for confusion matrix)
    test_label_to_test_phone = {value: key for key, value in test_labels.items()}

    return origin_phone_to_train_label, train_label_to_test_label, test_label_to_test_phone


def _preprocess_data(dataset_path, core_test=False):
    # get phone labels
    phone_labels, _, _ = get_phone_mapping()

    # get speakers in core test
    core_test_speakers = None   # we need them only if core_test=True, we initialize to None to avoid a warning
    if core_test:
        core_test_speakers = get_core_test_speakers()

    dataset = []
    file_paths = list(dataset_path.glob('**/*.wav'))
    for filepath in tqdm(file_paths, desc='Processing {}'.format(dataset_path)):
        info = path_to_info(filepath)

        # check sentence and speaker
        if info['text_type'] == 'sa':
            continue
        if core_test and not info['speaker_id'] in core_test_speakers[info['dialect']]:
            continue

        # load audio and transcription
        samples, sample_rate = load_sphere(filepath)
        filepath = filepath.with_suffix('.phn')
        transcription = load_transcription(filepath)

        # drop leading and trailing samples not in the transcription
        samples = samples[transcription[0][0]:transcription[-1][1]]

        # extract features and labels
        features = extract_features(samples, sample_rate, WIN_LEN, WIN_SHIFT)
        n_frames = get_number_of_frames(samples.shape[0], sample_rate, WIN_LEN, WIN_SHIFT)
        assert features.shape[0] - n_frames <= 1
        features = features[:n_frames]     # the last frame may have the window not fully inside, we drop it
        labels = extract_labels(transcription, sample_rate, n_frames, WIN_LEN, WIN_SHIFT)

        # drop frames with ignored phones as target (glottal stop /q/)
        labels = np.array([(phone_labels[label] if label in phone_labels else -1) for label in labels])
        valid_idx = np.where(labels != -1)[0]
        features = features[valid_idx]
        labels = labels[valid_idx]

        # add to dataset
        dataset.append({
            'dialect': info['dialect'],
            'sex': info['sex'],
            'speaker_id': info['speaker_id'],
            'features': features,
            'labels': labels
        })

    return np.array(dataset)


def load_data(dataset_path, core_test=True, force_preprocess=False):
    """
    Returns training and test set containing features (13 MFCC + delta + delta-delta) and labels (phones encoded as
    integers).

    The split in training and test sets is the recommended one (see timit/readme.doc and timit/doc/testset.doc).

    :param dataset_path: path to the dataset. Since the TIMIT dataset is protected by copyright, it is not distributed
        with the package.
    :param core_test: whether to use the core test set (see timit/doc/testset.doc) instead of the complete test set
    :param force_preprocess: force to pre-process again, even if saved data can be loaded
    :return: tuple (train_set, test_set), where train_set and test_set are numpy arrays of utterances. Each utterance
        is a dictionary containing utterance info useful for normalization, feature vectors, and phone labels.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise ValueError('Invalid dataset path')

    # training set
    filepath = get_root_dir() / 'data' / 'timit_train.npz'
    if filepath.is_file() and not force_preprocess:
        print('Loading training set...', end=' ')
        train_set = np.load(filepath, allow_pickle=True)['train_set']
        print('done')
    else:
        train_set = _preprocess_data(dataset_path / 'train')
        np.savez(filepath, train_set=train_set)

    # test set
    filepath = get_root_dir() / 'data' / ('timit_' + ('core_' if core_test else '') + 'test.npz')
    if filepath.is_file() and not force_preprocess:
        print('Loading test set...', end=' ')
        test_set = np.load(filepath, allow_pickle=True)['test_set']
        print('done')
    else:
        test_set = _preprocess_data(dataset_path / 'test', core_test)
        np.savez(filepath, test_set=test_set)

    return train_set, test_set


def _split_validation_unique(train_set):
    doc_path = get_root_dir() / 'data' / 'spkrinfo_spkrsent.txt'

    # shuffle data
    with open(doc_path, 'r') as source:
        lines = [line for line in source]
        lines[-1] += '\n'
        lines = np.array(lines)
        np.random.shuffle(lines)

    drs = [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]  # [male, female] for each dialect
    new_speakers = []  # unique speakers
    sentence_ids = []  # unique speakers sentences

    for line in lines:
        columns = line.split()
        condition = (not columns[4] in sentence_ids) and (not columns[5] in sentence_ids) and (
            not columns[6] in sentence_ids) and (not columns[7] in sentence_ids) and (
                        not columns[8] in sentence_ids)
        if columns[3] == 'TRN' and condition:  # never seen a speaker saying this sentence
            if drs[int(columns[2]) - 1][0] != 0 and columns[1] == 'M':  # if males is not filled
                sentence_ids.extend(
                    [columns[4], columns[5], columns[6], columns[7], columns[8], columns[9], columns[10],
                     columns[11]])
                new_speakers.append(columns[0])
                drs[int(columns[2]) - 1][0] -= 1

            elif (drs[int(columns[2]) - 1][1] != 0) and columns[1] == 'F':
                sentence_ids.extend(
                    [columns[4], columns[5], columns[6], columns[7], columns[8], columns[9], columns[10],
                     columns[11]])
                new_speakers.append(columns[0])
                drs[int(columns[2]) - 1][1] -= 1

    pair_speakers = []  # unique speakers pairs (for the complete_valid_set)
    pair_sentence_ids = []  # unique speakers pairs sentences (for the complete_valid_set)

    for line in lines:
        columns = line.split()
        condition = (not columns[4] in sentence_ids) and (not columns[5] in sentence_ids) and (
            not columns[6] in sentence_ids) and (not columns[7] in sentence_ids) and (
                        not columns[8] in sentence_ids)
        if columns[3] == 'TRN' and not condition:
            pair_speakers.append(columns[0])
            pair_sentence_ids.extend(
                [columns[4], columns[5], columns[6], columns[7], columns[8], columns[9], columns[10], columns[11]])

    valid = []
    train = []
    complete_valid = []
    new_speakers = [x.lower() for x in new_speakers]
    pair_speakers = [x.lower() for x in pair_speakers]

    for utterance in train_set:
        if utterance['speaker_id'] in new_speakers:
            valid.append(utterance)
        if utterance['speaker_id'] in pair_speakers:
            complete_valid.append(utterance)
        if (not utterance['speaker_id'] in pair_speakers) and (not utterance['speaker_id'] in new_speakers):
            train.append(utterance)

    train = np.asarray(train)
    valid = np.asarray(valid)
    complete_valid = np.asarray(complete_valid)

    return train, valid, complete_valid


def _split_validation_random(train_set):
    drs = [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]  # [male, female] for each dialect
    drs = np.array(drs)
    n_m = drs[:, 0]     # numbers of male speakers to use in the validation for each dialect
    n_f = drs[:, 1]     # numbers of female speakers to use in the validation for each dialect
    train_set_new = []
    val_set = []

    for i in range(len(n_m)):
        mask_m = [(utterance['dialect'] == 'dr' + str(i + 1) and utterance['sex'] == 'm') for utterance in train_set]
        mask_f = [(utterance['dialect'] == 'dr' + str(i + 1) and utterance['sex'] == 'f') for utterance in train_set]
        m = train_set[mask_m]
        f = train_set[mask_f]

        # sort in order to have always the same order (otherwise even seeding we get different results)
        speakers_m = sorted(list({item['speaker_id'] for item in m}))
        speakers_f = sorted(list({item['speaker_id'] for item in f}))

        # sampling males
        selected_speakers_m = np.random.choice(speakers_m, n_m[i])

        # sampling females
        selected_speakers_f = np.random.choice(speakers_f, n_f[i])

        # add speakers to train and validation sets
        for utterance in m:
            if utterance['speaker_id'] in selected_speakers_m:
                val_set.append(utterance)
            else:
                train_set_new.append(utterance)

        for utterance in f:
            if utterance['speaker_id'] in selected_speakers_f:
                val_set.append(utterance)
            else:
                train_set_new.append(utterance)

    train_set_new = np.array(train_set_new)
    val_set = np.array(val_set)

    return train_set_new, val_set


def split_validation(train_set, mode='random', seed=None):
    """
    Split the training set in training and validation set.

    It supports the following modes:
    - 'unique': create a validation set of unique sentences. There is no overlap of sentences in validation and training
        sets. Complete validation contains all the speakers which say a sentence pronounced also by someone in the
        validation set. Complete validation does not include the validation. For further details, visit documentation
        in testset.doc and refers to spkrinfo_spkrsent.txt.
    - 'random': speakers for the validation set are randomly selected.

    :param train_set: numpy array of dictionaries as returned by load_data()
    :param mode: one of 'unique' and 'random'
    :param seed: seed for random number generator
    :return: (train_set, val_set) in the same format of the input dataset
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == 'random':
        train_set, val_set = _split_validation_random(train_set)
    elif mode == 'unique':
        train_set, val_set, _ = _split_validation_unique(train_set)
    else:
        raise ValueError('Invalid mode')
    return train_set, val_set
