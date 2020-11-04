import os
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from dualstudent import get_root_dir
from dualstudent.speech import extract_features, extract_labels
from dualstudent.speech import load_sphere
from dualstudent.speech import load_transcription, get_number_of_frames

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

    The training and test phone subsets are chosen according to standard recipes for TIMIT.

    :return: tuple (phone_labels, evaluation_mapping), containing the described dictionaries.
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
    origin_to_train_label = {}
    for origin_phone in data_frame['origin']:
        train_phone = origin_to_train_phone[origin_phone]
        origin_to_train_label[origin_phone] = train_labels[train_phone]

    # get evaluation mapping (train label -> test label, to evaluate the model using a subset of phones)
    train_label_to_test_label = {}
    for origin_phone in data_frame['origin']:
        test_phone = origin_to_test_phone[origin_phone]
        train_label = origin_to_train_label[origin_phone]
        train_label_to_test_label[train_label] = test_labels[test_phone]

    return origin_to_train_label, train_label_to_test_label


def _preprocess_data(dataset_path, core_test=False):
    # get phone labels
    phone_labels, _ = get_phone_mapping()

    # get speakers in core test
    core_test_speakers = None   # we need them only if core_test=True, we initialize to None to avoid a warning
    if core_test:
        core_test_speakers = get_core_test_speakers()

    # prepare dataset
    dataset = []
    file_paths = list(dataset_path.glob('**/*.wav'))
    with tqdm(total=len(file_paths)) as bar:
        bar.set_description('Processing {}'.format(dataset_path))
        for filepath in file_paths:
            bar.update()
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
            features[:n_frames]     # the last frame may have the window not fully inside, we drop it
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
    :return: dictionary {'train': train_set, 'test': test_set}, where train_set and test_set are numpy arrays of
        utterances. Each utterance is a dictionary containing utterance info useful for normalization, feature vectors,
        and phone labels.
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


def split_val(dataset):
    """
    Split dataset in training and validation set.
    Create a validation set of unique sentences by using the table in spkrinfo_spkrsent.txt .
    There is no overlap of sentences in validion, and between train and validation.
    Complete validation contains all the speakers which say a sentence also said by someone in
    the validation set. Complete validation does not include the validatio.
    For further details, visit documentation in testset.doc and refers to spkrinfo_spkrsent.txt .
    :param dataset: dataset to be processed. np.array of dictionaries {‘dialect’,‘sex’,‘speaker_id’,‘features’,‘labels’}
    :return: (train, valid, complete_valid) in the same format of dataset
    """
    doc_path = get_root_dir() / 'data' / 'spkrinfo_spkrsent.txt'

    # shuffe data
    with open(doc_path, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()

    doc_path = doc_path.parent / (doc_path.stem + '_shuffled.txt')

    with open(doc_path, 'w') as target:
        for _, line in data:
            target.write(line)

    with open(doc_path) as f:
        next(f)
        next(f)
        drs = [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]  # [male, female] for each dialect
        new_speakers = []  # unique speakers
        sentence_ids = []  # unique speakers sentences

        for line in f:
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

    with open(doc_path) as f:
        next(f)
        next(f)
        pair_speakers = []  # unique speakers pairs (for the complete_valid_set)
        pair_sentence_ids = []  # unique speakers pairs sentences (for the complete_valid_set)

        for line in f:
            columns = line.split()
            condition = (not columns[4] in sentence_ids) and (not columns[5] in sentence_ids) and (
                not columns[6] in sentence_ids) and (not columns[7] in sentence_ids) and (
                            not columns[8] in sentence_ids)
            if ((columns[3] == 'TRN') and not condition):
                pair_speakers.append(columns[0])
                pair_sentence_ids.extend(
                    [columns[4], columns[5], columns[6], columns[7], columns[8], columns[9], columns[10], columns[11]])

    print(f"# of speakers: {len(new_speakers)}")
    print(f"# of sentences: {len(sentence_ids)}")
    print(f"# of pair_speakers: {len(pair_speakers)}")
    print(f"# of pair_sentences: {len(pair_sentence_ids)}")

    valid = []
    train = []
    complete_valid = []
    new_speakers = [x.lower() for x in new_speakers]
    pair_speakers = [x.lower() for x in pair_speakers]

    for utterance in dataset:
        if utterance['speaker_id'] in new_speakers:
            valid.append(utterance)
        if utterance['speaker_id'] in pair_speakers:
            complete_valid.append(utterance)
        if (not utterance['speaker_id'] in pair_speakers) and (not utterance['speaker_id'] in new_speakers):
            train.append(utterance)

    train = np.asarray(train)
    valid = np.asarray(valid)
    complete_valid = np.asarray(complete_valid)
    os.remove(doc_path)

    return train, valid, complete_valid


def split_val_random(dataset):
    """
    Ssplit the dataset into training and validation. Speakers for the validation set are randomly selected.
    :param dataset: dataset to be processed. np.array of dictionaries {'dialect','sex','speaker_id','features','labels'}
    :n_m: array containg the numbers of male speakers to use in the validation for each dialect.
    :n_f: array containg the numbers of female speakers to use in the validadion for each dialect.
    :return: (train, valid) where both sets are two numpy arrays
    """

    drs = [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]  # [male, female] for each dialect
    drs = np.asarray(drs)

    n_m = drs[:, 0]
    n_f = drs[:, 1]

    train = []
    valid = []

    for i in range(len(n_m)):

        mask_m = [(utterance['dialect'] == 'dr' + str(i + 1) and utterance['sex'] == 'm') for utterance in dataset]
        mask_f = [(utterance['dialect'] == 'dr' + str(i + 1) and utterance['sex'] == 'f') for utterance in dataset]

        m = dataset[mask_m]
        f = dataset[mask_f]

        speakers_m = list(set([item['speaker_id'] for item in m]))
        speakers_f = list(set([item['speaker_id'] for item in f]))

        # sampling males
        selected_speakers_m = np.random.choice(speakers_m, n_m[i])
        print('dr' + str(i + 1) + ', m:' + str(selected_speakers_m))

        # sampling females
        selected_speakers_f = np.random.choice(speakers_f, n_f[i])
        print('dr' + str(i + 1) + ', f:' + str(selected_speakers_f))

        # add speakers to train and validation sets
        for utterance in m:
            if utterance['speaker_id'] in selected_speakers_m:
                valid.append(utterance)
            else:
                train.append(utterance)

        for utterance in f:
            if utterance['speaker_id'] in selected_speakers_f:
                valid.append(utterance)
            else:
                train.append(utterance)

    train = np.asarray(train)
    valid = np.asarray(valid)

    return train, valid
