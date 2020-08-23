import json
import pandas as pd
import numpy as np
from dualstudent.utils import get_root_dir
from dualstudent.speech.preprocess import extract_features, extract_labels
from dualstudent.speech.sphere import load_audio
from dualstudent.speech.utils import load_transcription, get_number_of_frames

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
    filepath = get_root_dir() / 'data' / 'timit_core_test_set.json'
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
    for filepath in dataset_path.glob('**/*.wav'):
        info = path_to_info(filepath)

        # check sentence and speaker
        if info['text_type'] == 'sa':
            continue
        if core_test and not info['speaker_id'] in core_test_speakers[info['dialect']]:
            continue

        # load audio and transcription
        print('Processing ', filepath, '...', sep='', end=' ')
        samples, sample_rate = load_audio(filepath)
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
        print('done')

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
    if not dataset_path.is_dir():
        raise ValueError('Invalid dataset path')

    # training set
    filepath = get_root_dir() / 'data' / 'timit_train.npz'
    if filepath.is_file() and not force_preprocess:
        print('Loading training set...', end=' ')
        train_set = np.load(filepath, allow_pickle=True)['train_set']
        print('done')
    else:
        print('Preparing training set...')
        train_set = _preprocess_data(dataset_path / 'train')
        np.savez(filepath, train_set=train_set)

    # test set
    filepath = get_root_dir() / 'data' / ('timit_' + ('core_' if core_test else '') + 'test.npz')
    if filepath.is_file() and not force_preprocess:
        print('Loading test set...', end=' ')
        test_set = np.load(filepath, allow_pickle=True)['test_set']
        print('done')
    else:
        print('Preparing test set...')
        test_set = _preprocess_data(dataset_path / 'test', core_test)
        np.savez(filepath, test_set=test_set)

    return train_set, test_set
