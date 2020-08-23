# TODO: see https://docs.python-guide.org/writing/tests/

from dualstudent.datasets.timit import *
from dualstudent.utils import get_root_dir


def print_sorted_dictionary(dictionary, sort_by='key', header=None, footer=None):
    if header is not None:
        print(header)

    if sort_by == 'key':
        key = 0
    elif sort_by == 'value':
        key = 1
    else:
        raise ValueError('invalid sort_by argument')

    for k, v in sorted(dictionary.items(), key=lambda e: e[key]):
        print(k, '=>', v)

    if footer is not None:
        print(footer)


if __name__ == '__main__':
    core_test_speakers = get_core_test_speakers()
    print_sorted_dictionary(core_test_speakers, header='Core test speakers', footer='')

    # get_phone_mapping()
    train_phone_labels, test_phone_labels = get_phone_mapping()
    print_sorted_dictionary(train_phone_labels, sort_by='value', header='Train phone mapping', footer='')
    print_sorted_dictionary(test_phone_labels, sort_by='value', header='Test phone mapping', footer='')

    # path_to_info()
    path = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
    info = path_to_info(path)
    print('Utterance info:', info, end='\n\n')

    # load_data()
    dataset = load_data(get_root_dir() / 'data' / 'timit')
    print('# training utterances:', len(dataset['train']))
    print('# test utterances:', len(dataset['test']))
    print('# train speakers', len({u['speaker_id'] for u in dataset['train']}))
    print('# test speakers', len({u['speaker_id'] for u in dataset['test']}))
    print('# training frames:', sum([len(u['features']) for u in dataset['train']]))
    print('# test frames:', sum([len(u['features']) for u in dataset['test']]))
