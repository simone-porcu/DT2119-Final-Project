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
    print_sorted_dictionary(core_test_speakers, header='core test speakers', footer='')

    train_phone_labels, test_phone_labels = get_phone_mapping()
    print_sorted_dictionary(train_phone_labels, sort_by='value', header='train mapping', footer='')
    print_sorted_dictionary(test_phone_labels, sort_by='value', header='test mapping', footer='')

    path = os.path.join('timit', 'train', 'dr1', 'mwar0', 'sx415.wav')
    info = path_to_info(path)
    print('utterance info:', info, end='\n\n')

    (x_train, y_train), (x_test, y_test) = get_preprocessed_data(os.path.join(get_root_dir(), 'data', 'timit'),
                                                                 context=0)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
