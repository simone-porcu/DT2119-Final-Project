from dualstudent.speech.preprocess import *
from dualstudent.datasets.timit import load_data
from dualstudent.utils import get_root_dir


if __name__ == '__main__':
    dataset = load_data(get_root_dir() / 'data' / 'timit')

    # normalize()
    print('Before normalizing')
    print(dataset)
    dataset = normalize(dataset)
    print('After normalizing')
    print(dataset)
