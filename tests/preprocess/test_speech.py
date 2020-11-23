import unittest
import numpy as np
from dualstudent import get_root_dir
from dualstudent.preprocess import *
from dualstudent.datasets import timit
from dualstudent.io import load_sphere, load_transcription


class SpeechTestCase(unittest.TestCase):
    def test_extract_features(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        win_len = 0.03
        win_shift = 0.01
        samples, sample_rate = load_sphere(filepath)
        features = extract_features(samples, sample_rate, win_len, win_shift)
        n_frames = 1 + round((samples.shape[0] - win_len * sample_rate) / (win_shift * sample_rate))
        self.assertTrue(features.shape[0] - n_frames <= 1)

    def test_extract_labels(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        _, sample_rate = load_sphere(filepath)
        filepath = filepath.with_suffix('.phn')
        transcription = load_transcription(filepath)
        win_len = 0.03
        win_shift = 0.01
        n_frames = get_number_of_frames(38720, sample_rate, win_len, win_shift)
        labels = extract_labels(transcription, sample_rate, n_frames, win_len, win_shift)
        self.assertEqual(len(set(labels)), 25)
        self.assertEqual(len(labels), 240)

    def test_stack_acoustic_context(self):
        features = np.arange(6)
        features = stack_acoustic_context(features, 5)
        self.assertListEqual(list(features[0]), [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
        self.assertListEqual(list(features[1]), [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4])
        self.assertListEqual(list(features[2]), [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3])
        self.assertListEqual(list(features[3]), [2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2])
        self.assertListEqual(list(features[4]), [1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        self.assertListEqual(list(features[5]), [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])

    def test_normalize(self):
        dataset_path = get_root_dir() / 'data' / 'timit'
        train_set, _ = timit.load_data(dataset_path)

        # test normalization on whole dataset
        normalized_train_set, _ = normalize(train_set, mode='full')
        x_train = np.concatenate([utterance['features'] for utterance in normalized_train_set])
        mean = x_train.mean(axis=0)
        var = x_train.var(axis=0)
        for i in range(x_train.shape[1]):
            self.assertAlmostEqual(mean[i], 0)
            self.assertAlmostEqual(var[i], 1)

        # test normalization on
        # TODO: test other normalization modes

    def test_unlabel(self):
        dataset_path = get_root_dir() / 'data' / 'timit'
        train_set, _ = timit.load_data(dataset_path)
        n_total = len(train_set)

        unlabel(train_set, 0.7, seed=1)
        n_labeled = len([utterance for utterance in train_set if 'labels' in utterance])
        n_unlabeled = len([utterance for utterance in train_set if 'labels' not in utterance])

        self.assertEqual(n_labeled + n_unlabeled, n_total)
        self.assertTrue(n_labeled < n_unlabeled)
        self.assertEqual(n_labeled, 1104)
        self.assertEqual(n_unlabeled, 2592)


if __name__ == '__main__':
    unittest.main()
