import unittest
from dualstudent.speech.preprocess import *
from dualstudent.speech.sphere import load_audio
from dualstudent.speech.utils import load_transcription
from dualstudent.datasets.timit import load_data, get_number_of_frames
from dualstudent.utils import get_root_dir


class PreprocessTestCase(unittest.TestCase):
    def test_extract_features(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        win_len = 0.03
        win_shift = 0.01
        samples, sample_rate = load_audio(filepath)
        features = extract_features(samples, sample_rate, win_len, win_shift)
        n_frames = 1 + round((samples.shape[0] - win_len * sample_rate) / (win_shift * sample_rate))
        self.assertTrue(features.shape[0] - n_frames <= 1)

    def test_extract_labels(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        _, sample_rate = load_audio(filepath)
        filepath = filepath.with_suffix('.phn')
        transcription = load_transcription(filepath)
        win_len = 0.03
        win_shift = 0.01
        n_frames = get_number_of_frames(38720, sample_rate, win_len, win_shift)
        labels = extract_labels(transcription, sample_rate, n_frames, win_len, win_shift)
        self.assertEqual(len(set(labels)), 25)
        self.assertEqual(len(labels), 240)

    def test_normalize(self):
        dataset_path = get_root_dir() / 'data' / 'timit'
        train_set, test_set = load_data(dataset_path)
        train_set, test_set = normalize(train_set, test_set, mode='full')

        x_train = np.concatenate([utterance['features'] for utterance in train_set])
        mean = x_train.mean(axis=0)
        var = x_train.var(axis=0)
        for i in range(x_train.shape[1]):
            self.assertAlmostEqual(mean[i], 0)
            self.assertAlmostEqual(var[i], 1)

        self.assertRaises(NotImplementedError, normalize, train_set, test_set, mode='speaker')
        self.assertRaises(NotImplementedError, normalize, train_set, test_set, mode='utterance')


if __name__ == '__main__':
    unittest.main()
