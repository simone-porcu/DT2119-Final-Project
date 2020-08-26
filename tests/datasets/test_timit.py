import unittest
from dualstudent.datasets.timit import *
from dualstudent.utils import get_root_dir


class TimitTestCase(unittest.TestCase):
    def test_get_core_test_speakers(self):
        core_test_speakers = get_core_test_speakers()
        self.assertEqual(len(core_test_speakers.keys()), 8)
        self.assertEqual(sum([len(speakers) for speakers in core_test_speakers.values()]), 24)

    def test_get_phone_mapping(self):
        phone_labels, evaluation_mapping = get_phone_mapping()
        self.assertEqual(len(set(phone_labels.keys())), 60)
        self.assertEqual(len(set(phone_labels.values())), 48)
        self.assertEqual(len(set(evaluation_mapping.keys())), 48)
        self.assertEqual(len(set(evaluation_mapping.values())), 39)
        self.assertListEqual(sorted(set(phone_labels.values())), list(range(48)))
        self.assertListEqual(sorted(evaluation_mapping.keys()), list(range(48)))
        self.assertListEqual(sorted(set(evaluation_mapping.values())), list(range(39)))

    def test_path_to_info(self):
        path = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        info = path_to_info(path)
        self.assertEqual(info['usage'], 'train')
        self.assertEqual(info['dialect'], 'dr1')
        self.assertEqual(info['sex'], 'm')
        self.assertEqual(info['speaker_id'], 'war0')
        self.assertEqual(info['text_type'], 'sx')
        self.assertEqual(info['sentence_number'], 415)
        self.assertEqual(info['file_type'], '.wav')

    def test_load_data(self):
        dataset_path = get_root_dir() / 'data' / 'timit'
        for force_preprocess in [True, False]:
            train_set, test_set = load_data(dataset_path, force_preprocess=force_preprocess)

            # number of utterances
            self.assertEqual(len(train_set), 3696)
            self.assertEqual(len(test_set), 192)

            # number of speakers
            self.assertEqual(len({utterance['speaker_id'] for utterance in train_set}), 462)
            self.assertEqual(len({utterance['speaker_id'] for utterance in test_set}), 24)

            # number of frames
            self.assertEqual(sum([len(utterance['features']) for utterance in train_set]), 1104675)
            self.assertEqual(sum([len(utterance['features']) for utterance in test_set]), 56623)
            self.assertEqual(sum([len(utterance['labels']) for utterance in train_set]), 1104675)
            self.assertEqual(sum([len(utterance['labels']) for utterance in test_set]), 56623)

            # shape of features and labels
            for utterance in np.concatenate((train_set, test_set)):
                self.assertEqual(len(utterance['features'].shape), 2)
                self.assertEqual(len(utterance['labels'].shape), 1)
                self.assertEqual(utterance['features'].shape[1], 39)


if __name__ == '__main__':
    unittest.main()
