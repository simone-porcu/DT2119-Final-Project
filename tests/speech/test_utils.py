import unittest
from dualstudent.speech.utils import *
from dualstudent.utils import get_root_dir


class UtilsTestCase(unittest.TestCase):
    def test_load_transcription(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'fcjf0' / 'sa1.phn'
        transcription = load_transcription(filepath)
        self.assertTupleEqual(transcription[0], (0, 3050, 'h#'))
        self.assertTupleEqual(transcription[5], (8772, 9190, 'dcl'))
        self.assertTupleEqual(transcription[10], (12640, 14714, 'ah'))
        self.assertTupleEqual(transcription[15], (20417, 21199, 'q'))
        self.assertTupleEqual(transcription[20], (24229, 25566, 'ix'))
        self.assertTupleEqual(transcription[25], (31719, 33360, 'sh'))
        self.assertTupleEqual(transcription[30], (36326, 37556, 'axr'))
        self.assertTupleEqual(transcription[36], (44586, 46720, 'h#'))

    def test_get_number_of_frames(self):
        n_frames = get_number_of_frames(49361, 16000, 0.03, 0.01)
        self.assertEqual(n_frames, 306)


if __name__ == '__main__':
    unittest.main()
