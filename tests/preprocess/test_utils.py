import unittest
from dualstudent import get_root_dir
from dualstudent.preprocess import *


class UtilsTestCase(unittest.TestCase):
    def test_get_number_of_frames(self):
        n_frames = get_number_of_frames(49361, 16000, 0.03, 0.01)
        self.assertEqual(n_frames, 306)


if __name__ == '__main__':
    unittest.main()
