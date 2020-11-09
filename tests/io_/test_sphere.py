import unittest
from dualstudent import get_root_dir
from dualstudent.io import load_sphere


class SphereTestCase(unittest.TestCase):
    def test_load_sphere(self):
        filepath = get_root_dir() / 'data' / 'timit' / 'train' / 'dr1' / 'mwar0' / 'sx415.wav'
        samples, sample_rate = load_sphere(filepath)
        self.assertEqual(len(samples.shape), 1)
        self.assertEqual(samples.shape[0], 38810)
        self.assertEqual(sample_rate, 16000)


if __name__ == '__main__':
    unittest.main()
