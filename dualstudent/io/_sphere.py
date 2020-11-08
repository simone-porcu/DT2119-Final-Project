import numpy as np
from pysndfile import sndio


def load_sphere(filepath):
    """
    Loads the utterance samples from a file.

    Source: lab3 of DT2119 Speech and Speaker Recognition at KTH, by prof. Giampiero Salvi (slightly modified)

    :param filepath: path to the utterance file (.wav)
    :return: (samples, sample rate), where samples is a numpy array of shape (n_samples,)
    """
    snd_obj = sndio.read(filepath, dtype=np.int16)
    samples = np.array(snd_obj[0])
    sample_rate = snd_obj[1]
    return samples, sample_rate
